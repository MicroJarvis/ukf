#include "ukf.h"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ukf {

namespace {

constexpr double kQuaternionNormFloor = 1e-12;
constexpr double kCovarianceJitterBase = 1e-12;
constexpr int kCovarianceJitterTries = 6;
constexpr double kTimestampTolerance = 1e-9;

template <typename MatrixType>
void projectToPositiveSemidefinite(MatrixType& matrix, double variance_floor) {
    matrix = 0.5 * (matrix + matrix.transpose());

    Eigen::SelfAdjointEigenSolver<MatrixType> eigensolver(matrix);
    if (eigensolver.info() != Eigen::Success || !eigensolver.eigenvalues().array().isFinite().all()) {
        matrix = variance_floor * MatrixType::Identity();
        return;
    }

    const auto eigenvectors = eigensolver.eigenvectors();
    const auto eigenvalues = eigensolver.eigenvalues().cwiseMax(variance_floor);
    matrix = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    matrix = 0.5 * (matrix + matrix.transpose());
}

Quaternion averageQuaternion(const std::vector<Quaternion>& quaternions,
                             const Eigen::VectorXd& weights) {
    if (quaternions.empty()) {
        return Quaternion::Identity();
    }

    const Quaternion reference = normalizeQuaternion(quaternions.front());
    Eigen::Vector4d accumulator = Eigen::Vector4d::Zero();
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(quaternions.size()); ++i) {
        const Quaternion aligned = alignQuaternionHemisphere(quaternions[static_cast<std::size_t>(i)], reference);
        accumulator += weights(i) * Eigen::Vector4d(aligned.w(), aligned.x(), aligned.y(), aligned.z());
    }

    Quaternion mean(accumulator(0), accumulator(1), accumulator(2), accumulator(3));
    if (mean.norm() < kQuaternionNormFloor) {
        return reference;
    }
    return normalizeQuaternion(mean);
}

} // namespace

PoseUKF::PoseUKF()
    : alpha_(1e-1),
      beta_(2.0),
      kappa_(0.0),
      mahalanobis_gate_(25.0),
      covariance_floor_(1e-10),
      last_timestamp_(0.0),
      latest_observation_timestamp_(0.0),
      last_nis_(0.0),
      last_sequence_(0),
      initialized_(false),
      has_timestamp_(false),
      has_sequence_(false),
      status_(Status::kNotInitialized) {
    x_.setZero();
    x_(6) = 1.0;
    P_.setIdentity();
    P_ *= 1e-2;
    Q_.setIdentity();
    Q_ *= 1e-4;
    R_.setIdentity();
    R_ *= 1e-3;
    lambda_ = alpha_ * alpha_ * (kErrorDim + kappa_) - kErrorDim;
    stabilizeCovariance(P_);
    stabilizeCovariance(Q_);
    stabilizeMeasurementCovariance(R_);
}

bool PoseUKF::initialize(const PoseMeasurement& m, double pos_var, double ori_var) {
    if (!validateMeasurement(m) || !std::isfinite(pos_var) || !std::isfinite(ori_var) ||
        pos_var <= 0.0 || ori_var <= 0.0) {
        initialized_ = false;
        return setStatus(Status::kInvalidArgument, false);
    }

    x_.setZero();
    x_.segment<3>(0) = m.position;
    const Quaternion q = normalizeQuaternion(m.orientation);
    x_(6) = q.w();
    x_(7) = q.x();
    x_(8) = q.y();
    x_(9) = q.z();
    x_.segment<3>(10).setZero();

    P_.setZero();
    P_.block<3, 3>(0, 0) = pos_var * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(3, 3) = 1e-2 * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(6, 6) = ori_var * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(9, 9) = 1e-2 * Eigen::Matrix3d::Identity();
    stabilizeCovariance(P_);
    normalizeState(x_);
    stats_ = {};
    last_nis_ = 0.0;
    has_timestamp_ = true;
    last_timestamp_ = m.timestamp;
    latest_observation_timestamp_ = m.timestamp;
    has_sequence_ = m.sequence != 0;
    last_sequence_ = m.sequence;
    initialized_ = true;
    return setStatus(Status::kOk, true);
}

void PoseUKF::setProcessNoise(double pos_noise, double vel_noise, double ori_noise, double ang_noise) {
    Q_.setZero();
    Q_.block<3, 3>(0, 0) = std::max(pos_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(3, 3) = std::max(vel_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(6, 6) = std::max(ori_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    Q_.block<3, 3>(9, 9) = std::max(ang_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    stabilizeCovariance(Q_);
}

void PoseUKF::setMeasurementNoise(double pos_noise, double ori_noise) {
    R_.setZero();
    R_.block<3, 3>(0, 0) = std::max(pos_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    R_.block<3, 3>(3, 3) = std::max(ori_noise, covariance_floor_) * Eigen::Matrix3d::Identity();
    stabilizeMeasurementCovariance(R_);
}

void PoseUKF::setMahalanobisGate(double nis_threshold) {
    mahalanobis_gate_ = nis_threshold > 0.0 ? nis_threshold : std::numeric_limits<double>::infinity();
}

void PoseUKF::setCovarianceFloor(double variance_floor) {
    if (std::isfinite(variance_floor) && variance_floor > 0.0) {
        covariance_floor_ = variance_floor;
        stabilizeCovariance(P_);
        stabilizeCovariance(Q_);
        stabilizeMeasurementCovariance(R_);
    }
}

const char* PoseUKF::lastStatusMessage() const {
    switch (status_) {
        case Status::kOk:
            return "ok";
        case Status::kNotInitialized:
            return "filter is not initialized";
        case Status::kInvalidArgument:
            return "invalid argument";
        case Status::kRejectedMeasurement:
            return "measurement rejected by gate";
        case Status::kTimestampRegression:
            return "measurement timestamp regressed";
        case Status::kSequenceRegression:
            return "measurement sequence regressed";
        case Status::kNumericalFailure:
            return "numerical failure";
    }
    return "unknown status";
}

bool PoseUKF::predict(double dt) {
    if (!initialized_) {
        return setStatus(Status::kNotInitialized, false);
    }
    if (!std::isfinite(dt) || dt <= 0.0) {
        return setStatus(Status::kInvalidArgument, false);
    }

    SigmaSet sigma;
    if (!generateSigmaPoints(sigma)) {
        return false;
    }

    SigmaSet pred_sigma;
    for (int i = 0; i < kSigmaCount; ++i) {
        pred_sigma[i] = processModel(sigma[static_cast<std::size_t>(i)], dt);
    }

    Eigen::Matrix<double, kSigmaCount, 1> wm;
    Eigen::Matrix<double, kSigmaCount, 1> wc;
    computeWeights(wm, wc);

    x_ = meanFromSigma(pred_sigma, wm);
    P_ = covarianceFromSigma(pred_sigma, x_, wc) + Q_;
    normalizeState(x_);
    stabilizeCovariance(P_);
    ++stats_.predict_count;
    return setStatus(Status::kOk, true);
}

bool PoseUKF::update(const PoseMeasurement& m) {
    if (!initialized_) {
        return setStatus(Status::kNotInitialized, false);
    }
    if (!validateMeasurement(m)) {
        ++stats_.invalid_update_count;
        return setStatus(Status::kInvalidArgument, false);
    }
    if (has_timestamp_ && m.timestamp <= latest_observation_timestamp_ + kTimestampTolerance) {
        ++stats_.stale_update_count;
        ++stats_.rejected_update_count;
        return setStatus(Status::kTimestampRegression, false);
    }
    if (m.sequence != 0) {
        if (has_sequence_ && m.sequence <= last_sequence_) {
            ++stats_.stale_update_count;
            ++stats_.rejected_update_count;
            return setStatus(Status::kSequenceRegression, false);
        }
        has_sequence_ = true;
    }

    latest_observation_timestamp_ = m.timestamp;

    SigmaSet sigma;
    if (!generateSigmaPoints(sigma)) {
        return false;
    }

    MeasSigmaSet z_sigma{};
    for (int i = 0; i < kSigmaCount; ++i) {
        z_sigma[static_cast<std::size_t>(i)] = measurementModel(sigma[static_cast<std::size_t>(i)]);
    }

    Eigen::Matrix<double, kSigmaCount, 1> wm;
    Eigen::Matrix<double, kSigmaCount, 1> wc;
    computeWeights(wm, wc);

    const MeasurementSample z_mean = measurementMeanFromSigma(z_sigma, wm);
    MeasMat S = measurementCovFromSigma(z_sigma, z_mean, wc) + effectiveMeasurementNoise(m);
    stabilizeMeasurementCovariance(S);

    const CrossMat Pxz = crossCovariance(sigma, x_, z_sigma, z_mean, wc);
    const MeasVec innovation = measurementDifference(z_mean, makeMeasurementSample(m));

    Eigen::LDLT<MeasMat> ldlt(S);
    if (ldlt.info() != Eigen::Success) {
        ++stats_.numerical_failure_count;
        return setStatus(Status::kNumericalFailure, false);
    }

    const MeasVec solved_innovation = ldlt.solve(innovation);
    if (ldlt.info() != Eigen::Success || !solved_innovation.array().isFinite().all()) {
        ++stats_.numerical_failure_count;
        return setStatus(Status::kNumericalFailure, false);
    }

    const double nis = innovation.dot(solved_innovation);
    last_nis_ = nis;
    if (std::isfinite(mahalanobis_gate_) && nis > mahalanobis_gate_) {
        ++stats_.rejected_update_count;
        return setStatus(Status::kRejectedMeasurement, false);
    }

    const ErrorVec dx = Pxz * solved_innovation;
    if (!dx.array().isFinite().all()) {
        ++stats_.numerical_failure_count;
        return setStatus(Status::kNumericalFailure, false);
    }

    x_ = applyError(x_, dx);

    const StateMat correction = Pxz * ldlt.solve(Pxz.transpose());
    if (!correction.array().isFinite().all()) {
        ++stats_.numerical_failure_count;
        return setStatus(Status::kNumericalFailure, false);
    }

    P_ -= correction;
    normalizeState(x_);
    stabilizeCovariance(P_);
    last_timestamp_ = m.timestamp;
    last_sequence_ = m.sequence;
    ++stats_.accepted_update_count;
    return setStatus(Status::kOk, true);
}

bool PoseUKF::generateSigmaPoints(SigmaSet& sigma) {
    stabilizeCovariance(P_);

    const double c = kErrorDim + lambda_;
    StateMat scaled = c * P_;
    Eigen::LLT<StateMat> llt;
    double jitter = kCovarianceJitterBase;
    bool factorized = false;
    for (int attempt = 0; attempt < kCovarianceJitterTries; ++attempt) {
        llt.compute(scaled);
        if (llt.info() == Eigen::Success) {
            factorized = true;
            break;
        }
        scaled.diagonal().array() += jitter;
        jitter *= 10.0;
    }

    if (!factorized) {
        ++stats_.numerical_failure_count;
        return setStatus(Status::kNumericalFailure, false);
    }

    const StateMat A = llt.matrixL();
    sigma[0] = x_;
    for (int i = 0; i < kErrorDim; ++i) {
        const ErrorVec delta = A.col(i);
        sigma[static_cast<std::size_t>(i + 1)] = applyError(x_, delta);
        sigma[static_cast<std::size_t>(i + 1 + kErrorDim)] = applyError(x_, -delta);
    }
    return true;
}

bool PoseUKF::validateMeasurement(const PoseMeasurement& m) {
    return m.valid && isFiniteVector(m.position) && isFiniteQuaternion(m.orientation) &&
           m.orientation.norm() >= kQuaternionNormFloor && std::isfinite(m.timestamp) &&
           (!m.has_covariance || isFiniteMatrix(m.covariance));
}

PoseUKF::MeasurementSample PoseUKF::makeMeasurementSample(const PoseMeasurement& m) {
    MeasurementSample sample;
    sample.position = m.position;
    sample.orientation = normalizeQuaternion(m.orientation);
    return sample;
}

PoseUKF::MeasMat PoseUKF::effectiveMeasurementNoise(const PoseMeasurement& m) const {
    MeasMat effective = R_;
    if (m.has_covariance) {
        effective += m.covariance;
    }
    stabilizeMeasurementCovariance(effective);
    return effective;
}

bool PoseUKF::setStatus(Status status, bool ok) {
    status_ = status;
    return ok;
}

void PoseUKF::stabilizeCovariance(StateMat& P) const {
    projectToPositiveSemidefinite(P, covariance_floor_);
}

void PoseUKF::stabilizeMeasurementCovariance(MeasMat& R) const {
    projectToPositiveSemidefinite(R, covariance_floor_);
}

void PoseUKF::computeWeights(Eigen::Matrix<double, kSigmaCount, 1>& wm,
                             Eigen::Matrix<double, kSigmaCount, 1>& wc) const {
    const double c = kErrorDim + lambda_;
    wm.setConstant(0.5 / c);
    wc.setConstant(0.5 / c);
    wm(0) = lambda_ / c;
    wc(0) = lambda_ / c + (1.0 - alpha_ * alpha_ + beta_);
}

void PoseUKF::normalizeState(StateVec& x) {
    const Quaternion q = normalizeQuaternion(Quaternion(x(6), x(7), x(8), x(9)));
    x(6) = q.w();
    x(7) = q.x();
    x(8) = q.y();
    x(9) = q.z();
}

PoseUKF::StateVec PoseUKF::processModel(const StateVec& x, double dt) {
    StateVec y = x;
    y.segment<3>(0) += x.segment<3>(3) * dt;

    const Quaternion q = normalizeQuaternion(Quaternion(x(6), x(7), x(8), x(9)));
    const Vec3 omega = x.segment<3>(10);
    const Quaternion dq = deltaQuatFromOmega(omega, dt);
    const Quaternion qn = normalizeQuaternion(q * dq);
    y(6) = qn.w();
    y(7) = qn.x();
    y(8) = qn.y();
    y(9) = qn.z();
    return y;
}

PoseUKF::MeasurementSample PoseUKF::measurementModel(const StateVec& x) {
    MeasurementSample sample;
    sample.position = x.segment<3>(0);
    sample.orientation = normalizeQuaternion(Quaternion(x(6), x(7), x(8), x(9)));
    return sample;
}

PoseUKF::StateVec PoseUKF::applyError(const StateVec& x, const ErrorVec& dx) {
    StateVec y = x;
    y.segment<3>(0) += dx.segment<3>(0);
    y.segment<3>(3) += dx.segment<3>(3);
    const Quaternion q = normalizeQuaternion(Quaternion(x(6), x(7), x(8), x(9)));
    const Quaternion dq = quatExp(dx.segment<3>(6));
    const Quaternion qn = normalizeQuaternion(q * dq);
    y(6) = qn.w();
    y(7) = qn.x();
    y(8) = qn.y();
    y(9) = qn.z();
    y.segment<3>(10) += dx.segment<3>(9);
    return y;
}

PoseUKF::ErrorVec PoseUKF::stateDifference(const StateVec& ref, const StateVec& target) {
    ErrorVec d = ErrorVec::Zero();
    d.segment<3>(0) = target.segment<3>(0) - ref.segment<3>(0);
    d.segment<3>(3) = target.segment<3>(3) - ref.segment<3>(3);
    d.segment<3>(6) = rotationVectorError(
        Quaternion(ref(6), ref(7), ref(8), ref(9)),
        Quaternion(target(6), target(7), target(8), target(9)));
    d.segment<3>(9) = target.segment<3>(10) - ref.segment<3>(10);
    return d;
}

PoseUKF::MeasVec PoseUKF::measurementDifference(const MeasurementSample& ref,
                                                const MeasurementSample& target) {
    MeasVec d = MeasVec::Zero();
    d.segment<3>(0) = target.position - ref.position;
    d.segment<3>(3) = rotationVectorError(ref.orientation, target.orientation);
    return d;
}

PoseUKF::StateVec PoseUKF::meanFromSigma(
    const SigmaSet& sigma, const Eigen::Matrix<double, kSigmaCount, 1>& wm) {
    StateVec mean = StateVec::Zero();
    std::vector<Quaternion> quaternions;
    quaternions.reserve(kSigmaCount);

    for (int i = 0; i < kSigmaCount; ++i) {
        const StateVec& sample = sigma[static_cast<std::size_t>(i)];
        mean.segment<3>(0) += wm(i) * sample.segment<3>(0);
        mean.segment<3>(3) += wm(i) * sample.segment<3>(3);
        mean.segment<3>(10) += wm(i) * sample.segment<3>(10);
        quaternions.push_back(Quaternion(sample(6), sample(7), sample(8), sample(9)));
    }

    const Quaternion q = averageQuaternion(quaternions, wm);
    mean(6) = q.w();
    mean(7) = q.x();
    mean(8) = q.y();
    mean(9) = q.z();
    return mean;
}

PoseUKF::StateMat PoseUKF::covarianceFromSigma(
    const SigmaSet& sigma, const StateVec& mean,
    const Eigen::Matrix<double, kSigmaCount, 1>& wc) {
    StateMat P = StateMat::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        const ErrorVec d = stateDifference(mean, sigma[static_cast<std::size_t>(i)]);
        P += wc(i) * d * d.transpose();
    }
    return 0.5 * (P + P.transpose());
}

PoseUKF::MeasurementSample PoseUKF::measurementMeanFromSigma(
    const MeasSigmaSet& sigma, const Eigen::Matrix<double, kSigmaCount, 1>& wm) {
    MeasurementSample mean;
    std::vector<Quaternion> quaternions;
    quaternions.reserve(kSigmaCount);

    for (int i = 0; i < kSigmaCount; ++i) {
        mean.position += wm(i) * sigma[static_cast<std::size_t>(i)].position;
        quaternions.push_back(sigma[static_cast<std::size_t>(i)].orientation);
    }

    mean.orientation = averageQuaternion(quaternions, wm);
    return mean;
}

PoseUKF::MeasMat PoseUKF::measurementCovFromSigma(
    const MeasSigmaSet& sigma, const MeasurementSample& mean,
    const Eigen::Matrix<double, kSigmaCount, 1>& wc) {
    MeasMat S = MeasMat::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        const MeasVec d = measurementDifference(mean, sigma[static_cast<std::size_t>(i)]);
        S += wc(i) * d * d.transpose();
    }
    return 0.5 * (S + S.transpose());
}

PoseUKF::CrossMat PoseUKF::crossCovariance(
    const SigmaSet& xs, const StateVec& x_mean, const MeasSigmaSet& zs,
    const MeasurementSample& z_mean, const Eigen::Matrix<double, kSigmaCount, 1>& wc) {
    CrossMat Pxz = CrossMat::Zero();
    for (int i = 0; i < kSigmaCount; ++i) {
        const ErrorVec dx = stateDifference(x_mean, xs[static_cast<std::size_t>(i)]);
        const MeasVec dz = measurementDifference(z_mean, zs[static_cast<std::size_t>(i)]);
        Pxz += wc(i) * dx * dz.transpose();
    }
    return Pxz;
}

Vec3 PoseUKF::position() const { return x_.segment<3>(0); }

Vec3 PoseUKF::velocity() const { return x_.segment<3>(3); }

Quaternion PoseUKF::orientation() const {
    return normalizeQuaternion(Quaternion(x_(6), x_(7), x_(8), x_(9)));
}

Vec3 PoseUKF::angularVelocity() const { return x_.segment<3>(10); }

} // namespace ukf
