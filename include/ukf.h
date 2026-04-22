#pragma once

#include "pose_utils.h"
#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace ukf {

class PoseUKF {
public:
    static constexpr int kStateDim = 13;
    static constexpr int kErrorDim = 12;
    static constexpr int kMeasDim = 6;
    static constexpr int kSigmaCount = 2 * kErrorDim + 1;

    using StateVec = Eigen::Matrix<double, kStateDim, 1>;
    using StateMat = Eigen::Matrix<double, kErrorDim, kErrorDim>;
    using ErrorVec = Eigen::Matrix<double, kErrorDim, 1>;
    using MeasVec = Eigen::Matrix<double, kMeasDim, 1>;
    using MeasMat = Eigen::Matrix<double, kMeasDim, kMeasDim>;
    using CrossMat = Eigen::Matrix<double, kErrorDim, kMeasDim>;

    enum class Status {
        kOk = 0,
        kNotInitialized,
        kInvalidArgument,
        kRejectedMeasurement,
        kTimestampRegression,
        kSequenceRegression,
        kNumericalFailure,
    };

    struct Statistics {
        std::size_t predict_count{0};
        std::size_t accepted_update_count{0};
        std::size_t rejected_update_count{0};
        std::size_t stale_update_count{0};
        std::size_t invalid_update_count{0};
        std::size_t numerical_failure_count{0};
    };

    PoseUKF();

    bool initialize(const PoseMeasurement& m, double pos_var = 1e-3, double ori_var = 1e-3);
    void setProcessNoise(double pos_noise, double vel_noise, double ori_noise, double ang_noise);
    void setMeasurementNoise(double pos_noise, double ori_noise);
    void setMahalanobisGate(double nis_threshold);
    void setCovarianceFloor(double variance_floor);

    bool predict(double dt);
    bool update(const PoseMeasurement& m);

    const StateVec& state() const { return x_; }
    const StateMat& covariance() const { return P_; }
    bool initialized() const { return initialized_; }
    Status lastStatus() const { return status_; }
    const char* lastStatusMessage() const;
    const Statistics& statistics() const { return stats_; }
    double lastNormalizedInnovationSquared() const { return last_nis_; }

    Vec3 position() const;
    Vec3 velocity() const;
    Quaternion orientation() const;
    Vec3 angularVelocity() const;

private:
    StateVec x_;
    StateMat P_;
    StateMat Q_;
    MeasMat R_;
    double alpha_;
    double beta_;
    double kappa_;
    double lambda_;
    double mahalanobis_gate_;
    double covariance_floor_;
    double last_timestamp_;
    double latest_observation_timestamp_;
    double last_nis_;
    std::uint64_t last_sequence_;
    bool initialized_;
    bool has_timestamp_;
    bool has_sequence_;
    Status status_;
    Statistics stats_;

    struct MeasurementSample {
        Vec3 position{Vec3::Zero()};
        Quaternion orientation{Quaternion::Identity()};
    };

    using SigmaSet = std::array<StateVec, kSigmaCount>;
    using MeasSigmaSet = std::array<MeasurementSample, kSigmaCount>;

    bool generateSigmaPoints(SigmaSet& sigma);
    static bool validateMeasurement(const PoseMeasurement& m);
    static MeasurementSample makeMeasurementSample(const PoseMeasurement& m);
    MeasMat effectiveMeasurementNoise(const PoseMeasurement& m) const;
    bool setStatus(Status status, bool ok);
    void stabilizeCovariance(StateMat& P) const;
    void stabilizeMeasurementCovariance(MeasMat& R) const;
    void computeWeights(Eigen::Matrix<double, kSigmaCount, 1>& wm,
                        Eigen::Matrix<double, kSigmaCount, 1>& wc) const;
    static void normalizeState(StateVec& x);
    static StateVec processModel(const StateVec& x, double dt);
    static MeasurementSample measurementModel(const StateVec& x);
    static StateVec applyError(const StateVec& x, const ErrorVec& dx);
    static ErrorVec stateDifference(const StateVec& ref, const StateVec& target);
    static MeasVec measurementDifference(const MeasurementSample& ref, const MeasurementSample& target);
    static StateVec meanFromSigma(const SigmaSet& sigma, const Eigen::Matrix<double, kSigmaCount, 1>& wm);
    static StateMat covarianceFromSigma(const SigmaSet& sigma, const StateVec& mean, const Eigen::Matrix<double, kSigmaCount, 1>& wc);
    static MeasurementSample measurementMeanFromSigma(const MeasSigmaSet& sigma, const Eigen::Matrix<double, kSigmaCount, 1>& wm);
    static MeasMat measurementCovFromSigma(const MeasSigmaSet& sigma, const MeasurementSample& mean, const Eigen::Matrix<double, kSigmaCount, 1>& wc);
    static CrossMat crossCovariance(const SigmaSet& xs, const StateVec& x_mean,
                                    const MeasSigmaSet& zs, const MeasurementSample& z_mean,
                                    const Eigen::Matrix<double, kSigmaCount, 1>& wc);
};

} // namespace ukf
