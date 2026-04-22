#include "ukf.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ukf::PoseMeasurement makeMeasurement(double timestamp,
                                     const ukf::Vec3& position = ukf::Vec3::Zero(),
                                     const ukf::Quaternion& orientation = ukf::Quaternion::Identity()) {
    ukf::PoseMeasurement measurement;
    measurement.timestamp = timestamp;
    measurement.position = position;
    measurement.orientation = orientation;
    return measurement;
}

ukf::Mat6 diagonalCovariance(double pos_var, double ori_var) {
    ukf::Mat6 covariance = ukf::Mat6::Zero();
    covariance.block<3, 3>(0, 0) = pos_var * ukf::Mat3::Identity();
    covariance.block<3, 3>(3, 3) = ori_var * ukf::Mat3::Identity();
    return covariance;
}

void testRejectsInvalidInitialization() {
    ukf::PoseUKF filter;
    ukf::PoseMeasurement invalid = makeMeasurement(
        0.0, ukf::Vec3(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
    require(!filter.initialize(invalid), "initialize should reject NaN position");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kInvalidArgument,
            "invalid initialize should set invalid-argument status");
}

void testRejectsPredictBeforeInitialize() {
    ukf::PoseUKF filter;
    require(!filter.predict(0.01), "predict should fail before initialize");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kNotInitialized,
            "predict before init should set not-initialized");
}

void testRejectsInvalidPredictStep() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    require(!filter.predict(-0.1), "predict should reject negative dt");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kInvalidArgument,
            "negative dt should be invalid argument");
}

void testQuaternionHemisphereConsistency() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");

    ukf::PoseMeasurement flipped = makeMeasurement(0.01);
    flipped.orientation = ukf::Quaternion(-1.0, 0.0, 0.0, 0.0);
    require(filter.update(flipped), "update should accept hemisphere-flipped quaternion");

    const ukf::Quaternion q = filter.orientation();
    require(std::abs(q.w()) > 0.999, "orientation should remain near identity");
}

void testMahalanobisGateRejectsOutlier() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    filter.setMahalanobisGate(9.0);
    filter.setMeasurementNoise(1e-4, 1e-4);
    require(filter.predict(0.01), "predict should succeed");

    const ukf::Vec3 before = filter.position();
    ukf::PoseMeasurement outlier = makeMeasurement(0.01, ukf::Vec3(10.0, 0.0, 0.0));
    require(!filter.update(outlier), "outlier should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kRejectedMeasurement,
            "outlier should set rejected-measurement status");
    require((filter.position() - before).norm() < 1e-9,
            "rejected outlier must not move state");
    require(filter.statistics().rejected_update_count == 1,
            "rejected update count should increment");
    require(filter.lastNormalizedInnovationSquared() > 9.0,
            "NIS should reflect the outlier");
}

void testPerMeasurementCovarianceDownweightsUncertainFrame() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    filter.setMahalanobisGate(9.0);
    filter.setMeasurementNoise(1e-6, 1e-6);
    require(filter.predict(0.01), "predict should succeed");

    ukf::PoseMeasurement uncertain = makeMeasurement(0.01, ukf::Vec3(1.0, 0.0, 0.0));
    uncertain.has_covariance = true;
    uncertain.covariance = diagonalCovariance(100.0, 10.0);
    require(filter.update(uncertain), "uncertain frame should be accepted");
    require(filter.position().x() < 0.05,
            "large per-frame covariance should keep the correction small");
    require(filter.lastNormalizedInnovationSquared() < 9.0,
            "large covariance should soften the innovation gate");
}

void testTimestampRegressionRejected() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(1.0)), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");
    require(filter.update(makeMeasurement(1.01, ukf::Vec3(0.001, 0.0, 0.0))),
            "first update should succeed");
    require(!filter.update(makeMeasurement(1.0, ukf::Vec3(0.002, 0.0, 0.0))),
            "timestamp regression should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kTimestampRegression,
            "timestamp regression should set dedicated status");
}

void testDuplicateTimestampRejected() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(1.0)), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");
    require(filter.update(makeMeasurement(1.01, ukf::Vec3(0.001, 0.0, 0.0))),
            "first update should succeed");
    require(!filter.update(makeMeasurement(1.01, ukf::Vec3(0.002, 0.0, 0.0))),
            "duplicate timestamp should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kTimestampRegression,
            "duplicate timestamp should use timestamp-regression status");
}

void testSequenceRegressionRejected() {
    ukf::PoseUKF filter;
    ukf::PoseMeasurement initial = makeMeasurement(1.0);
    initial.sequence = 10;
    require(filter.initialize(initial), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");

    ukf::PoseMeasurement accepted = makeMeasurement(1.01, ukf::Vec3(0.001, 0.0, 0.0));
    accepted.sequence = 11;
    require(filter.update(accepted), "monotonic sequence should succeed");

    ukf::PoseMeasurement stale = makeMeasurement(1.02, ukf::Vec3(0.002, 0.0, 0.0));
    stale.sequence = 11;
    require(!filter.update(stale), "duplicate sequence should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kSequenceRegression,
            "duplicate sequence should set sequence-regression status");
}

void testOlderFrameRejectedAfterFutureOutlier() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    filter.setMahalanobisGate(4.0);
    filter.setMeasurementNoise(1e-4, 1e-4);
    require(filter.predict(0.01), "predict should succeed");

    ukf::PoseMeasurement future_outlier = makeMeasurement(0.02, ukf::Vec3(10.0, 0.0, 0.0));
    require(!filter.update(future_outlier), "future outlier should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kRejectedMeasurement,
            "future outlier should be gate-rejected");

    require(!filter.update(makeMeasurement(0.015, ukf::Vec3(0.001, 0.0, 0.0))),
            "older frame after a newer one should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kTimestampRegression,
            "late frame after future outlier should be rejected by timestamp ordering");
}

void testRejectsInvalidPerFrameCovariance() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");

    ukf::PoseMeasurement invalid = makeMeasurement(0.01);
    invalid.has_covariance = true;
    invalid.covariance = diagonalCovariance(1.0, 1.0);
    invalid.covariance(0, 0) = std::numeric_limits<double>::quiet_NaN();
    require(!filter.update(invalid), "invalid per-frame covariance should be rejected");
    require(filter.lastStatus() == ukf::PoseUKF::Status::kInvalidArgument,
            "invalid covariance should set invalid-argument status");
}

void testStatisticsTrackAcceptedAndInvalidUpdates() {
    ukf::PoseUKF filter;
    require(filter.initialize(makeMeasurement(0.0)), "initialize should succeed");
    require(filter.predict(0.01), "predict should succeed");
    require(filter.update(makeMeasurement(0.01, ukf::Vec3(0.001, 0.0, 0.0))),
            "valid update should succeed");

    ukf::PoseMeasurement invalid = makeMeasurement(0.02);
    invalid.valid = false;
    require(!filter.update(invalid), "invalid measurement should fail");

    const auto& stats = filter.statistics();
    require(stats.predict_count == 1, "predict count should increment");
    require(stats.accepted_update_count == 1, "accepted update count should increment");
    require(stats.invalid_update_count == 1, "invalid update count should increment");
}

} // namespace

int main() {
    try {
        testRejectsInvalidInitialization();
        testRejectsPredictBeforeInitialize();
        testRejectsInvalidPredictStep();
        testQuaternionHemisphereConsistency();
        testMahalanobisGateRejectsOutlier();
        testPerMeasurementCovarianceDownweightsUncertainFrame();
        testTimestampRegressionRejected();
        testDuplicateTimestampRejected();
        testSequenceRegressionRejected();
        testOlderFrameRejectedAfterFutureOutlier();
        testRejectsInvalidPerFrameCovariance();
        testStatisticsTrackAcceptedAndInvalidUpdates();
    } catch (const std::exception& e) {
        std::cerr << "robustness test failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "robustness tests passed\n";
    return 0;
}
