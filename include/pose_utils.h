#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace ukf {

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Quaternion = Eigen::Quaterniond;

struct PoseMeasurement {
    Vec3 position{Vec3::Zero()};
    Quaternion orientation{Quaternion::Identity()};
    Mat6 covariance{Mat6::Zero()};
    double timestamp{0.0};
    std::uint64_t sequence{0};
    bool valid{true};
    bool has_covariance{false};
};

bool isFiniteVector(const Vec3& v);
bool isFiniteMatrix(const Mat6& m);
bool isFiniteQuaternion(const Quaternion& q);
Quaternion normalizeQuaternion(const Quaternion& q);
Quaternion alignQuaternionHemisphere(const Quaternion& q, const Quaternion& reference);
Quaternion deltaQuatFromOmega(const Vec3& omega, double dt);
Vec3 quatLog(const Quaternion& q);
Quaternion quatExp(const Vec3& v);
Quaternion quatMultiply(const Quaternion& a, const Quaternion& b);
Quaternion quatConjugate(const Quaternion& q);
Vec3 rotationVectorError(const Quaternion& ref, const Quaternion& target);

} // namespace ukf
