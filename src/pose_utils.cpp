#include "pose_utils.h"
#include <cmath>

namespace ukf {

bool isFiniteVector(const Vec3& v) {
    return v.array().isFinite().all();
}

bool isFiniteMatrix(const Mat6& m) {
    return m.array().isFinite().all();
}

bool isFiniteQuaternion(const Quaternion& q) {
    return std::isfinite(q.w()) && std::isfinite(q.x()) &&
           std::isfinite(q.y()) && std::isfinite(q.z());
}

Quaternion normalizeQuaternion(const Quaternion& q) {
    if (!isFiniteQuaternion(q)) {
        return Quaternion::Identity();
    }

    Quaternion n = q;
    if (n.norm() < 1e-12) {
        return Quaternion::Identity();
    }

    n.normalize();
    if (n.w() < 0.0) {
        n.coeffs() *= -1.0;
    }
    return n;
}

Quaternion alignQuaternionHemisphere(const Quaternion& q, const Quaternion& reference) {
    Quaternion aligned = normalizeQuaternion(q);
    const Quaternion ref = normalizeQuaternion(reference);
    if (aligned.coeffs().dot(ref.coeffs()) < 0.0) {
        aligned.coeffs() *= -1.0;
    }
    return aligned;
}

Quaternion quatMultiply(const Quaternion& a, const Quaternion& b) {
    return a * b;
}

Quaternion quatConjugate(const Quaternion& q) {
    return q.conjugate();
}

Quaternion quatExp(const Vec3& v) {
    const double theta = v.norm();
    if (theta < 1e-12) {
        return Quaternion(1.0, 0.5 * v.x(), 0.5 * v.y(), 0.5 * v.z()).normalized();
    }
    const Vec3 axis = v / theta;
    const double half = 0.5 * theta;
    return Quaternion(std::cos(half), axis.x() * std::sin(half), axis.y() * std::sin(half), axis.z() * std::sin(half)).normalized();
}

Vec3 quatLog(const Quaternion& q_in) {
    Quaternion q = normalizeQuaternion(q_in);
    const double w = std::clamp(q.w(), -1.0, 1.0);
    const double theta = 2.0 * std::acos(w);
    const double s = std::sqrt(std::max(1.0 - w * w, 0.0));
    if (s < 1e-12) {
        return Vec3(q.x(), q.y(), q.z()) * 2.0;
    }
    return theta * Vec3(q.x(), q.y(), q.z()) / s;
}

Quaternion deltaQuatFromOmega(const Vec3& omega, double dt) {
    return quatExp(omega * dt);
}

Vec3 rotationVectorError(const Quaternion& ref, const Quaternion& target) {
    return quatLog(normalizeQuaternion(ref).conjugate() * alignQuaternionHemisphere(target, ref));
}

} // namespace ukf
