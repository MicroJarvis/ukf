#include "ukf.h"
#include <iostream>

int main() {
    ukf::PoseUKF filter;
    ukf::PoseMeasurement init;
    init.position = ukf::Vec3(0.0, 0.0, 0.0);
    init.orientation = ukf::Quaternion::Identity();
    filter.initialize(init);
    filter.setProcessNoise(1e-5, 1e-4, 1e-6, 1e-5);
    filter.setMeasurementNoise(1e-3, 1e-4);

    for (int i = 0; i < 10; ++i) {
        filter.predict(0.01);
        ukf::PoseMeasurement m;
        m.position = ukf::Vec3(0.01 * i, 0.0, 0.0);
        m.orientation = ukf::Quaternion::Identity();
        filter.update(m);
    }

    std::cout << "pos = " << filter.position().transpose() << "\n";
    std::cout << "vel = " << filter.velocity().transpose() << "\n";
    auto q = filter.orientation();
    std::cout << "quat = " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
    return 0;
}
