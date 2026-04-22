#include "ukf.h"
#include <cmath>
#include <iostream>
#include <random>

int main() {
    std::mt19937 rng(42);
    std::normal_distribution<double> npos(0.0, 0.002);
    std::normal_distribution<double> nori(0.0, 0.001);

    ukf::PoseUKF filter;
    ukf::PoseMeasurement init;
    init.position = ukf::Vec3(0.0, 0.0, 0.0);
    init.orientation = ukf::Quaternion::Identity();
    filter.initialize(init);
    filter.setProcessNoise(1e-5, 1e-4, 1e-6, 1e-5);
    filter.setMeasurementNoise(1e-3, 1e-4);

    double t = 0.0;
    for (int k = 0; k < 200; ++k) {
        double dt = 0.01;
        t += dt;
        ukf::Vec3 true_pos(0.2 * std::cos(0.5 * t), 0.2 * std::sin(0.5 * t), 0.1 * t);
        ukf::Vec3 axis(0.0, 0.0, 1.0);
        double ang = 0.2 * t;
        ukf::Quaternion true_q(std::cos(ang * 0.5), axis.x() * std::sin(ang * 0.5), axis.y() * std::sin(ang * 0.5), axis.z() * std::sin(ang * 0.5));

        ukf::PoseMeasurement meas;
        meas.position = true_pos + ukf::Vec3(npos(rng), npos(rng), npos(rng));
        meas.orientation = (true_q * ukf::quatExp(ukf::Vec3(nori(rng), nori(rng), nori(rng)))).normalized();

        filter.predict(dt);
        filter.update(meas);

        if (k % 20 == 0) {
            auto p = filter.position();
            auto q = filter.orientation();
            std::cout << "k=" << k << " p=" << p.transpose() << " q=[" << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "]\n";
        }
    }
    return 0;
}
