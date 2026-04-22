# UKF Pose Tracker

C++17 + Eigen-only UKF prototype for stereo-vision marker-based 6DoF pose tracking.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/ukf_smoke
./build/ukf_demo
```

## Notes

- Current implementation is a minimal working prototype.
- It uses a 13D state: position, velocity, quaternion, angular velocity.
- Demo simulates a moving target and noisy pose measurements.
- Next step: adapt the measurement model to your stereo marker pipeline.
