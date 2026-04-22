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
./build/ukf_robustness
```

## Notes

- Current implementation is a hardened prototype with an SE(3)-style error-state UKF update.
- It uses a 13D nominal state: position, velocity, quaternion, angular velocity.
- Runtime safeguards include quaternion hemisphere alignment, PSD covariance projection,
  sigma-point factorization jitter, Mahalanobis gating, strict timestamp / sequence ordering,
  and runtime statistics.
- `PoseMeasurement` can now carry per-frame 6x6 covariance from the vision front-end.
- `ukf_robustness` covers invalid inputs, outlier rejection, quaternion sign consistency,
  per-frame covariance handling, and stale-frame rejection.
- Next step: adapt the measurement model to your stereo marker pipeline and add dataset-driven regression.
