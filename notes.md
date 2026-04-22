# Project Notes

- Focus: stereo marker-based 6DoF tracking for moving tool fixture on robot flange.
- Dependencies: only C++ standard library + Eigen.
- Filter choice: UKF as default; IMM-UKF later if mode switching becomes important.
- Implemented hardening work: error-state covariance on SE(3)-style perturbations, per-frame measurement
  covariance support, Mahalanobis gating, strict timestamp / sequence checks, PSD covariance projection,
  and robustness tests.
- Practical next work: add a measurement adapter from stereo marker detections and dataset-driven regression.
