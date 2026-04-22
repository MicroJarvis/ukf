# Project Notes

- Focus: stereo marker-based 6DoF tracking for moving tool fixture on robot flange.
- Dependencies: only C++ standard library + Eigen.
- Filter choice: UKF as default; IMM-UKF later if mode switching becomes important.
- Practical next work: add a proper SE(3) error-state formulation and a measurement adapter from stereo marker detections.
