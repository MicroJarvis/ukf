# Candidate Datasets for Testing

Best-fit public datasets for validating pose tracking / 6DoF robustness:

1. **NTU ISMAr 2017 Benchmark**
   - Robot arm moves a target object through various trajectories.
   - Useful for pose tracking under motion.

2. **RobotP**
   - Synthetic robot-arm-driven pose data.
   - Good for algorithmic regression tests.

3. **BOP / T-LESS / related industrial pose datasets**
   - Not arm trajectories, but strong for robust 6DoF pose estimation and occlusion stress tests.

4. **StereOBJ-1M**
   - Stereo 6D object pose benchmark.
   - Helpful if the stereo front-end is central.

Recommended workflow:
- Validate filter logic on synthetic motion.
- Validate measurement robustness on BOP/T-LESS or StereOBJ-1M.
- Validate motion continuity on NTU ISMAr or robot-generated own data.
