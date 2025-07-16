# SymForce Integration Analysis (2025-07-16)

## 1. Current State of SymForce Integration

An analysis of the codebase reveals that the project is not fully utilizing SymForce for factor graph optimization. The key findings are:

- **Symbolic Definitions Exist:** The files `cc_slam_sym/slam_core/symforce_factors.py` and `cc_slam_sym/slam_core/cone_color_factor.py` correctly define symbolic models for odometry, landmark observations, and cone colors using SymForce.
- **Backend Uses Standard GTSAM:** The primary backend logic in `cc_slam_sym/slam_core/backend.py` uses standard, hard-coded GTSAM factors (`gtsam.BetweenFactorPose2`, `gtsam.BearingRangeFactor2D`). It does not use any SymForce-generated factors.
- **Integration Points are Placeholders:** The `use_custom_factor` flag in `backend.py` is hardcoded to `False`. The intended file for the integrated backend, `cc_slam_sym/slam_core/symforce_backend.py`, is currently an empty placeholder.
- **Redundant Manual Factors:** The file `cc_slam_sym/slam_core/custom_factors.py` represents an attempt to create custom factors manually, which is not the intended approach when using SymForce's code generation capabilities.

## 2. What is Missing for Full Integration

To realize the performance benefits of SymForce, the following components need to be built:

1.  **Code Generation Pipeline:** A script is needed to take the symbolic factors and use SymForce's `Codegen` utility to generate highly optimized Python functions for calculating residuals and Jacobians. The existing `test_symforce_generation.py` can serve as a starting point.
2.  **Custom Factor Wrappers:** The generated Python functions must be wrapped within a new class that inherits from the appropriate GTSAM factor base class (e.g., `gtsam.NoiseModelFactor`). This wrapper will serve as the bridge between GTSAM's optimizer and SymForce's high-speed calculations.
3.  **Backend Logic:** The `symforce_backend.py` file must be implemented to build the factor graph using these new custom factors instead of the standard GTSAM ones.

## 3. Unnecessary/Redundant Components

The following files are considered redundant in the context of a full SymForce integration and should be phased out:

- **`cc_slam_sym/slam_core/backend.py`:** This will be replaced by the fully implemented `symforce_backend.py`.
- **`cc_slam_sym/slam_core/custom_factors.py`:** This is superseded by the SymForce code generation pipeline, which is a more robust and maintainable approach.

## 4. Proposed Action Plan

1.  **Generate Factors:** Create a script to generate optimized Python functions from the symbolic definitions in `symforce_factors.py`.
2.  **Implement Custom Factors:** Develop the GTSAM wrapper classes for the generated functions.
3.  **Build SymForce Backend:** Implement the factor graph construction logic in `symforce_backend.py` using the new custom factors.
4.  **Update System to Use New Backend:** Modify the main SLAM node to use the `symforce_backend` instead of the old `backend`.
5.  **Cleanup:** Remove `backend.py` and `custom_factors.py`.
6.  **Update Documentation:** Update `docs/symforce_integration.md` and `docs/gtsam_integration.md` to reflect the new, fully integrated architecture.
