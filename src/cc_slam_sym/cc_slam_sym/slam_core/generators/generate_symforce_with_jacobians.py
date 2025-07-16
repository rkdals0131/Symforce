#!/usr/bin/env python3
"""
Generate SymForce factors with analytical Jacobians for GTSAM integration
"""

import symforce
symforce.set_epsilon_to_symbol()

import numpy as np
from pathlib import Path
from symforce import ops
import symforce.symbolic as sf
from symforce import typing as T
from symforce.values import Values
from symforce.codegen import Codegen, CppConfig, PythonConfig
from symforce.codegen import codegen_util

# Import factor definitions
from cone_color_factor import ConeColorFactor
from motion_model_factor import MotionModelFactor
from bearing_range_factor import BearingRangeColorFactor

def generate_cone_color_with_jacobians():
    """Generate cone color factor residual and Jacobian functions"""
    print("Generating cone color factor with Jacobians...")
    
    # Create symbolic variables
    robot_pose = sf.Pose2.symbolic("robot_pose")
    landmark_pos = sf.M21.symbolic("landmark_pos")
    observation = sf.M21.symbolic("observation")
    observed_color = sf.Symbol("observed_color")
    landmark_color = sf.Symbol("landmark_color")
    color_weight = sf.Symbol("color_weight")
    epsilon = sf.Symbol("epsilon")
    
    # Compute residual
    factor = ConeColorFactor(
        robot_pose=robot_pose,
        landmark_pos=landmark_pos,
        observation=observation,
        observed_color=observed_color,
        landmark_color=landmark_color,
        color_weight=color_weight,
        epsilon=epsilon
    )
    residual = factor.residual()
    
    # Generate residual function
    residual_codegen = Codegen(
        inputs=[robot_pose, landmark_pos, observation, observed_color, landmark_color, color_weight, epsilon],
        outputs=[residual],
        config=PythonConfig(),
        name="cone_color_factor_residual_with_jacobians",
        return_key="residual",
        output_names=["residual", "jacobian_pose", "jacobian_landmark"]
    )
    
    # Compute Jacobians
    jacobian_pose = residual.jacobian(robot_pose)
    jacobian_landmark = residual.jacobian(landmark_pos)
    
    # Update codegen with Jacobians
    residual_codegen.outputs = [residual, jacobian_pose, jacobian_landmark]
    
    output_dir = Path(__file__).parent / "generated"
    residual_codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    
    print(f"✓ Generated: {output_dir}/cone_color_factor_residual_with_jacobians.py")


def generate_motion_model_with_jacobians():
    """Generate motion model factor residual and Jacobian functions"""
    print("Generating motion model factor with Jacobians...")
    
    # Create symbolic variables
    pose1 = sf.Pose2.symbolic("pose1")
    pose2 = sf.Pose2.symbolic("pose2")
    odom = sf.M31.symbolic("odom")
    wheelbase = sf.Symbol("wheelbase")
    epsilon = sf.Symbol("epsilon")
    
    # Compute residual
    factor = MotionModelFactor(
        pose1=pose1,
        pose2=pose2,
        odom=odom,
        wheelbase=wheelbase,
        epsilon=epsilon
    )
    residual = factor.residual()
    
    # Compute Jacobians
    jacobian_pose1 = residual.jacobian(pose1)
    jacobian_pose2 = residual.jacobian(pose2)
    
    # Generate function with Jacobians
    residual_codegen = Codegen(
        inputs=[pose1, pose2, odom, wheelbase, epsilon],
        outputs=[residual, jacobian_pose1, jacobian_pose2],
        config=PythonConfig(),
        name="motion_model_residual_with_jacobians",
        return_key="residual",
        output_names=["residual", "jacobian_pose1", "jacobian_pose2"]
    )
    
    output_dir = Path(__file__).parent / "generated"
    residual_codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    
    print(f"✓ Generated: {output_dir}/motion_model_residual_with_jacobians.py")


def generate_bearing_range_with_jacobians():
    """Generate bearing-range color factor residual and Jacobian functions"""
    print("Generating bearing-range color factor with Jacobians...")
    
    # Create symbolic variables
    pose = sf.Pose2.symbolic("pose")
    landmark = sf.M21.symbolic("landmark")
    bearing_obs = sf.Symbol("bearing_obs")
    range_obs = sf.Symbol("range_obs")
    obs_color = sf.Symbol("obs_color")
    landmark_color = sf.Symbol("landmark_color")
    color_weight = sf.Symbol("color_weight")
    epsilon = sf.Symbol("epsilon")
    
    # Compute residual
    factor = BearingRangeColorFactor(
        pose=pose,
        landmark=landmark,
        bearing_obs=bearing_obs,
        range_obs=range_obs,
        obs_color=obs_color,
        landmark_color=landmark_color,
        color_weight=color_weight,
        epsilon=epsilon
    )
    residual = factor.residual()
    
    # Compute Jacobians
    jacobian_pose = residual.jacobian(pose)
    jacobian_landmark = residual.jacobian(landmark)
    
    # Generate function with Jacobians
    residual_codegen = Codegen(
        inputs=[pose, landmark, bearing_obs, range_obs, obs_color, landmark_color, color_weight, epsilon],
        outputs=[residual, jacobian_pose, jacobian_landmark],
        config=PythonConfig(),
        name="bearing_range_color_residual_with_jacobians",
        return_key="residual",
        output_names=["residual", "jacobian_pose", "jacobian_landmark"]
    )
    
    output_dir = Path(__file__).parent / "generated"
    residual_codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)
    
    print(f"✓ Generated: {output_dir}/bearing_range_color_residual_with_jacobians.py")


def main():
    print("=== Generating SymForce Factors with Analytical Jacobians ===\n")
    
    # Create output directory
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    
    # Generate all factors with Jacobians
    generate_cone_color_with_jacobians()
    generate_motion_model_with_jacobians()
    generate_bearing_range_with_jacobians()
    
    print("\n✅ All factors generated successfully with analytical Jacobians!")
    print("\nNext steps:")
    print("1. Update symforce_gtsam_factors.py to use the new functions with Jacobians")
    print("2. Modify CustomFactor error functions to provide Jacobians when requested")
    print("3. Test the improved performance and accuracy")


if __name__ == "__main__":
    main()