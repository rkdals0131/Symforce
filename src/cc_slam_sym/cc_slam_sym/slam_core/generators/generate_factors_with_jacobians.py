#!/usr/bin/env python3
"""
Generate SymForce factors with analytical Jacobians for GTSAM integration
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure SymForce before any other imports
import symforce
try:
    symforce.set_epsilon_to_symbol()
except:
    pass

import symforce.symbolic as sf
from symforce import codegen
from symforce.values import Values
import numpy as np

# Import our factor modules
from cc_slam_sym.slam_core.cone_color_factor import ConeColorFactor


def generate_cone_color_with_jacobians():
    """Generate cone color factor residual with analytical Jacobians"""
    print("Generating cone color factor with Jacobians...")
    
    # Create symbolic variables
    robot_pose = sf.Pose2.symbolic("robot_pose")
    landmark_pos = sf.V2.symbolic("landmark_pos")
    observation = sf.V2.symbolic("observation")
    observed_color = sf.Symbol("observed_color")
    landmark_color = sf.Symbol("landmark_color")
    color_weight = sf.Symbol("color_weight")
    epsilon = sf.Symbol("epsilon")
    
    # Compute residual using the static method
    residual = ConeColorFactor.residual(
        robot_pose=robot_pose,
        landmark_pos=landmark_pos,
        observation=observation,
        observed_color=observed_color,
        landmark_color=landmark_color,
        color_weight=color_weight,
        epsilon=epsilon
    )
    
    # Compute Jacobians
    jacobian_pose = residual.jacobian(robot_pose)
    jacobian_landmark = residual.jacobian(landmark_pos)
    
    # Generate function with residual and Jacobians
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    
    codegen_obj = codegen.Codegen(
        inputs=Values(
            robot_pose=robot_pose,
            landmark_pos=landmark_pos,
            observation=observation,
            observed_color=observed_color,
            landmark_color=landmark_color,
            color_weight=color_weight,
            epsilon=epsilon
        ),
        outputs=Values(
            residual=residual,
            jacobian_pose=jacobian_pose,
            jacobian_landmark=jacobian_landmark
        ),
        config=codegen.PythonConfig(),
        name="cone_color_factor_with_jacobians"
    )
    
    codegen_obj.generate_function(
        output_dir=str(output_dir),
        skip_directory_nesting=True
    )
    
    print(f"✓ Generated: {output_dir}/cone_color_factor_with_jacobians.py")


def generate_motion_model_with_jacobians():
    """Generate motion model factor with analytical Jacobians"""
    print("Generating motion model factor with Jacobians...")
    
    # Define symbolic inputs
    pose1 = sf.Pose2.symbolic("pose1")
    pose2 = sf.Pose2.symbolic("pose2") 
    odom = sf.V3.symbolic("odom")  # [dx, dy, dtheta]
    wheelbase = sf.Symbol("wheelbase")
    epsilon = sf.Symbol("epsilon")
    
    # Build the residual function
    def motion_residual_func(pose1, pose2, odom, wheelbase, epsilon):
        # Expected pose after motion
        expected_pose = pose1 * sf.Pose2(sf.Rot2.from_angle(odom[2]), sf.V2(odom[0], odom[1]))
        
        # Compute error between expected and actual
        error_pose = expected_pose.inverse() * pose2
        
        # Extract position and rotation errors
        position_error = error_pose.position()
        rotation_error = error_pose.rotation().to_tangent()
        
        # Add Ackermann constraint - penalize lateral motion
        # In robot frame, lateral motion is y-component of velocity
        velocity = pose1.inverse() * pose2.position()
        lateral_penalty = 10.0 * velocity[1]  # Penalize y-velocity
        
        return sf.V4(
            position_error[0],
            position_error[1], 
            rotation_error[0],
            lateral_penalty
        )
    
    # Compute residual
    residual = motion_residual_func(pose1, pose2, odom, wheelbase, epsilon)
    
    # Compute Jacobians
    jacobian_pose1 = residual.jacobian(pose1)
    jacobian_pose2 = residual.jacobian(pose2)
    
    # Generate code
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    
    codegen_obj = codegen.Codegen(
        inputs=Values(
            pose1=pose1,
            pose2=pose2,
            odom=odom,
            wheelbase=wheelbase,
            epsilon=epsilon
        ),
        outputs=Values(
            residual=residual,
            jacobian_pose1=jacobian_pose1,
            jacobian_pose2=jacobian_pose2
        ),
        config=codegen.PythonConfig(),
        name="motion_model_with_jacobians"
    )
    
    codegen_obj.generate_function(
        output_dir=str(output_dir),
        skip_directory_nesting=True
    )
    
    print(f"✓ Generated: {output_dir}/motion_model_with_jacobians.py")


def generate_bearing_range_with_jacobians():
    """Generate bearing-range factor with analytical Jacobians"""
    print("Generating bearing-range factor with Jacobians...")
    
    # Define symbolic inputs
    pose = sf.Pose2.symbolic("pose")
    landmark = sf.V2.symbolic("landmark")
    bearing_obs = sf.Symbol("bearing_obs")  # Observed bearing angle
    range_obs = sf.Symbol("range_obs")     # Observed range
    obs_color = sf.Symbol("obs_color")
    landmark_color = sf.Symbol("landmark_color")
    color_weight = sf.Symbol("color_weight")
    epsilon = sf.Symbol("epsilon")
    
    # Build the residual function
    def bearing_range_residual_func(pose, landmark, bearing_obs, range_obs, 
                                    obs_color, landmark_color, color_weight, epsilon):
        # Transform landmark to robot frame
        rel_pos = pose.inverse() * landmark
        
        # Compute expected bearing and range
        expected_bearing = sf.atan2(rel_pos[1], rel_pos[0])
        expected_range = rel_pos.norm(epsilon=epsilon)
        
        # Bearing error (handle angle wrapping)
        bearing_error = sf.wrap_angle(expected_bearing - bearing_obs)
        
        # Range error
        range_error = expected_range - range_obs
        
        # Color error
        color_diff = sf.Min(sf.Abs(landmark_color - obs_color), 1.0)
        color_error = color_weight * color_diff
        
        return sf.V3(bearing_error, range_error, color_error)
    
    # Compute residual
    residual = bearing_range_residual_func(pose, landmark, bearing_obs, range_obs,
                                          obs_color, landmark_color, color_weight, epsilon)
    
    # Compute Jacobians
    jacobian_pose = residual.jacobian(pose)
    jacobian_landmark = residual.jacobian(landmark)
    
    # Generate code
    output_dir = Path(__file__).parent / "generated"
    output_dir.mkdir(exist_ok=True)
    
    codegen_obj = codegen.Codegen(
        inputs=Values(
            pose=pose,
            landmark=landmark,
            bearing_obs=bearing_obs,
            range_obs=range_obs,
            obs_color=obs_color,
            landmark_color=landmark_color,
            color_weight=color_weight,
            epsilon=epsilon
        ),
        outputs=Values(
            residual=residual,
            jacobian_pose=jacobian_pose,
            jacobian_landmark=jacobian_landmark
        ),
        config=codegen.PythonConfig(),
        name="bearing_range_with_jacobians"
    )
    
    codegen_obj.generate_function(
        output_dir=str(output_dir),
        skip_directory_nesting=True
    )
    
    print(f"✓ Generated: {output_dir}/bearing_range_with_jacobians.py")


def main():
    print("=== Generating SymForce Factors with Analytical Jacobians ===\n")
    
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