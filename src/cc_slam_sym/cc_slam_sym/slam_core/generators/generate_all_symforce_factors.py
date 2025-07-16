#!/usr/bin/env python3
"""
Generate all SymForce factors for the SLAM system
This script generates optimized Python code for all symbolic factor definitions
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

# Import our factor modules
from cc_slam_sym.slam_core.cone_color_factor import ConeColorFactor


def generate_motion_model_factor():
    """Generate optimized code for motion model with Ackermann constraints"""
    
    # Define symbolic inputs
    pose1 = sf.Pose2.symbolic("pose1")
    pose2 = sf.Pose2.symbolic("pose2") 
    odom = sf.V3.symbolic("odom")  # [dx, dy, dtheta]
    wheelbase = sf.Symbol("wheelbase")
    epsilon = sf.Symbol("epsilon")
    
    # Build the residual function
    def motion_residual(pose1, pose2, odom, wheelbase, epsilon):
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
            residual=motion_residual(pose1, pose2, odom, wheelbase, epsilon)
        ),
        config=codegen.PythonConfig(),
        name="motion_model_residual"
    )
    
    codegen_obj.generate_function(
        output_dir=str(output_dir),
        skip_directory_nesting=True
    )
    
    print(f"Generated motion_model_residual in {output_dir}")


def generate_bearing_range_factor():
    """Generate optimized bearing-range factor with color for landmark observations"""
    
    # Define symbolic inputs
    pose = sf.Pose2.symbolic("pose")
    landmark = sf.V2.symbolic("landmark")
    bearing_obs = sf.Symbol("bearing_obs")  # Observed bearing angle
    range_obs = sf.Symbol("range_obs")     # Observed range
    obs_color = sf.Symbol("obs_color")
    landmark_color = sf.Symbol("landmark_color")
    color_weight = sf.Symbol("color_weight")
    epsilon = sf.Symbol("epsilon")
    
    def bearing_range_residual(pose, landmark, bearing_obs, range_obs, 
                              obs_color, landmark_color, color_weight, epsilon):
        # Transform landmark to robot frame
        landmark_robot = pose.inverse() * landmark
        
        # Predicted bearing and range
        predicted_bearing = sf.atan2(landmark_robot[1], landmark_robot[0])
        predicted_range = landmark_robot.norm()
        
        # Bearing error (wrapped to [-pi, pi])
        bearing_error = sf.wrap_angle(predicted_bearing - bearing_obs)
        
        # Range error
        range_error = predicted_range - range_obs
        
        # Color error
        color_diff = sf.Abs(obs_color - landmark_color)
        color_error = color_weight * sf.Min(color_diff, 1.0)
        
        return sf.V3(bearing_error, range_error, color_error)
    
    # Generate code
    output_dir = Path(__file__).parent / "generated"
    
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
            residual=bearing_range_residual(pose, landmark, bearing_obs, range_obs,
                                           obs_color, landmark_color, color_weight, epsilon)
        ),
        config=codegen.PythonConfig(),
        name="bearing_range_color_residual"
    )
    
    codegen_obj.generate_function(
        output_dir=str(output_dir),
        skip_directory_nesting=True
    )
    
    print(f"Generated bearing_range_color_residual in {output_dir}")


def main():
    """Generate all SymForce factors"""
    print("=== Generating All SymForce Factors ===\n")
    
    # Cone color factor
    print("1. Generating cone color factor...")
    try:
        output_dir = ConeColorFactor.generate_code()
        print(f"   ✓ Generated in {output_dir}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Motion model factor
    print("\n2. Generating motion model factor...")
    try:
        generate_motion_model_factor()
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Bearing-range factor
    print("\n3. Generating bearing-range factor with color...")
    try:
        generate_bearing_range_factor()
        print("   ✓ Success")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n=== Code Generation Complete ===")
    
    # List all generated files
    generated_dir = Path(__file__).parent / "generated"
    if generated_dir.exists():
        print(f"\nGenerated files in {generated_dir}:")
        for file in sorted(generated_dir.glob("*.py")):
            if file.name != "__init__.py":
                print(f"  - {file.name}")


if __name__ == "__main__":
    main()