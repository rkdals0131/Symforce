#!/usr/bin/env python3
"""
Cone color factor for Symforce
Incorporates color information into cone observation constraints
"""

import symforce
import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util
from symforce.values import Values
import numpy as np
from pathlib import Path

@codegen.with_codegen_namespace("cc_slam_sym")
class ConeColorFactor:
    """
    Cone observation factor that considers color matching.
    Applies higher uncertainty when colors don't match.
    """
    
    @staticmethod
    def residual(
        robot_pose: sf.Pose2,
        landmark_pos: sf.V2,
        observation: sf.V2,
        observed_color: sf.Scalar,  # 0: yellow, 1: blue, 2: red, 3: orange
        landmark_color: sf.Scalar,
        color_weight: sf.Scalar,
        epsilon: sf.Scalar = sf.numeric_epsilon
    ) -> sf.V3:
        """
        Compute residual for cone observation with color penalty
        
        Args:
            robot_pose: Robot SE(2) pose in world frame
            landmark_pos: Landmark 2D position in world frame
            observation: Observed cone position in robot frame
            observed_color: Observed cone color (encoded as scalar)
            landmark_color: Landmark color (encoded as scalar)
            color_weight: Weight for color mismatch penalty
            epsilon: Small value for numerical stability
            
        Returns:
            3D residual vector [position_error_x, position_error_y, color_error]
        """
        # Transform landmark to robot frame
        landmark_in_world = sf.V3(landmark_pos[0], landmark_pos[1], 1)
        T_world_robot = robot_pose.to_homogeneous_matrix()
        T_robot_world = T_world_robot.inv()
        
        landmark_in_robot_homo = T_robot_world * landmark_in_world
        predicted = sf.V2(landmark_in_robot_homo[0], landmark_in_robot_homo[1])
        
        # Position residual
        position_residual = predicted - observation
        
        # Color residual - penalize if colors don't match
        color_diff = sf.Abs(observed_color - landmark_color)
        # If colors match (diff < 0.5), residual is small
        # If colors don't match (diff >= 0.5), residual is large
        color_residual = color_weight * sf.Min(color_diff, 1.0)
        
        return sf.V3(
            position_residual[0],
            position_residual[1],
            color_residual
        )
    
    @staticmethod
    def generate_code(output_dir: str = None):
        """Generate C++ code for this factor"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "generated" / "cone_color_factor"
        
        # Define inputs
        inputs = Values(
            robot_pose=sf.Pose2.identity(),
            landmark_pos=sf.V2.zero(),
            observation=sf.V2.zero(),
            observed_color=sf.Scalar(0),
            landmark_color=sf.Scalar(0),
            color_weight=sf.Scalar(1.0),
            epsilon=sf.Scalar(sf.numeric_epsilon)
        )
        
        # Define outputs
        outputs = Values(
            residual=sf.V3.zero()
        )
        
        # Create codegen object
        codegen_obj = codegen.Codegen(
            inputs=inputs,
            outputs=outputs,
            config=codegen.CppConfig(),
            name="cone_color_factor",
            return_key="residual",
            sparse_matrices=True,
            docstring=ConeColorFactor.residual.__doc__
        )
        
        # Generate the code
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        codegen_obj.generate_function(
            output_dir=str(output_dir),
            skip_directory_nesting=True
        )
        
        return str(output_dir)
    
    @staticmethod
    def color_to_scalar(color_str: str) -> float:
        """Convert color string to scalar encoding"""
        color_map = {
            'yellow': 0.0,
            'blue': 1.0,
            'red': 2.0,
            'orange': 3.0
        }
        return color_map.get(color_str.lower(), -1.0)
    
    @staticmethod
    def compute_color_weight(base_weight: float = 1.0) -> float:
        """
        Compute adaptive color weight based on conditions
        Could be extended to consider lighting, distance, etc.
        """
        return base_weight


def test_cone_color_factor():
    """Test the cone color factor"""
    # Test case 1: Matching colors
    robot_pose = sf.Pose2(sf.Rot2.from_angle(0.0), sf.V2(0, 0))
    landmark_pos = sf.V2(5.0, 0.0)
    observation = sf.V2(5.0, 0.0)
    
    residual = ConeColorFactor.residual(
        robot_pose=robot_pose,
        landmark_pos=landmark_pos,
        observation=observation,
        observed_color=sf.Scalar(0.0),  # yellow
        landmark_color=sf.Scalar(0.0),   # yellow
        color_weight=sf.Scalar(1.0)
    )
    
    print("Test 1 - Matching colors:")
    print(f"  Residual: {residual}")
    print(f"  Color component: {residual[2]}")
    
    # Test case 2: Non-matching colors
    residual2 = ConeColorFactor.residual(
        robot_pose=robot_pose,
        landmark_pos=landmark_pos,
        observation=observation,
        observed_color=sf.Scalar(0.0),  # yellow
        landmark_color=sf.Scalar(1.0),   # blue
        color_weight=sf.Scalar(10.0)
    )
    
    print("\nTest 2 - Non-matching colors:")
    print(f"  Residual: {residual2}")
    print(f"  Color component: {residual2[2]}")
    
    # Generate code
    print("\nGenerating C++ code...")
    output_dir = ConeColorFactor.generate_code()
    print(f"Code generated in: {output_dir}")


if __name__ == "__main__":
    test_cone_color_factor()