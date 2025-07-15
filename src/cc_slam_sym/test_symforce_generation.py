#!/usr/bin/env python3
"""Test SymForce code generation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Configure SymForce before any other imports
import symforce
try:
    if symforce.get_epsilon() == 0:
        symforce.set_epsilon_to_symbol()
except symforce.AlreadyUsedEpsilon:
    print("SymForce epsilon already configured")
except AttributeError:
    print("SymForce may not be available")

from cc_slam_sym.slam_core.cone_color_factor import ConeColorFactor

if __name__ == "__main__":
    print("Testing SymForce code generation...")
    
    # Generate code
    output_dir = ConeColorFactor.generate_code()
    print(f"Generated code in: {output_dir}")
    
    # List generated files
    generated_files = list(Path(output_dir).glob("*.py"))
    print(f"Generated files: {[f.name for f in generated_files]}")
    
    # Test the generated function
    print("\nTesting generated function...")
    sys.path.insert(0, output_dir)
    try:
        from cone_color_factor_residual import cone_color_factor_residual
        
        import numpy as np
        
        # Test case 1: Matching colors
        result = cone_color_factor_residual(
            robot_pose=np.array([0, 0, 0]),  # x, y, theta
            landmark_pos=np.array([5, 0]),    # world position
            observation=np.array([5, 0]),     # robot frame
            observed_color=0.0,               # yellow
            landmark_color=0.0,               # yellow
            color_weight=1.0,
            epsilon=1e-8
        )
        
        print(f"Test 1 (matching colors):")
        print(f"  Residual: {result['residual']}")
        if 'jacobian' in result:
            print(f"  Jacobian shape: {result['jacobian'].shape}")
        
        # Test case 2: Non-matching colors
        result2 = cone_color_factor_residual(
            robot_pose=np.array([0, 0, 0]),
            landmark_pos=np.array([5, 0]),
            observation=np.array([5, 0]),
            observed_color=0.0,  # yellow
            landmark_color=1.0,  # blue
            color_weight=10.0,
            epsilon=1e-8
        )
        
        print(f"\nTest 2 (non-matching colors):")
        print(f"  Residual: {result2['residual']}")
        print(f"  Color penalty: {result2['residual'][2]}")
        
        print("\nSymForce code generation successful!")
        
    except Exception as e:
        print(f"Error testing generated function: {e}")
        import traceback
        traceback.print_exc()