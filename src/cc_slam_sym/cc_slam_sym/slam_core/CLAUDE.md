# slam_core Module Status

## Overview
The SLAM core module implements a high-performance backend using SymForce-generated factors with GTSAM optimization. After resolving CustomFactor compatibility issues, the system now uses batch optimization for stability.

## Current Architecture

### Core Components (6 files)
1. **backend.py** - GTSAM backend with batch optimization
   - Uses LevenbergMarquardt instead of ISAM2 (CustomFactor compatibility)
   - Integrates SymForce-generated factors exclusively
   - Handles prior, odometry, and observation factors

2. **frontend.py** - Keyframe and landmark management
   - Processes sensor observations
   - Manages keyframe creation criteria
   - Tracks landmark lifecycle

3. **data_association.py** - Mahalanobis distance matching
   - Chi-squared gating (95% confidence)
   - Considers both observation and landmark covariances
   - Color-aware association

4. **local_map.py** - Spatial indexing for efficiency
   - KD-tree based nearest neighbor search
   - Essential for real-time performance with 1000+ landmarks

5. **cone_color_factor.py** - SymForce factor definitions
   - Defines symbolic factors for code generation
   - Includes color penalty in residual computation

6. **symforce_gtsam_factors_stable.py** - GTSAM-SymForce bridge
   - Stable wrapper functions for CustomFactor
   - Handles pose conversions (GTSAM ↔ SymForce)
   - Simplified residuals to avoid numerical issues

### Generated Code (generated/)
- **cone_color_factor_residual.py** - Optimized cone observation
- **motion_model_residual.py** - Ackermann motion constraints
- **bearing_range_color_residual.py** - Alternative observation model
- Plus Jacobian versions (not currently used due to stability)

### Development Tools (generators/)
- Code generation scripts (not needed at runtime)
- Moved to separate directory for clarity

## Recent Changes (2025-07-16)

### 1. Fixed GTSAM CustomFactor Compatibility
- **Problem**: ISAM2 incremental updates incompatible with Python CustomFactor
- **Solution**: Switched to batch optimization (LevenbergMarquardt)
- **Result**: Stable optimization without Jacobian dimension errors

### 2. Resolved SymForce Integration Issues
- **Problem**: Division by zero in generated code
- **Solution**: Added epsilon safeguards in residual functions
- **Result**: No crashes when robot at x=0

### 3. Cleaned Up Redundant Files
- Removed 3 redundant factor variants
- Moved generators to development directory
- Reduced from 21 to 13 essential files

## Performance Status
- ✅ Using SymForce-generated residual functions
- ✅ Batch optimization provides stable performance
- ⚠️ Numerical Jacobians (analytical available but not tested)
- 🎯 Future: Test analytical Jacobians for 2-5x speedup

## Factor Types

### 1. Motion Model Factor
- Ackermann constraints for car-like motion
- 4D residual: [x, y, theta, lateral_penalty]
- Penalizes lateral slip heavily

### 2. Cone Observation Factor  
- Color-aware landmark observations
- 3D residual: [x, y, color_penalty]
- Dynamic color weight based on confidence

### 3. Prior Factor
- Standard GTSAM prior for initial pose
- Provides absolute reference frame

## Known Limitations
1. Batch optimization less scalable than ISAM2
2. Analytical Jacobians not yet enabled
3. No loop closure detection implemented

## Next Steps
1. Test analytical Jacobians for performance
2. Implement sliding window for scalability
3. Add loop closure detection
4. Profile and optimize hot paths