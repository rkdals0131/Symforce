# SymForce/GTSAM Full Integration Plan

## Executive Summary

The current SLAM system is NOT using SymForce despite having the infrastructure in place. This causes severe performance degradation after one lap due to unoptimized factor computation. This plan outlines how to achieve full SymForce/GTSAM integration.

## Current State Analysis

### What's Working
- Basic GTSAM backend with standard factors
- ROS2 interface and simulation environment
- Data structures and frontend processing

### What's NOT Working  
- SymForce code generation (generated/ directory empty)
- Custom factors disabled (backend.py line 215: use_custom_factor = False)
- No analytical Jacobians - using numerical differentiation
- Poor performance due to Python-based computation

### Dead Code to Remove
- symforce_backend.py (unused alternative implementation)
- symforce_factors.py (symbolic definitions not integrated)
- Missing imports in __init__.py (slam_system.py, slam_visualizer.py)

## Integration Approach Decision

After analysis, **Option B** is recommended: **Integrate SymForce factors into existing backend.py**

Rationale:
- backend.py is already working and integrated with ROS
- Minimal changes to existing codebase
- Can incrementally add SymForce optimization
- Lower risk of breaking existing functionality

## Implementation Steps

### Phase 1: Enable SymForce Code Generation (Day 1)

1. **Fix SymForce code generation**
   ```bash
   cd cc_slam_sym/slam_core
   python cone_color_factor.py  # Generate optimized functions
   ```

2. **Verify generated code**
   - Check generated/ directory for cone_color_factor_residual.py
   - Test generated functions with test_symforce_generation.py

3. **Create build script**
   ```python
   # generate_symforce_factors.py
   from cone_color_factor import ConeColorFactor
   ConeColorFactor.generate_code()
   ```

### Phase 2: Create GTSAM Wrapper (Day 1-2)

1. **Create new custom factor using SymForce**
   ```python
   # symforce_gtsam_factors.py
   class SymforceConeObservationFactor(gtsam.CustomFactor):
       def error(self, values):
           # Use generated cone_color_factor_residual function
           pass
   ```

2. **Implement Jacobian computation**
   - Extract Jacobians from SymForce output
   - Map to GTSAM's expected format

### Phase 3: Integrate into Backend (Day 2)

1. **Modify backend.py**
   - Line 215: Change use_custom_factor = True
   - Import SymForce-based factors
   - Replace BearingRangeFactor2D with SymforceConeObservationFactor

2. **Add configuration option**
   ```yaml
   # slam_config.yaml
   backend:
     use_symforce_factors: true
     factor_type: "symforce"  # or "standard"
   ```

### Phase 4: Testing and Validation (Day 3)

1. **Unit tests**
   - Compare SymForce vs standard factor outputs
   - Verify Jacobian correctness
   - Benchmark performance improvement

2. **Integration tests**
   - Run with dummy_publisher_node
   - Monitor performance metrics
   - Validate trajectory accuracy

### Phase 5: Cleanup (Day 3)

1. **Remove dead code**
   - Delete unused symforce_backend.py
   - Clean up redundant factor definitions
   - Update __init__.py imports

2. **Documentation**
   - Update all CLAUDE.md files
   - Create usage examples
   - Document performance improvements

## Expected Outcomes

### Performance Improvements
- 5-10x speedup in factor evaluation
- Maintain 10Hz processing rate throughout operation
- Reduced CPU usage from optimized Jacobians

### Code Quality
- Single, unified backend implementation
- Clear separation between standard and SymForce factors
- Maintainable and extensible architecture

## Risk Mitigation

1. **Fallback mechanism**: Keep standard factors as option
2. **Incremental rollout**: Test one factor type at a time
3. **Performance monitoring**: Track metrics before/after
4. **Version control**: Create feature branch for integration

## Success Criteria

✅ SymForce code successfully generated
✅ Custom factors using analytical Jacobians
✅ 10Hz processing maintained after full lap
✅ Improved data association with color constraints
✅ All tests passing
✅ Dead code removed

## Timeline

- Day 1: Code generation and wrapper creation
- Day 2: Backend integration
- Day 3: Testing, validation, and cleanup
- Total: 3 days for full integration

## Next Immediate Action

Run this command to start:
```bash
cd /home/user1/ROS2_Workspace/Symforce_ws/src/cc_slam_sym
python cc_slam_sym/slam_core/cone_color_factor.py
```

Then verify generated code exists before proceeding with integration.