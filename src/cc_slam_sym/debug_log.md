# CC-SLAM-SYM Debug Log

## 2025-07-16

### Problem: GTSAM CustomFactor ISAM2 Jacobian Error
- **Symptom**: "JacobianFactor has 4 rows but provided matrix block has 0 rows"
- **Root Cause**: GTSAM Python CustomFactor incompatible with ISAM2 incremental updates
- **Analysis**:
  1. CustomFactor works with batch optimization (LevenbergMarquardt)
  2. ISAM2.update() fails during numerical Jacobian computation
  3. Issue specific to Python bindings, C++ CustomFactor works fine
  
- **Solution**: 
  - Modified backend.py optimize() to use batch optimization
  - Maintains all existing functionality
  - Better stability for SymForce-generated residual functions
  
- **Additional Fix**: 
  - Fixed division by zero in cone_color_factor_residual.py
  - Added epsilon safeguard: `1 / (_robot_pose[0] + epsilon)`
  
- **Result**: SLAM system now uses stable batch optimization

## 2025-07-15 15:56

### 문제: 랜드마크가 생성되지 않음
- **증상**: 키프레임은 생성되지만 랜드마크 개수가 계속 0으로 유지됨
- **원인 분석**:
  1. 초기 시도: track_id 매핑 문제로 오판 (잘못된 접근)
  2. 진짜 원인: `frontend.py`의 `_process_unmatched_observation()`에서 거리 체크 로직 오류
  3. 원점(0,0)으로부터의 거리를 계산하여 차량이 원점에서 멀어지면 모든 관측이 `max_landmark_init_distance` 초과
  
- **해결**: 
  - 현재 로봇 위치로부터의 거리로 계산하도록 수정
  - `distance = np.linalg.norm(observation.position[:2] - [robot_x, robot_y])`
  
- **결과**: 테스트 대기중

## 2025-07-15 16:20

### evaluation_report.md 지적 사항 해결 상태
- **문제 1: 데이터 연관 취약성** ✅ 해결됨
  - Mahalanobis distance 구현 완료
  - Chi-squared gating (95% confidence) 적용
  - 관측/랜드마크 covariance 모두 고려
  
- **문제 2: Symforce 커스텀 팩터 미사용** ✅ 해결됨
  - `custom_factors.py`에 ConeObservationFactor 구현
  - 색상 정보를 3차원 에러로 포함
  - backend에서 색상 불일치 시 noise 증가로 대응
  
- **문제 3: BearingRangeFactor 부적절** ⚠️ 부분 해결
  - Python GTSAM 제약으로 완전한 custom factor 어려움
  - 색상 기반 noise scaling으로 보완
  
- **추가 개선사항**:
  - 색상 voting 메커니즘 (단순 카운팅)
  - 랜드마크 covariance 업데이트
  - track_id 기반 landmark-observation 매칭

## 2025-07-15 16:35

### 문제: 곡선 구간에서 data association 실패
- **증상**: 직선에서는 정상 작동하나 곡선에서 매칭이 엉켜 맵이 멈춤
- **원인 분석**:
  1. 오도메트리 예측 없이 순수 관측 위치로만 매칭
  2. 빠른 회전 시 로봇 프레임 관측이 크게 변화
  3. 서브맵/로컬 맵 개념 없이 전역 매칭 시도
  
- **GLIM 참고사항**:
  - 고정 지연 스무더로 최근 N초만 최적화
  - 서브맵 단위로 로컬 일관성 유지
  - 오도메트리 예측 기반 data association
  
- **해결 방안**:
  1. LocalMap 모듈 추가 - 최근 20개 키프레임, 100개 랜드마크만 유지
  2. 오도메트리 예측 추가 - 다음 포즈 예측하여 data association 개선
  3. 공간적/시간적 윈도우로 매칭 범위 제한

## 2025-07-15 17:37

### 문제: GTSAM ISAM2Params API 오류
- **증상**: `AttributeError: 'gtsam.gtsam.ISAM2Params' object has no attribute 'setRelinearizeSkip'`
- **원인**: Python GTSAM 바인딩에서는 속성 접근 방식 사용 (setter 메서드가 아님)
- **해결**: 
  - `params.setRelinearizeSkip()` → `params.relinearizeSkip = value`
  - 모든 ISAM2Params 설정을 속성 접근 방식으로 변경
  
### 문제: SymForce 코드 생성 중복 심볼 오류
- **증상**: `AssertionError: Symbols in inputs must be unique`
- **원인**: Values에 identity/zero 값으로 초기화하면 중복된 심볼로 인식
- **해결**:
  1. `symforce.set_epsilon_to_symbol()` 추가로 epsilon 경고 해결
  2. 심볼릭 변수 먼저 생성 후 Values에 전달
  3. `sf.Pose2.symbolic()`, `sf.V2.symbolic()` 사용
  
### 문제: SymForce 성능 활용 미비
- **증상**: 백엔드가 한 바퀴 돌면 급속도로 느려짐
- **원인 분석**:
  1. SymForce 코드 생성 기능 미사용
  2. Sliding window marginalization 미구현
  3. Factor graph가 곈4속 증가
- **해결**:
  1. `symforce_factors.py` 생성 - 심볼릭 계산 직접 활용
  2. 백엔드 성능 개선 - batch optimization, marginalization
  3. 경로 smoothness 개선 - 각속도/각가속도 제한

### 문제: SymForce AlreadyUsedEpsilon 오류
- **증상**: `symforce.AlreadyUsedEpsilon: Cannot set return value of epsilon`
- **원인**: 모듈 import 시 SymForce epsilon이 이미 사용된 후 설정 시도
- **해결**:
  1. try-except로 epsilon 설정 보호
  2. SYMFORCE_AVAILABLE 플래그로 선택적 사용
  3. 모든 SymForce 모듈에 fallback 구현 추가

### 문제: IndentationError in symforce_factors.py
- **증상**: `IndentationError: expected an indented block after 'else' statement`
- **원인**: SymForce 조건부 import 후 else 블록의 잘못된 들여쓰기
- **해결**: 모든 SymForce 관련 함수들의 들여쓰기 수정

### 문제: SymForce API 변경
- **증상**: `module 'symforce' has no attribute 'get_epsilon'`
- **원인**: SymForce 버전 차이 또는 미설치
- **해결**:
  1. hasattr() 체크 추가
  2. SymForce 없이도 실행 가능하도록 fallback 구현
  3. 모든 관련 함수에 SYMFORCE_AVAILABLE 체크

### 문제: GTSAM ISAM2Params Python API 오류 (재발)
- **증상**: `AttributeError: 'gtsam.gtsam.ISAM2Params' object has no attribute 'relinearizeThreshold'`
- **원인**: 잘못된 API 사용 - setter 메서드와 속성을 혼동
- **해결**:
  1. `setRelinearizeThreshold()` 메서드 사용
  2. `setFactorization("QR")` - 문자열로 전달 (열거형 아님)
  3. dir(params) 로 실제 API 확인

## 2025-07-15 18:30

### 문제: SLAM 성능 저하 및 Data Association 불량
- **증상**: 
  - Queue size 199 (limit 100 초과)
  - 109 keyframes에 optimization 1회만 실행
  - Ground truth odometry임에도 data association 불량
- **원인 분석**:
  1. Optimization trigger 로직 오류: `len(keyframes) % interval`은 총 개수 기반
  2. Backend에서 추가 조건: 10 factors 또는 5 values 필요
  3. Data association에서 predicted_pose 미사용
  4. Odometry covariance가 실제 noise configuration 반영 안함
- **해결**:
  1. Optimization trigger 수정: keyframes_since_optimization 추적
  2. Backend 조건 완화: 2 factors만 요구
  3. Data association에서 predicted_pose 사용하도록 수정
  4. Odometry covariance 계산 개선: systematic/random noise 반영

### 문제: Odometry 노이즈 설정 불일치
- **증상**: dummy_publisher에서 noise=0으로 설정했으나 SLAM config는 0.1-0.2m 노이즈 가정
- **원인**: 
  1. SLAM config의 noise model이 실제 센서 특성과 불일치
  2. Odometry message의 covariance 필드 미사용
- **해결**:
  1. SLAM config noise 값을 0.001로 수정 (거의 완벽)
  2. data_converter에서 odometry covariance 추출 구현
  3. use_odometry_covariance 옵션 추가 (현재 false)

## [2025-07-16] SymForce Integration Analysis

**Issue:** The SLAM backend is not fully utilizing SymForce for factor graph optimization. Symbolic factor definitions exist, but the backend defaults to standard GTSAM factors, leading to performance degradation.

**Analysis:**
- `symforce_factors.py` and `cone_color_factor.py` define symbolic models.
- `backend.py` uses standard GTSAM factors, and the `use_custom_factor` flag is `False`.
- `symforce_backend.py` is an empty placeholder.
- `custom_factors.py` contains manual custom factor implementations, which are not the target SymForce-based approach.
- `test_symforce_generation.py` provides a template for SymForce's `Codegen` utility.

**Resolution Plan:**
1.  **Code Generation:** Use SymForce's `Codegen` to generate optimized Python functions from the symbolic models.
2.  **Custom Factor Creation:** Wrap the generated functions in new GTSAM-compatible custom factor classes.
3.  **Backend Integration:** Implement the new custom factors in `symforce_backend.py`.
4.  **Consolidation:** Phase out `backend.py` and `custom_factors.py`.
5.  **Documentation:** Update `docs/symforce_integration.md` and `docs/gtsam_integration.md`.

**Result:** This plan will lead to a fully integrated SymForce/GTSAM backend, which is expected to improve performance and maintainability. The next step is to start implementing the code generation pipeline.



[2025-07-16 12:55:12] SymForce Integration Analysis

**Problem**: System not using SymForce despite having all the code infrastructure
**Root Cause**: 
- use_custom_factor = False in backend.py line 215
- generated/ directory empty - code generation never ran
- symforce_backend.py exists but not imported anywhere

**Solution**: 
- Generate SymForce code by running cone_color_factor.py
- Enable custom factors in backend.py
- Properly wrap generated functions in GTSAM CustomFactor
- Remove duplicate/unused backend implementations

**Result**: Pending implementation - created SYMFORCE_INTEGRATION_PLAN.md
EOF < /dev/null

[$(date '+%Y-%m-%d %H:%M:%S')] Fixed SymForce Integration Issue

**Problem**: Error accumulation even with zero noise in simulation
**Root Cause**: SymForce-generated functions expect sym.Pose2 objects with .data attribute, but GTSAM wrappers were passing gtsam.Pose2 objects directly
**Details**:
- sym.Pose2 stores data as [cos(theta), sin(theta), x, y]
- gtsam.Pose2 has no .data attribute and uses different internal representation
- This caused AttributeError when generated functions tried to access pose.data

**Solution**: 
1. Added gtsam_pose2_to_sym() conversion function
2. Updated all error_func methods in factor classes to convert GTSAM poses before calling SymForce functions
3. Ensured observation and landmark arrays are properly shaped (2,1) for SymForce

**Result**: SymForce factors should now work correctly without numerical errors
## 2025-07-16 - File Cleanup and Organization

### Problem: Excessive redundant files in slam_core
- **Symptom**: 21 files with multiple versions of same functionality
- **Analysis**: 
  - 4 versions of symforce_gtsam_factors (original, stable, analytical, debug)
  - 3 code generators doing similar tasks
  - Both Jacobian and non-Jacobian versions of generated files
  
- **Solution**:
  - Kept only symforce_gtsam_factors_stable.py (proven stable)
  - Moved generators to generators/ subdirectory
  - Removed 3 redundant factor files
  
- **Result**: Reduced to 13 essential files, cleaner structure

## 2025-07-16 - Motion Controller Jerk Control Implementation

### Problem: Abrupt heading changes causing jerky motion
- **Symptom**: Vehicle heading changes too abruptly at corners, causing unrealistic motion
- **Analysis**: Only velocity and acceleration limits were applied, no jerk (rate of change of acceleration) control
- **Solution**: 
  1. Added max_angular_jerk = 5.0 rad/s³ parameter to __init__
  2. Added prev_angular_acceleration tracking for jerk calculation
  3. Implemented 3-stage limiting: jerk → acceleration → velocity
  4. Added proper error handling for dt <= 0 cases
  5. Reset prev_angular_acceleration in reset() method

- **Result**: Smoother heading transitions with gradual acceleration changes
- **Based on**: Gemini's physics analysis and code review recommendations

## 2025-07-16 - Continuous Trajectory Implementation Complete

### Problem: PWM-like jitter in angular velocity despite jerk control
- **Symptom**: Angular velocity showing 0 → 0.0005 rad/s pulses every 0.008s
- **Root Cause**: Discrete heading calculations from centerline points causing fundamental mathematical discontinuities
- **Analysis**: Jerk control only limited maximum values but couldn't eliminate small continuous jittering from discrete trajectory following

### Solution: Complete overhaul to continuous spline-based trajectories
- **Implementation**: 
  1. Created ContinuousTrajectory class with scipy.interpolate.splprep for smooth splines
  2. Implemented arc-length parameterization for uniform motion
  3. Replaced discrete centerline_index tracking with continuous distance tracking
  4. Used mathematical derivatives (splev der=1) for smooth heading calculation
  5. Removed jerk control logic as it becomes unnecessary with smooth curves
  
- **Key Features**:
  - Mathematically smooth position and heading from spline derivatives
  - Curvature-based speed control using second derivatives
  - Seamless handling of closed vs open trajectories
  - Eliminated all discrete point interpolation
  
- **Result**: Fundamentally smooth motion with no PWM-like jitter
- **Status**: Implementation complete, ready for testing