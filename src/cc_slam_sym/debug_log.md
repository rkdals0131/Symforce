# CC-SLAM-SYM Debug Log

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