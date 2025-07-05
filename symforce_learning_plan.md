# Symforce 학습 계획: nano-pgo를 통한 단계별 접근

## 개요

이 문서는 nano-pgo를 활용하여 Symforce와 pose-graph optimization을 체계적으로 학습하는 계획입니다. 
cc_slam_graph 패키지 개발 경험을 바탕으로, g2o-python의 한계를 극복하고 Symforce의 강력한 symbolic 최적화 기능을 습득하는 것이 목표입니다.

## 학습 목표

### 주요 목표
1. **Symforce의 symbolic 계산 이해**: 자동 야코비안 생성 및 최적화
2. **Pose-graph optimization 심화**: SE(3) 공간에서의 최적화 이론과 구현
3. **Robust optimization 기법**: Cauchy kernel, LM 알고리즘 등 실전 기법
4. **Rotation initialization**: Chordal relaxation 기반 초기화
5. **센서 융합 준비**: IMU/GPS 통합을 위한 기초 지식

### cc_slam_graph 개발 의도와의 연계
- **멀티스레딩 최적화**: Symforce의 codegen으로 단일 스레드에서도 고성능 달성
- **실시간 처리**: 컴파일된 야코비안으로 g2o 대비 30배 이상 속도 향상
- **전역 최적화**: Loop closure와 full graph optimization 구현
- **센서 융합**: Pre-integration 기법 이해 및 구현

## 단계별 학습 계획

### Phase 1: 기초 다지기 (1주차)

#### 1.1 개발 환경 구축
```bash
# Python 환경 설정
python3.11 -m venv ~/envs/symforce_env
source ~/envs/symforce_env/bin/activate

# 의존성 설치
pip install "numpy<2.0"
pip install scipy
sudo apt-get install libsuitesparse-dev
pip install scikit-sparse
pip install symforce
pip install matplotlib
pip install open3d
```

#### 1.2 첫 번째 예제: SE(3) 변환 이해
- **목표**: Rotation과 Translation의 표현 방법 이해
- **구현 내용**:
  ```python
  # rotation_basics.py
  - 쿼터니언 ↔ 회전 행렬 ↔ 회전 벡터 변환
  - SE(3) 그룹 연산 (compose, inverse)
  - tangent space와 manifold 개념 이해
  ```
- **nano-pgo 참고**: `quat_to_rotmat()`, `rotvec_to_rotmat()` 함수들

#### 1.3 두 번째 예제: g2o 파일 파싱
- **목표**: Pose-graph 데이터 구조 이해
- **구현 내용**:
  ```python
  # g2o_parser.py
  - VERTEX_SE3 파싱
  - EDGE_SE3 파싱
  - Information matrix 이해
  ```
- **nano-pgo 참고**: `read_g2o_file()` 메서드

### Phase 2: Symforce 핵심 기능 습득 (2주차)

#### 2.1 Symbolic 표현과 자동 미분
- **목표**: Symforce의 symbolic 계산 이해
- **구현 내용**:
  ```python
  # symbolic_basics.py
  - sf.V3, sf.Rot3, sf.Pose3 사용법
  - Between factor residual 정의
  - 자동 야코비안 계산
  ```
- **nano-pgo 참고**: 전역 변수 정의 부분 (`sf_ri`, `sf_ti` 등)

#### 2.2 Codegen을 통한 최적화
- **목표**: 30배 속도 향상 달성
- **구현 내용**:
  ```python
  # codegen_optimization.py
  - codegen.Codegen.function() 사용
  - with_jacobians() 메서드
  - 컴파일된 함수 임포트 및 사용
  ```
- **nano-pgo 참고**: `generate_compiled_between_error_func()` 함수

#### 2.3 실습: 간단한 2D SLAM
- **목표**: 2D 환경에서 전체 파이프라인 구현
- **데이터셋**: `input_INTEL_g2o.g2o` (쉬운 데이터셋)
- **구현 내용**:
  - Pose graph 구축
  - H 행렬과 b 벡터 생성
  - 최적화 실행

### Phase 3: 고급 최적화 기법 (3주차)

#### 3.1 Robust Kernel 구현
- **목표**: 아웃라이어에 강건한 최적화
- **구현 내용**:
  ```python
  # robust_optimization.py
  - Cauchy weight 함수 구현
  - Loop closure vs Odometry edge 구분
  - Adaptive kernel size 조정
  ```
- **nano-pgo 참고**: `cauchy_weight()` 메서드

#### 3.2 Levenberg-Marquardt 알고리즘
- **목표**: 수렴 속도 개선
- **구현 내용**:
  ```python
  # lm_algorithm.py
  - Damping factor (λ) 조정
  - Error 기반 파라미터 업데이트
  - 수렴 조건 구현
  ```
- **nano-pgo 참고**: `adjust_parameters()` 메서드

#### 3.3 실습: 어려운 데이터셋 도전
- **데이터셋**: `sphere2500.g2o` (어려운 데이터셋)
- **목표**: Rotation initialization의 중요성 체험

### Phase 4: Rotation Initialization (4주차)

#### 4.1 Chordal Relaxation 이론
- **목표**: 비선형 회전 최적화를 선형 문제로 변환
- **참고 논문**: Carlone et al., ICRA 2015
- **구현 내용**:
  ```python
  # chordal_relaxation.py
  - 회전 행렬의 row-wise 최적화
  - SVD를 통한 orthogonality 복원
  - Prior 추가로 gauge freedom 해결
  ```

#### 4.2 실습: With/Without 초기화 비교
- **실험 1**: 쉬운 데이터셋 (M3500)
- **실험 2**: 빠른 수렴 확인 (INTEL)
- **실험 3**: 어려운 데이터셋 (sphere2500)
- **nano-pgo 참고**: `relax_rotation()` 메서드

### Phase 5: 대규모 최적화 (5주차)

#### 5.1 Sparse Matrix 처리
- **목표**: 대규모 문제 효율적 해결
- **구현 내용**:
  ```python
  # sparse_optimization.py
  - scipy.sparse 활용
  - cholmod solver 사용
  - H 행렬 sparsity pattern 분석
  ```

#### 5.2 병렬 처리 기법
- **목표**: Multi-core 활용
- **구현 내용**:
  ```python
  # parallel_processing.py
  - Edge별 야코비안 병렬 계산
  - Process pool 활용
  - 결과 취합 및 행렬 조립
  ```
- **nano-pgo 참고**: `process_edge()` 메서드

### Phase 6: 시각화와 평가 (6주차)

#### 6.1 3D 시각화 구현
- **Open3D 활용**: Point cloud와 edge 시각화
- **Plotly 활용**: Interactive 3D plot
- **실시간 업데이트**: 반복 과정 시각화

#### 6.2 성능 평가 메트릭
- **구현 내용**:
  ```python
  # evaluation_metrics.py
  - Total error 계산
  - Chi-squared 통계
  - ATE (Absolute Trajectory Error)
  - RPE (Relative Pose Error)
  ```

### Phase 7: 통합 프로젝트 (7-8주차)

#### 7.1 전역 Loop Closure 시스템
- **목표**: cc_slam_graph의 미구현 기능 완성
- **구현 내용**:
  ```python
  # global_loop_closure.py
  - KD-Tree 기반 후보 탐색
  - 시간 간격 검증
  - Loop closure edge 추가
  - 전체 그래프 재최적화
  ```

#### 7.2 IMU Pre-integration 준비
- **목표**: 센서 융합을 위한 기초
- **학습 내용**:
  - IMU 노이즈 모델
  - Pre-integration 이론
  - Factor graph에서의 IMU factor

## 실습 데이터셋 활용 순서

1. **입문용**: `input_INTEL_g2o.g2o`, `input_M3500_g2o.g2o`
2. **중급용**: `FR079_P_toro.graph`, `parking-garage.g2o`
3. **고급용**: `sphere2500.g2o`, `input_M3500b_g2o.g2o`
4. **도전용**: `grid3D.g2o`, `rim.g2o`

## 예상 학습 성과

### 기술적 성과
- Symforce를 활용한 고성능 SLAM 백엔드 구현 능력
- 자동 미분과 코드 생성을 통한 최적화 가속화
- Robust optimization 기법 실전 적용
- 대규모 pose-graph 처리 능력

### 이론적 성과
- SE(3) manifold 상의 최적화 이해
- Gauge freedom과 prior의 중요성
- Rotation initialization의 필요성
- 센서 융합을 위한 기초 지식

## 추가 학습 자료

### 필수 참고 자료
1. nano-pgo README 및 소스 코드
2. Symforce 공식 문서 (https://symforce.org)
3. Carlone et al., "Initialization techniques for 3D SLAM", ICRA 2015

### 추천 도서/논문
1. "Factor Graphs for Robot Perception" - Dellaert & Kaess
2. "State Estimation for Robotics" - Timothy D. Barfoot
3. GTSAM 라이브러리 문서

## 마무리

이 학습 계획을 통해 g2o-python의 한계를 극복하고, Symforce의 강력한 기능을 활용한 고성능 SLAM 시스템을 구현할 수 있게 됩니다. 특히 자동 야코비안 생성과 코드 최적화를 통해 실시간 처리가 가능한 수준의 성능을 달성할 수 있을 것입니다.