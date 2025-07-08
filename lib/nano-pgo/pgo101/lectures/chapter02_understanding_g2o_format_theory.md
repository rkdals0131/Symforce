# PGO 101 - Chapter 2 이론 강의: SLAM 데이터의 표준어, g2o 포맷

**강의 목표:** 이 강의를 마치면, 여러분은 SLAM 커뮤니티에서 표준처럼 사용되는 `g2o` 파일 포맷의 구조를 완벽하게 이해하게 됩니다. 더 나아가 그래프 기반 SLAM의 수학적 기초를 탄탄히 다지고, **VERTEX** 와 **EDGE** 가 각각 무엇을 의미하는지, 그리고 측정의 신뢰도를 나타내는 **Information Matrix** 가 최적화에 어떤 영향을 미치는지 깊이 있게 설명할 수 있게 됩니다. 이 강의는 `chapter02_understanding_g2o_format.ipynb` 실습에서 실제 데이터셋을 파싱하고 분석하기 위한 필수적인 이론적 기반을 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 SLAM 문제를 그래프로 표현하는가?
> - g2o 포맷이 SLAM 커뮤니티의 표준이 된 이유는?
> - Information Matrix는 단순한 가중치 이상의 어떤 의미를 가지는가?
> - 대규모 SLAM 문제를 효율적으로 풀 수 있는 비결은?

---

## 목차

1. [그래프 기반 SLAM의 등장 배경](#1-그래프-기반-slam의-등장-배경)
2. [그래프 기반 SLAM의 수학적 정식화](#2-그래프-기반-slam의-수학적-정식화)
3. [g2o 포맷: 그래프를 파일로 저장하는 약속](#3-g2o-포맷-그래프를-파일로-저장하는-약속)
4. [Information Matrix의 깊은 이해](#4-information-matrix의-깊은-이해)
5. [다양한 Edge 타입과 응용](#5-다양한-edge-타입과-응용)
6. [희소성을 활용한 효율적 계산](#6-희소성을-활용한-효율적-계산)
7. [실제 SLAM 시스템에서의 활용](#7-실제-slam-시스템에서의-활용)
8. [요약 및 다음 장 예고](#8-요약-및-다음-장-예고)

---

## 1. 그래프 기반 SLAM의 등장 배경

### 1.1 Filter 기반 SLAM의 한계

초기 SLAM 연구는 주로 **Extended Kalman Filter (EKF)** 나 **Particle Filter** 같은 필터 기반 방법을 사용했습니다.

**필터 기반 SLAM의 특징:**
- 매 시간 단계마다 새로운 측정값으로 상태를 업데이트
- 과거의 모든 정보를 현재 상태에 "압축"하여 저장
- 계산 복잡도: $O(n^2)$ (n은 랜드마크 수)

**근본적인 문제점:**
1. **선형화 오차 누적**: 비선형 시스템을 반복적으로 선형화하면서 오차가 누적
2. **계산 복잡도**: 랜드마크가 증가할수록 공분산 행렬이 커짐
3. **일관성 문제**: 한번 잘못된 추정은 되돌리기 어려움
4. **Loop Closure 처리**: 과거 위치로 돌아왔을 때 전체 지도 수정이 어려움

### 1.2 그래프 기반 접근법의 혁신

2000년대 중반부터 **그래프 기반 SLAM (Graph-based SLAM)** 이 주목받기 시작했습니다.

**핵심 아이디어**: 
> "실시간으로 모든 것을 추정하려 하지 말고, 일단 데이터를 모아서 전체적으로 최적화하자!"

**그래프 표현의 장점:**
1. **직관성**: 로봇의 경로와 관측을 노드와 엣지로 표현
2. **유연성**: 다양한 센서와 제약조건을 쉽게 추가
3. **전역 최적화**: 모든 제약을 동시에 고려하여 최적해 도출
4. **희소성 활용**: 효율적인 계산 가능

### 1.3 역사적 맥락

**주요 마일스톤:**
- 1997: Lu & Milios - 최초의 그래프 기반 SLAM 제안
- 2006: Grisetti et al. - TORO (Tree-based netwORk Optimizer)
- 2011: Kümmerle et al. - g2o 프레임워크 발표
- 현재: 대부분의 현대 SLAM 시스템이 그래프 기반 접근법 채택

---

## 2. 그래프 기반 SLAM의 수학적 정식화

### 2.1 기본 개념

그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 는 다음으로 구성됩니다:

- **정점 (Vertices)** $\mathcal{V}$: 추정하고자 하는 상태 변수
  - 로봇 포즈: $\mathbf{x}_i \in SE(2)$ 또는 $SE(3)$
  - 랜드마크: $\mathbf{l}_j \in \mathbb{R}^2$ 또는 $\mathbb{R}^3$

- **간선 (Edges)** $\mathcal{E}$: 정점 간의 제약 조건
  - 측정값: $\mathbf{z}_{ij}$
  - 정보 행렬: $\Omega_{ij}$

### 2.2 비선형 최소 제곱 문제로의 정식화

SLAM 문제는 다음의 **Maximum a Posteriori (MAP)** 추정 문제로 표현됩니다:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} p(\mathbf{x} | \mathbf{z})$$

Bayes 정리와 가우시안 노이즈 가정을 통해, 이는 다음의 **비선형 최소 제곱 문제**로 변환됩니다:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{(i,j) \in \mathcal{E}} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j)^T \Omega_{ij} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j)$$

여기서:
- $\mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{z}_{ij} \boxminus h_{ij}(\mathbf{x}_i, \mathbf{x}_j)$: 오차 함수
- $h_{ij}(\mathbf{x}_i, \mathbf{x}_j)$: 예측 함수
- $\boxminus$: 매니폴드 상에서의 차이 연산

### 2.3 오차 함수의 구체적 형태

**SE(2) Pose-Pose Edge:**
```
예측: T_ij = T_i^{-1} * T_j
측정: Z_ij
오차: e_ij = Log(Z_ij^{-1} * T_ij)
```

수식으로 표현하면:
$$\mathbf{e}_{ij} = \log((\mathbf{z}_{ij})^{-1} \cdot \mathbf{x}_i^{-1} \cdot \mathbf{x}_j)_\vee$$

여기서 $\log(\cdot)_\vee$ 는 SE(2)에서 se(2)로의 logarithm map입니다.

**SE(3) Pose-Pose Edge:**
동일한 구조이지만 6차원 오차 벡터를 생성합니다:
$$\mathbf{e}_{ij} = \log((\mathbf{z}_{ij})^{-1} \cdot \mathbf{x}_i^{-1} \cdot \mathbf{x}_j)_\vee \in \mathbb{R}^6$$

### 2.4 최적화 알고리즘

**Gauss-Newton 방법:**

1. **선형화**: 현재 추정치 $\mathbf{x}^{(k)}$ 에서 Taylor 전개
   $$\mathbf{e}_{ij}(\mathbf{x}^{(k)} + \Delta\mathbf{x}) \approx \mathbf{e}_{ij} + \mathbf{J}_{ij}\Delta\mathbf{x}$$

2. **선형 시스템 구성**:
   $$\mathbf{H}\Delta\mathbf{x} = -\mathbf{b}$$
   
   여기서:
   - $\mathbf{H} = \sum_{ij} \mathbf{J}_{ij}^T \Omega_{ij} \mathbf{J}_{ij}$ (Information matrix)
   - $\mathbf{b} = \sum_{ij} \mathbf{J}_{ij}^T \Omega_{ij} \mathbf{e}_{ij}$ (Information vector)

3. **업데이트**:
   $$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} \boxplus \Delta\mathbf{x}$$

**Levenberg-Marquardt 방법:**

더 안정적인 수렴을 위해 damping factor $\lambda$ 를 추가:
$$(\mathbf{H} + \lambda\mathbf{I})\Delta\mathbf{x} = -\mathbf{b}$$

$\lambda$ 는 trust region의 크기를 조절하며, 적응적으로 조정됩니다.

---

## 3. g2o 포맷: 그래프를 파일로 저장하는 약속

### 3.1 g2o의 탄생과 의미

**g2o (General Graph Optimization)** 는 2011년 Rainer Kümmerle 등이 개발한 C++ 프레임워크입니다.

**설계 철학:**
- **일반성**: 다양한 SLAM 문제에 적용 가능
- **효율성**: 희소 행렬 연산 최적화
- **확장성**: 새로운 vertex/edge 타입 쉽게 추가
- **표준화**: 커뮤니티가 공유할 수 있는 포맷

### 3.2 파일 구조 상세

#### VERTEX 정의

**2D 포즈 (SE2):**
```
VERTEX_SE2 id x y theta
```
- `id`: 고유 식별자 (음이 아닌 정수)
- `x, y`: 2D 위치 [미터]
- `theta`: 방향 [라디안]

**예시:**
```
VERTEX_SE2 0 0.0 0.0 0.0        # 시작 위치
VERTEX_SE2 1 1.0 0.0 0.0        # 1m 전진
VERTEX_SE2 2 1.0 1.0 1.5708     # 90도 회전 후 위치
```

**3D 포즈 (SE3:QUAT):**
```
VERTEX_SE3:QUAT id x y z qx qy qz qw
```
- `x, y, z`: 3D 위치 [미터]
- `qx, qy, qz, qw`: 쿼터니언 (Hamilton convention)

**3D 랜드마크:**
```
VERTEX_POINTXYZ id x y z
```

#### EDGE 정의

**2D 상대 포즈 측정:**
```
EDGE_SE2 id1 id2 dx dy dtheta info_11 info_12 info_13 info_22 info_23 info_33
```

**Information matrix 저장 방식:**
3×3 대칭 행렬의 상삼각 부분만 저장:
```
      [info_11  info_12  info_13]
Ω =   [    *    info_22  info_23]
      [    *       *     info_33]
```

**3D 상대 포즈 측정:**
```
EDGE_SE3:QUAT id1 id2 dx dy dz dqx dqy dqz dqw info(1,1) info(1,2) ... info(6,6)
```
6×6 행렬의 21개 원소를 저장합니다.

### 3.3 확장 포맷들

**Fixed Vertex:**
```
FIX 0    # 0번 vertex를 고정 (global reference)
```

**Prior Edge:**
```
EDGE_SE2_PRIOR id x y theta info_11 ... info_33
```

**Landmark Observation:**
```
EDGE_SE2_XY pose_id landmark_id observation_x observation_y info_11 info_12 info_22
```

### 3.4 다른 포맷과의 비교

| 포맷 | 장점 | 단점 | 사용처 |
|------|------|------|---------|
| g2o | 표준화, 다양한 edge 타입 | 텍스트 기반 (큰 파일) | 대부분의 SLAM 연구 |
| TORO | 간단함, 빠른 파싱 | SE(2)만 지원 | 2D SLAM |
| GTSAM | 프로그래밍 API 우수 | 파일 포맷 복잡 | 실시간 시스템 |
| ROS bag | 시간 정보 포함 | SLAM 특화 아님 | ROS 기반 시스템 |

---

## 4. Information Matrix의 깊은 이해

### 4.1 확률론적 관점

**공분산과 정보의 이중성:**

측정의 불확실성은 두 가지 방식으로 표현할 수 있습니다:

1. **공분산 행렬** $\Sigma$: 불확실성의 "크기"
   - 대각 원소: 각 차원의 분산
   - 비대각 원소: 차원 간 상관관계

2. **정보 행렬** $\Omega = \Sigma^{-1}$: 정보의 "양"
   - 큰 값 = 정확한 측정
   - 작은 값 = 불확실한 측정

**왜 Information Matrix를 사용하는가?**
1. 정보는 **가산적(additive)**: $\Omega_{total} = \Omega_1 + \Omega_2$
2. 최적화에서 자연스러운 가중치 역할
3. Marginalization이 효율적

### 4.2 Fisher Information과의 연결

Information Matrix는 **Fisher Information Matrix**와 밀접한 관련이 있습니다:

$$\mathcal{I}(\theta) = -\mathbb{E}\left[\frac{\partial^2 \log p(x|\theta)}{\partial\theta^2}\right]$$

이는 **Cramér-Rao lower bound**와 연결됩니다:
$$\text{Var}(\hat{\theta}) \geq \mathcal{I}^{-1}(\theta)$$

즉, Information Matrix가 클수록 추정의 이론적 하한이 작아집니다.

### 4.3 실제 예시: 센서별 Information Matrix

**휠 오도메트리 (평지):**
```python
Ω_wheel = diag([100, 100, 10, inf, inf, 50])
# x,y: 정확, z: 불가능(inf), roll,pitch: 불가능, yaw: 보통
```

**Visual Odometry (단안 카메라):**
```python
Ω_visual = diag([50, 50, 10, 20, 20, 30])
# 위치: 보통, 회전: 상대적으로 부정확, 스케일 드리프트 존재
```

**LiDAR (3D):**
```python
Ω_lidar = diag([500, 500, 500, 100, 100, 100])
# 위치: 매우 정확, 회전: 정확
```

**IMU 통합:**
```python
Ω_imu = diag([1, 1, 10, 1000, 1000, 100])
# 위치: 드리프트, 회전: 매우 정확 (자이로스코프)
```

### 4.4 Information Matrix의 기하학적 의미

Information Matrix의 고유값 분해:
$$\Omega = V\Lambda V^T$$

- **고유값** $\lambda_i$: 각 주축 방향의 정보량
- **고유벡터** $v_i$: 주축의 방향

이는 **불확실성 타원체(uncertainty ellipsoid)**로 시각화됩니다:
- 타원체의 축 길이: $1/\sqrt{\lambda_i}$
- 축의 방향: 고유벡터 $v_i$

### 4.5 Marginalization과 Schur Complement

큰 시스템을 작은 시스템으로 축소할 때 사용:

원래 시스템:
$$\begin{bmatrix} \mathbf{H}_{aa} & \mathbf{H}_{ab} \\ \mathbf{H}_{ba} & \mathbf{H}_{bb} \end{bmatrix} \begin{bmatrix} \Delta\mathbf{x}_a \\ \Delta\mathbf{x}_b \end{bmatrix} = \begin{bmatrix} \mathbf{b}_a \\ \mathbf{b}_b \end{bmatrix}$$

$\mathbf{x}_a$ 를 marginalize out:
$$(\mathbf{H}_{bb} - \mathbf{H}_{ba}\mathbf{H}_{aa}^{-1}\mathbf{H}_{ab})\Delta\mathbf{x}_b = \mathbf{b}_b - \mathbf{H}_{ba}\mathbf{H}_{aa}^{-1}\mathbf{b}_a$$

**Schur complement** $\mathbf{S} = \mathbf{H}_{bb} - \mathbf{H}_{ba}\mathbf{H}_{aa}^{-1}\mathbf{H}_{ab}$ 는 $\mathbf{x}_a$ 의 불확실성이 $\mathbf{x}_b$ 에 전파된 결과입니다.

---

## 5. 다양한 Edge 타입과 응용

### 5.1 기본 Edge 타입들

**1. Odometry Edge**
- 연속된 포즈 간 상대 변환
- 높은 빈도, 작은 오차
- 드리프트 누적 문제

**2. Loop Closure Edge**
- 시간적으로 멀지만 공간적으로 가까운 포즈 연결
- 낮은 빈도, 큰 영향력
- 전역 일관성 확보

**3. Landmark Observation Edge**
- 포즈에서 랜드마크로의 관측
- Bundle Adjustment의 핵심
- 특징: 방향성 있는 엣지

### 5.2 고급 Edge 타입들

**4. GPS/GNSS Edge**
```
EDGE_SE2_PRIOR pose_id gps_x gps_y 0 info_xx info_xy info_yy 0 0 0
```
- 전역 위치 제약
- 회전 정보 없음 (info_theta = 0)
- 실내에서는 사용 불가

**5. IMU Preintegration Edge**
```
EDGE_SE3:IMU id1 id2 dt preint_values... info...
```
- 고주파 IMU 측정을 통합
- 중력 방향 제약 제공
- Visual-Inertial SLAM의 핵심

**6. Plane/Line Constraint**
- 구조적 제약 활용
- Manhattan world assumption
- 실내 환경에서 유용

### 5.3 Robust Kernel Functions

이상치(outlier)에 강인한 최적화를 위해:

**Huber Loss:**
$$\rho_H(e) = \begin{cases} \frac{1}{2}e^2 & |e| \leq \delta \\ \delta|e| - \frac{1}{2}\delta^2 & |e| > \delta \end{cases}$$

**Cauchy Loss:**
$$\rho_C(e) = \frac{c^2}{2}\log\left(1 + \frac{e^2}{c^2}\right)$$

**구현 예시:**
```
EDGE_SE2 0 1 1.0 0.0 0.0 info... ROBUST_KERNEL Huber 0.1
```

### 5.4 Dynamic Covariance Scaling

측정 품질에 따라 Information Matrix 동적 조정:

```python
def compute_information(measurement_quality):
    base_info = np.diag([100, 100, 50])
    
    # 품질 지표에 따른 스케일링
    if measurement_quality < 0.5:
        scale = 0.1  # 낮은 신뢰도
    elif measurement_quality < 0.8:
        scale = 0.5  # 보통 신뢰도
    else:
        scale = 1.0  # 높은 신뢰도
    
    return scale * base_info
```

---

## 6. 희소성을 활용한 효율적 계산

### 6.1 SLAM의 희소 구조

SLAM의 Information Matrix $\mathbf{H}$ 는 매우 희소(sparse)합니다:

**이유:**
1. 각 포즈는 인접 포즈와만 연결
2. 대부분의 랜드마크는 몇 개 포즈에서만 관측
3. Loop closure는 전체 엣지의 작은 부분

**희소성 정도:**
- Dense: $O(n^2)$ 원소
- SLAM: $O(n)$ 비영 원소 (n은 변수 개수)

### 6.2 Variable Ordering의 중요성

**Fill-in 현상:**
Cholesky 분해 중 0이었던 원소가 0이 아닌 값으로 바뀌는 현상

**최적 ordering 전략:**
1. **Minimum Degree**: 연결이 적은 노드부터
2. **Nested Dissection**: 그래프를 재귀적으로 분할
3. **COLAMD**: Column Approximate Minimum Degree

**예시: 1D SLAM**
```
나쁜 순서: x0, l0, l1, ..., ln, x1, x2, ..., xm
좋은 순서: x0, x1, ..., xm, l0, l1, ..., ln
```

### 6.3 계산 복잡도 분석

| 방법 | 시간 복잡도 | 공간 복잡도 | 조건 |
|------|------------|------------|------|
| Dense Cholesky | $O(n^3)$ | $O(n^2)$ | - |
| Sparse Cholesky | $O(n^{3/2})$ | $O(n\log n)$ | 2D 격자 |
| Sparse Cholesky | $O(n)$ | $O(n)$ | 1D 체인 |
| Iterative (PCG) | $O(n\sqrt{\kappa})$ | $O(n)$ | $\kappa$: 조건수 |

### 6.4 실제 구현 기법

**1. Sparse Matrix 저장:**
```cpp
// Compressed Column Storage (CCS)
struct SparseMatrix {
    vector<double> values;     // 비영 원소 값
    vector<int> row_indices;   // 행 인덱스
    vector<int> col_pointers;  // 열 시작 위치
};
```

**2. Block 구조 활용:**
```cpp
// 3x3 또는 6x6 블록 단위 연산
Matrix3d block = H.block<3,3>(i*3, j*3);
```

**3. 증분적 최적화:**
- iSAM2: Incremental Smoothing and Mapping
- 새로운 측정값만 처리
- Bayes Tree 구조 활용

---

## 7. 실제 SLAM 시스템에서의 활용

### 7.1 대표적인 SLAM 시스템들

**Visual SLAM:**
- **ORB-SLAM3**: 실시간 단안/스테레오/RGB-D SLAM
  - Frontend: ORB 특징점 추출 및 매칭
  - Backend: g2o 기반 Bundle Adjustment
  - Loop Closure: DBoW2 vocabulary

- **DSO (Direct Sparse Odometry)**:
  - Direct method (특징점 없음)
  - Photometric Bundle Adjustment
  - g2o로 키프레임 그래프 최적화

**LiDAR SLAM:**
- **LOAM (LiDAR Odometry And Mapping)**:
  - Point-to-edge, Point-to-plane 매칭
  - 두 단계 최적화 (고주파/저주파)
  - g2o로 백엔드 최적화

- **LeGO-LOAM**:
  - Ground 분리를 통한 효율성
  - Loop closure detection
  - Pose graph optimization

### 7.2 Multi-Robot SLAM

**분산 SLAM의 도전과제:**
1. 로봇 간 상대 포즈 추정
2. 맵 정합 (Map Alignment)
3. 통신 제약

**g2o 활용:**
```
# Robot 1 trajectory
VERTEX_SE2 1000 ...
VERTEX_SE2 1001 ...

# Robot 2 trajectory  
VERTEX_SE2 2000 ...
VERTEX_SE2 2001 ...

# Inter-robot measurements
EDGE_SE2 1050 2030 ... # 로봇 간 만남
```

### 7.3 센서 융합 SLAM

**Visual-Inertial SLAM:**
```python
# Visual constraints
add_visual_edge(pose_i, pose_j, visual_measurement)

# IMU preintegration
add_imu_edge(pose_i, pose_j, imu_preintegration)

# 상호 보완적 특성
# - Vision: 절대 스케일 없음, 저주파
# - IMU: 드리프트, 고주파
```

**LiDAR-Visual-Inertial:**
- 각 센서의 장점 결합
- 다중 해상도 최적화
- Robust to 센서 실패

### 7.4 실시간 성능 최적화

**1. Sliding Window:**
- 최근 N개 키프레임만 최적화
- Marginalization으로 과거 정보 압축

**2. Parallel Processing:**
- Frontend/Backend 분리
- GPU 가속 (특징 추출)
- Multi-threaded 최적화

**3. Adaptive Quality:**
```python
if computation_time > threshold:
    reduce_keyframe_rate()
    use_approximate_solver()
else:
    increase_accuracy()
```

---

## 8. 요약 및 다음 장 예고

### 8.1 핵심 내용 정리

이 장에서 우리는 다음을 배웠습니다:

1. **그래프 기반 SLAM의 우월성**
   - Filter 기반의 한계 극복
   - 전역 최적화 가능
   - 유연한 확장성

2. **수학적 기초**
   - 비선형 최소 제곱 문제
   - Gauss-Newton/Levenberg-Marquardt
   - 매니폴드 상의 최적화

3. **g2o 포맷의 구조**
   - VERTEX와 EDGE 정의
   - Information Matrix 표현
   - 다양한 확장 가능성

4. **Information Matrix의 의미**
   - 측정의 신뢰도 정량화
   - Fisher Information과의 연결
   - Marginalization 도구

5. **계산 효율성**
   - 희소성 활용
   - Variable ordering
   - 실시간 처리 기법

### 8.2 실전 체크리스트

g2o 파일 작업 시 확인사항:

- [ ] 모든 VERTEX ID가 고유한가?
- [ ] EDGE가 참조하는 VERTEX가 존재하는가?
- [ ] Information Matrix가 양정치 행렬인가?
- [ ] 좌표계와 단위가 일관적인가?
- [ ] Loop closure edge가 적절히 포함되었는가?
- [ ] Fixed vertex로 전역 참조가 설정되었는가?

### 8.3 다음 장 예고

**Chapter 3: SymForce를 이용한 Symbolic Computation**

다음 장에서는:
- 수동 vs 자동 미분의 트레이드오프
- SymForce의 symbolic 표현력
- 자동 생성된 Jacobian의 효율성
- 실제 최적화 문제 구현

### 8.4 추가 학습 자료

**논문:**
- Grisetti et al., "A Tutorial on Graph-Based SLAM" (2010)
- Kümmerle et al., "g2o: A General Framework for Graph Optimization" (2011)
- Carlone et al., "Factor Graphs for Robot Perception" (2021)

**오픈소스 프로젝트:**
- g2o: https://github.com/RainerKuemmerle/g2o
- GTSAM: https://github.com/borglab/gtsam
- Ceres Solver: http://ceres-solver.org

**데이터셋:**
- TUM RGB-D: https://vision.in.tum.de/data/datasets/rgbd-dataset
- KITTI: http://www.cvlibs.net/datasets/kitti/
- EuRoC MAV: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

### 8.5 핵심 메시지

> "g2o는 단순한 파일 포맷이 아닙니다. 이는 불확실한 센서 측정들을 체계적으로 정리하고, 전역적으로 일관된 지도를 만들어내는 강력한 프레임워크입니다. VERTEX는 우리가 알고 싶은 '미지수'이고, EDGE는 센서가 제공하는 '단서'이며, Information Matrix는 각 단서의 '신뢰도'입니다. 이 세 요소가 조화롭게 작동할 때, 로봇은 자신이 어디에 있는지 정확히 알 수 있게 됩니다."

이제 여러분은 SLAM 데이터의 표준 언어인 g2o를 완벽히 이해했습니다. 다음 장에서는 이러한 최적화 문제를 더욱 효율적으로 푸는 방법을 배워보겠습니다!