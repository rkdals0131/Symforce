# GTSAM 통합 상세 설계 (Python 기반)

## 1. 개요

본 문서는 CC-SLAM-SYM에서 GTSAM(Georgia Tech Smoothing and Mapping) 라이브러리의 Python 래퍼를 활용한 Factor Graph 기반 최적화의 상세 설계를 다룹니다.

## 2. Factor Graph 구조 설계

### 2.1 전체 그래프 구조

```
Variables (노드):
- X_i: 로봇 포즈 (gtsam.Pose2) at time i
- L_j: 랜드마크 위치 (gtsam.Point2) for landmark j
- V_i: 로봇 속도 (numpy.ndarray) at time i (IMU 사용 시)
- B_i: IMU 바이어스 (gtsam.imuBias_ConstantBias) at time i

Factors (엣지):
- Prior factors: 초기 상태
- Odometry factors: 연속 포즈 간 제약
- Landmark factors: 포즈-랜드마크 관측
- IMU factors: IMU 사전적분
- GPS factors: 절대 위치 제약
- Loop closure factors: 루프 제약
```

### 2.2 Variable 명명 규칙

```python
import gtsam

# 포즈: X0, X1, X2, ...
def pose_key(idx):
    return gtsam.symbol(ord('x'), idx)

# 랜드마크: L0, L1, L2, ...
def landmark_key(idx):
    return gtsam.symbol(ord('l'), idx)

# 속도: V0, V1, V2, ...
def velocity_key(idx):
    return gtsam.symbol(ord('v'), idx)

# IMU 바이어스: B0, B1, B2, ...
def bias_key(idx):
    return gtsam.symbol(ord('b'), idx)
```

## 3. Factor 구현 상세 (Python 예시)

### 3.1 Prior Factor

초기 포즈 또는 고정된 랜드마크에 대한 절대적 제약입니다.

```python
import numpy as np

# 초기 포즈 Prior
def add_initial_pose_prior(graph, initial_pose, sigmas=np.array([0.1, 0.1, 0.05])):
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    graph.add(gtsam.PriorFactorPose2(pose_key(0), initial_pose, prior_noise))

# 랜드마크 Prior (시작/종료 라인 등)
def add_landmark_prior(graph, landmark_id, position, sigma=0.05):
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    graph.add(gtsam.PriorFactorPoint2(landmark_key(landmark_id), position, prior_noise))
```

### 3.2 Odometry Factor

연속된 포즈 간의 상대적 움직임 제약입��다.

```python
def add_odometry_factor(graph, from_idx, to_idx, odometry_pose, noise_params):
    # 적응적 노이즈 모델 (이동 거리에 비례)
    distance = odometry_pose.translation().norm()
    rotation = abs(odometry_pose.rotation().theta())

    sigmas = np.array([
        noise_params['sigma_x'] * (1.0 + noise_params['scale_x'] * distance),
        noise_params['sigma_y'] * (1.0 + noise_params['scale_y'] * distance),
        noise_params['sigma_theta'] * (1.0 + noise_params['scale_theta'] * rotation)
    ])
    
    noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    graph.add(gtsam.BetweenFactorPose2(
        pose_key(from_idx), pose_key(to_idx), odometry_pose, noise
    ))
```

### 3.3 Landmark Observation Factor

로봇 포즈에서 랜드마크 관측에 대한 제약입니다. `gtsam.CustomFactor`를 상속받아 Python으로 직접 커스텀 팩터를 정의할 수 있습니다.

```python
class ConeObservationFactor(gtsam.CustomFactor):
    def __init__(self, pose_k, landmark_k, measured_point, model):
        super().__init__(model, [pose_k, landmark_k], self.error_func)
        self.measured = measured_point

    def error_func(self, values, H_list=None):
        pose = values.atPose2(self.keys()[0])
        landmark = values.atPoint2(self.keys()[1])
        
        if H_list is not None:
            H1 = np.zeros((2, 3))
            H2 = np.zeros((2, 2))
            predicted = pose.transformTo(landmark, H1, H2)
            H_list[0] = H1
            H_list[1] = H2
        else:
            predicted = pose.transformTo(landmark)

        error = predicted - self.measured
        return error
```
또는 간단하게 내장 팩터를 사용할 수 있습니다.
```python
def add_landmark_factor(graph, pose_idx, landmark_id, measured_br, noise_model):
    # measured_br: gtsam.BearingRange2D
    graph.add(gtsam.BearingRangeFactor2D(
        pose_key(pose_idx), landmark_key(landmark_id),
        measured_br.bearing(), measured_br.range(), noise_model
    ))
```

### 3.4 IMU Factor

IMU 사전적분을 사용한 연속 상태 간 제약입니다.

```python
class ImuIntegration:
    def __init__(self, params):
        self.preintegrated = gtsam.PreintegratedImuMeasurements(params)

    def add_measurement(self, imu_data, dt):
        self.preintegrated.integrateMeasurement(
            imu_data.linear_acceleration,
            imu_data.angular_velocity,
            dt
        )

    def add_to_graph(self, graph, from_idx, to_idx):
        graph.add(gtsam.ImuFactor(
            pose_key(from_idx), velocity_key(from_idx),
            pose_key(to_idx), velocity_key(to_idx),
            bias_key(from_idx),
            self.preintegrated
        ))
        # ... 바이어스 제약 추가 ...
```

### 3.5 GPS Factor

RTK-GPS의 절대 위치 제약입니다.

```python
def add_gps_factor(graph, pose_idx, gps_position, noise_model):
    # gtsam.GPSFactor는 3D용이므로, 2D SLAM에서는 PriorFactor를 활용
    # gtsam.Pose2(x, y, theta)
    gps_pose = gtsam.Pose2(gps_position[0], gps_position[1], 0) 
    # GPS는 방향 정보가 없으므로, 위치에 대한 Prior로 추가하되, 방향에 대한 노이즈는 매우 크게 설정
    gps_noise_sigmas = np.array([noise_model.sigmas()[0], noise_model.sigmas()[1], 1e9])
    gps_noise = gtsam.noiseModel.Diagonal.Sigmas(gps_noise_sigmas)
    graph.add(gtsam.PriorFactorPose2(pose_key(pose_idx), gps_pose, gps_noise))
```

## 4. ISAM2 증분 최적화

### 4.1 ISAM2 설정

```python
class SlamOptimizer:
    def __init__(self):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.setRelinearizeSkip(10)
        self.isam2 = gtsam.ISAM2(params)
        self.current_estimate = gtsam.Values()

    def update(self, new_factors, new_values):
        result = self.isam2.update(new_factors, new_values)
        self.current_estimate = self.isam2.calculateEstimate()
```

## 5. 강건성을 위한 기법

### 5.1 Robust Kernel

```python
# Huber robust kernel
base_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
huber_noise = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber.Create(1.345),
    base_noise
)
```

## 6. 구현 예시

### 6.1 전체 파이프라인

```python
class GTSAMBackend:
    def __init__(self):
        self.optimizer = SlamOptimizer()
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()

    def process_keyframe(self, keyframe, matches):
        # 1. 오도메트리 팩터 추가
        # ...
        
        # 2. 랜드마크 관측 팩터 추가
        for match in matches:
            # ...
            add_landmark_factor(...)

        # 3. 초기값 설정
        self.new_values.insert(pose_key(keyframe.id), keyframe.pose)

        # 4. ISAM2 업데이트
        self.optimizer.update(self.new_factors, self.new_values)
        self.new_values.clear()
        self.new_factors.resize(0)
```

## 7. 디버깅 및 시각화

### 7.1 Factor Graph 시각화

```python
def visualize_factor_graph(graph, values):
    # GraphViz dot 파일 생성
    dot_string = graph.dot(values)
    with open("factor_graph.dot", "w") as f:
        f.write(dot_string)
```

### 7.2 공분산 추출

```python
def extract_pose_covariance(isam2, pose_idx):
    try:
        return isam2.marginalCovariance(pose_key(pose_idx))
    except Exception as e:
        print(f"Cannot compute marginal covariance: {e}")
        return np.eye(3) * 1e6
```

## 8. 성능 최적화 팁

1. **배치 업데이트**: 매 프레임마다 최적화하지 않고 키프레임 단위로
2. **변수 순서**: 시간 순서대로 변수 추가 (ISAM2 효율성)
3. **스파스성 활용**: 불필요한 팩터 연결 최소화
4. **Numba/Cython**: 순수 Python으로 구현된 계산 집약적 로직(에러 함수 등) 가속화

## 9. 주의사항

1. **좌표계 일관성**: 모든 데이터가 동일한 좌표계 사용 확인
2. **시간 동기화**: 센서 간 정확한 시간 동기화 필수
3. **초기값 품질**: GTSAM은 비선형 최적화이므로 좋은 초기값 중요
4. **수치 안정성**: 매우 작거나 큰 값 피하기
