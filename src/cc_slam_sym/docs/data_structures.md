# CC-SLAM-SYM 데이터 구조 상세 명세 (Python 기반)

## 1. 개요

본 문서는 CC-SLAM-SYM 프로젝트에서 사용되는 모든 데이터 구조의 상세 명세를 Python `dataclass`를 기반으로 정의합니다.

## 2. 기본 데이터 구조

### 2.1 ConeCluster

콘 감지 모듈에서 생성되는 원시 관측 데이터입니다.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ConeCluster:
    # 기본 속성
    timestamp: float
    position: np.ndarray  # 로봇 기준 3D 위치 (x, y, z)
    color: str            # 콘 색��: "yellow", "blue", "red", "unknown"
    
    # 추가 속성
    track_id: int = -1
    covariance: np.ndarray = np.eye(3) * 0.1
```

### 2.2 Landmark

맵에 등록된 콘 랜드마크입니다.

```python
from enum import Enum
import gtsam

class LandmarkType(Enum):
    CONE_YELLOW = 0
    CONE_BLUE = 1
    CONE_RED = 2
    UNKNOWN = 3

@dataclass
class Landmark:
    # 식별자
    id: int
    symbol: gtsam.Symbol
    
    # 속성
    position: np.ndarray  # 맵 기준 2D 위치 (x, y)
    color: str
    type: LandmarkType
    
    # 통계 정보
    observation_count: int = 0
    first_seen_timestamp: float = -1.0
    last_seen_timestamp: float = -1.0
    
    # 불확실성
    covariance: np.ndarray = np.eye(2) * 1.0

    def to_gtsam(self) -> gtsam.Point2:
        return gtsam.Point2(self.position[0], self.position[1])
```

### 2.3 Keyframe

SLAM 백엔드에서 사용하는 키프레임입니다.

```python
@dataclass
class Keyframe:
    # 식별자
    id: int
    timestamp: float
    pose_symbol: gtsam.Symbol
    
    # 상태
    pose: gtsam.Pose2
    
    # 센서 데이터
    observations: list[ConeCluster]
    
    # 연결 정보
    observed_landmarks: list[int]
```

## 3. GTSAM 관련 데이터 구조

### 3.1 Factor Graph ��성 요소

```python
import gtsam

class SlamFactorGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

    def add_prior_factor(self, key, prior, noise_model):
        self.graph.add(gtsam.PriorFactorPose2(key, prior, noise_model))

    def add_odometry_factor(self, key1, key2, odometry, noise_model):
        self.graph.add(gtsam.BetweenFactorPose2(key1, key2, odometry, noise_model))

    def add_landmark_factor(self, pose_key, landmark_key, observation, noise_model):
        # 예시: BearingRangeFactor2D 사용
        bearing = observation.bearing()
        range_ = observation.range()
        self.graph.add(gtsam.BearingRangeFactor2D(pose_key, landmark_key, bearing, range_, noise_model))

    def update_isam2(self):
        new_factors = self.graph
        new_values = self.initial_values
        self.isam2.update(new_factors, new_values)
        self.graph.resize(0)
        self.initial_values.clear()
        return self.isam2.calculateEstimate()
```

### 3.2 Noise Models

```python
import gtsam
import numpy as np

# 오도메트리 노이즈 모델
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))

# 랜드마크 관측 노이즈
LANDMARK_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.15])) # bearing, range

# GPS 노이즈 (RTK 정밀도)
GPS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02])) # x, y
```

## 4. 데이터 연관 (Data Association)

### 4.1 연관 결과

```python
@dataclass
class Match:
    landmark_id: int
    observation_idx: int
    distance: float

@dataclass
class DataAssociationResult:
    matches: list[Match]
    new_landmark_indices: list[int]
    outlier_indices: list[int]
```

## 5. Symforce 관련 데이터 구조

### 5.1 Symbolic 표현

```python
# Symforce를 위한 심볼릭 타입 정의
import symforce.symbolic as sf

Pose2 = sf.Pose2
Vector2 = sf.V2
Scalar = sf.Scalar

# 커스텀 팩터를 위한 심볼릭 함수
def cone_landmark_residual(
    robot_pose: Pose2,
    landmark_pos: Vector2,
    observation: Vector2,
    epsilon: Scalar = sf.numeric_epsilon
) -> Vector2:
    # ... 잔차 계산 ...
    return predicted - observation
```

## 6. 시스템 상태

### 6.1 SLAM 상태

```python
@dataclass
class SlamState:
    current_pose: gtsam.Pose2
    current_timestamp: float
    landmarks: dict[int, Landmark]
    keyframes: dict[int, Keyframe]
    
    # 통계
    total_keyframes: int = 0
    total_landmarks: int = 0
```
