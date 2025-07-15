# Symforce 통합 상세 설계 (Python 기반)

## 1. 개요

Symforce는 심볼릭 연산과 자동 미분을 통해 최적화된 코드를 생성하는 라이브러리입니다. CC-SLAM-SYM에서는 복잡한 커스텀 팩터의 야코비안을 Python 내에서 자동으로 생성하고 최적화하는 데 활용합니다.

## 2. Symforce 활용 전략

### 2.1 왜 Symforce인가?

1.  **자동 미분**: 복잡한 야코비안을 Python 코드 상에서 수동으로 유도할 필요가 없습니다.
2.  **코드 생성**: 최적화된 Python 함수를 생성하여, 계산이 많이 필요한 팩터의 성능을 향상시킬 수 있습니다.
3.  **수치 안정성**: Epsilon 처리 등을 자동으로 수행하여 수치적 안정성을 높입니다.
4.  **개발 속도**: 새로운 제약 조건을 Python으로 빠르게 프로토타이핑하고 테스트할 수 있습니다.

### 2.2 GTSAM과의 통합 방식

Symforce로 생성된 Python 함수를 `gtsam.CustomFactor`와 결합하여 사용합니다.

```
Symforce (Python)                GTSAM (Python)
┌───────────────────────┐       ┌────────────────────────┐
│ Symbolic Factor       │       │ gtsam.CustomFactor     │
│ Definition in Python  │  ───> │ (Python Subclass)      │
└───────────────────────┘       └────────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────┐  ┌────────────────────────┐
│ Generated Optimized Python   │  │ Factor Graph에 추가      │
│ Function (Residual, Jacobian)│  │ (graph.add)            │
└──────────────────────────────���  └────────────────────────┘
```

## 3. 커스텀 팩터 정의 (Python)

### 3.1 콘 색상 제약 팩터

콘의 색상 정보를 활용한 데이터 연관 강화 팩터입니다.

```python
# cc_slam_sym/slam_core/symforce_factors/cone_color_factor.py
import symforce.symbolic as sf
from symforce.values import Values
from symforce.codegen import Codegen, PythonConfig

def cone_color_residual(
    pose: sf.Pose2,
    landmark: sf.V2,
    observation: sf.V2,
    observed_color: sf.Scalar,
    landmark_color: sf.Scalar,
    color_weight: sf.Scalar,
    epsilon: sf.Scalar = sf.numeric_epsilon,
) -> sf.V3:
    """
    위치 잔차와 색상 불일치 페널티를 포함하는 3D 잔차를 반환합니다.
    """
    predicted_observation = pose.inverse() * landmark
    position_residual = predicted_observation - observation
    
    color_diff = sf.Abs(observed_color - landmark_color)
    # 색상이 다르면 1, 같으면 0
    is_different = sf.Min(color_diff, 1.0)
    color_residual = color_weight * is_different
    
    return sf.V3(position_residual[0], position_residual[1], color_residual)

def generate_cone_color_factor():
    """
    Cone Color Factor의 최적화된 Python 함수를 생성합니다.
    """
    inputs = Values(
        pose=sf.Pose2(),
        landmark=sf.V2(),
        observation=sf.V2(),
        observed_color=sf.Scalar(),
        landmark_color=sf.Scalar(),
        color_weight=sf.Scalar(),
    )
    
    codegen_obj = Codegen(
        inputs=inputs,
        output_names=["residual", "jacobian"],
        func=cone_color_residual,
        config=PythonConfig(),
    )
    
    metadata = codegen_obj.generate_function(
        output_dir="generated",
        skip_directory_nesting=True
    )
    
    print(f"Generated Python factor: {metadata.generated_files}")
    return metadata
```

## 4. GTSAM 통합 (Python)

### 4.1 생성된 Python 함수를 사용한 커스텀 팩터

Symforce로 생성된 `cone_color_factor` 함수를 `gtsam.CustomFactor`로 래핑합니다.

```python
# cc_slam_sym/slam_core/custom_factors.py
import gtsam
import numpy as np
# from generated.cone_color_factor import cone_color_factor

class SymforceConeColorFactor(gtsam.CustomFactor):
    def __init__(self, pose_key, landmark_key, observation, obs_color, lm_color, color_weight, model):
        super().__init__(model, [pose_key, landmark_key], self.error_func)
        self.observation = observation
        self.obs_color = obs_color
        self.lm_color = lm_color
        self.color_weight = color_weight

    def error_func(self, values, H_list=None):
        pose = values.atPose2(self.keys()[0])
        landmark = values.atPoint2(self.keys()[1])

        # Symforce로 생성된 함수 호출
        if H_list is not None:
            # residual, jacobian = cone_color_factor(
            #     pose, landmark, self.observation, self.obs_color, self.lm_color, self.color_weight
            # )
            # H_list[0] = jacobian[:, :3] # Pose Jacobian
            # H_list[1] = jacobian[:, 3:] # Landmark Jacobian
            # return residual
            pass # Placeholder for Jacobian implementation
        else:
            # residual, _ = cone_color_factor(...)
            # return residual
            pass # Placeholder for residual implementation
        
        # 임시 Python 구현 (Symforce 생성 함수 사용 전)
        predicted = pose.inverse().transformFrom(landmark)
        pos_error = predicted - self.observation
        color_error = self.color_weight if self.obs_color != self.lm_color else 0.0
        return np.array([pos_error[0], pos_error[1], color_error])

```

### 4.2 팩터 사용 예시

```python
# SLAM 백엔드에서 사용
from .custom_factors import SymforceConeColorFactor

def add_cone_observation(graph, pose_id, landmark_id, observation, landmark):
    
    # 노이즈 모델 (위치 2D + 색상 1D)
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 1.0]))
    
    factor = SymforceConeColorFactor(
        pose_key(pose_id),
        landmark_key(landmark_id),
        observation.position,
        observation.color_int,
        landmark.color_int,
        color_weight=10.0, # 튜닝 필요
        model=noise
    )
    graph.add(factor)
```

## 5. 코드 생성 파이프라인

### 5.1 빌드 시스템 통합 (CMake)

`setup.py` 또는 `CMakeLists.txt`에서 코드 생성 스크립트를 빌드 프로세스의 일부로 실행할 수 있습니다.

```python
# setup.py
from setuptools import setup
# ...
# 빌드 전 코드 생성 스크립트 실행 로직 추가
# generate_cone_color_factor()
# ...
setup(...)
```

### 5.2 자동 생성 스크립트

하나의 스크립트에서 모든 Symforce 기반 팩터를 생성하도록 관리합니다.

```python
# symforce_factors/generate_all.py
from .cone_color_factor import generate_cone_color_factor
# from .ackermann_motion_factor import generate_ackermann_factor

def main():
    print("Generating all Symforce Python factors...")
    generate_cone_color_factor()
    # generate_ackermann_factor()
    print("Done.")

if __name__ == "__main__":
    main()
```

## 6. 결론

Symforce를 Python-only 전략에 통합함으로써, C++로 전환하지 않고도 다음과 같은 이점을 얻을 수 있습니다.

1.  **빠른 개발**: 복잡한 수학적 모델을 Python으로 빠르게 정의하고 테스트합니다.
2.  **성능 확보**: 계산이 복잡한 부분은 Symforce가 생성한 최적화된 Python 코드로 대체하여 성능을 높입니다.
3.  **유지보수 용이성**: 전체 코드베이스가 Python으로 통일되어 관리가 용이합니다.

이를 통해 CC-SLAM-SYM은 대회 준비라는 목표에 맞춰 개발 속도와 성능 사이의 균형을 맞출 수 있습니다.
