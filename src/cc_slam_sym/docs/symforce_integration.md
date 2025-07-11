# Symforce 통합 상세 설계

## 1. 개요

Symforce는 심볼릭 연산과 자동 미분을 통해 최적화된 코드를 생성하는 라이브러리입니다. CC-SLAM-SYM에서는 복잡한 커스텀 팩터의 야코비안을 자동으로 생성하고 최적화하는 데 활용합니다.

## 2. Symforce 활용 전략

### 2.1 왜 Symforce인가?

1. **자동 미분**: 복잡한 야코비안 수동 유도 불필요
2. **코드 최적화**: 심볼릭 단순화로 효율적인 C++ 코드 생성
3. **수치 안정성**: epsilon 처리 등 자동화
4. **개발 속도**: 새로운 제약 조건 빠른 프로토타이핑

### 2.2 GTSAM과의 통합 방식

```
Symforce (Python)          GTSAM (C++)
┌─────────────────┐       ┌──────────────────┐
│ Symbolic Factor │  -->  │ Generated C++    │
│ Definition      │       │ Code             │
└─────────────────┘       └──────────────────┘
         |                         |
         v                         v
   Code Generation            Custom Factor
```

## 3. 커스텀 팩터 정의

### 3.1 콘 색상 제약 팩터

콘의 색상 정보를 활용한 데이터 연관 강화 팩터입니다.

```python
# symforce_factors/cone_color_factor.py
import symforce
import symforce.symbolic as sf
from symforce import codegen
from symforce.codegen import codegen_util
from symforce.values import Values

@codegen.with_codegen_namespace("cc_slam_sym")
class ConeColorFactor:
    """
    콘 관측 시 색상 일치도를 고려한 팩터
    색상이 다르면 큰 페널티를 부과
    """
    
    @staticmethod
    def residual(
        robot_pose: sf.Pose2,
        landmark_pos: sf.V2,
        observation: sf.V2,
        observed_color: sf.Scalar,  # 0: yellow, 1: blue, 2: red
        landmark_color: sf.Scalar,
        color_weight: sf.Scalar,
        epsilon: sf.Scalar = sf.numeric_epsilon
    ) -> sf.V3:
        """
        Returns:
            [position_error_x, position_error_y, color_error]
        """
        # 예측된 관측 위치
        landmark_in_world = sf.V3(landmark_pos[0], landmark_pos[1], 1)
        T_world_robot = robot_pose.to_homogeneous_matrix()
        T_robot_world = T_world_robot.inv()
        
        landmark_in_robot = T_robot_world * landmark_in_world
        predicted = sf.V2(landmark_in_robot[0], landmark_in_robot[1])
        
        # 위치 잔차
        position_residual = predicted - observation
        
        # 색상 잔차 (색상이 다르면 큰 값)
        color_diff = sf.Abs(observed_color - landmark_color)
        color_residual = color_weight * sf.Min(color_diff, 1.0)
        
        return sf.V3(
            position_residual[0],
            position_residual[1], 
            color_residual
        )
    
    @staticmethod
    def generate_code():
        """C++ 코드 생성"""
        inputs = Values(
            robot_pose=sf.Pose2(),
            landmark_pos=sf.V2(),
            observation=sf.V2(),
            observed_color=sf.Scalar(),
            landmark_color=sf.Scalar(),
            color_weight=sf.Scalar(),
            epsilon=sf.Scalar()
        )
        
        outputs = Values(
            residual=sf.V3()
        )
        
        codegen_obj = codegen.Codegen(
            inputs=inputs,
            outputs=outputs,
            config=codegen.CppConfig(),
            name="cone_color_factor",
            return_key="residual",
            sparse_matrices=True
        )
        
        # C++ 코드 생성
        generated_dir = "generated/cone_color_factor"
        codegen_obj.generate_function(
            output_dir=generated_dir,
            skip_directory_nesting=True
        )
        
        return generated_dir
```

### 3.2 동적 모델 팩터

차량의 Ackermann 조향 모델을 고려한 모션 팩터입니다.

```python
# symforce_factors/ackermann_motion_factor.py
@codegen.with_codegen_namespace("cc_slam_sym")
class AckermannMotionFactor:
    """
    Ackermann 조향 기하를 고려한 차량 모션 모델
    """
    
    @staticmethod
    def residual(
        pose_i: sf.Pose2,
        pose_j: sf.Pose2,
        velocity: sf.Scalar,      # 전진 속도
        steering_angle: sf.Scalar, # 조향각
        dt: sf.Scalar,            # 시간 간격
        wheelbase: sf.Scalar,     # 축거
        epsilon: sf.Scalar = sf.numeric_epsilon
    ) -> sf.V3:
        """
        Ackermann 모델 기반 예측 포즈와 실제 포즈 간 잔차
        """
        # 조향각이 작을 때 수치 안정성
        turning_radius = wheelbase / (sf.tan(steering_angle) + epsilon)
        
        # 예측된 포즈 변화
        if sf.Abs(steering_angle) < epsilon:
            # 직진 운동
            dx = velocity * dt
            dy = 0
            dtheta = 0
        else:
            # 원호 운동
            dtheta = velocity * dt / turning_radius
            dx = turning_radius * sf.sin(dtheta)
            dy = turning_radius * (1 - sf.cos(dtheta))
        
        # 예측 포즈
        delta_pose = sf.Pose2(
            t=sf.V2(dx, dy),
            R=sf.Rot2.from_angle(dtheta)
        )
        predicted_pose = pose_i * delta_pose
        
        # 실제 포즈와의 차이
        pose_error = predicted_pose.inverse() * pose_j
        
        return sf.V3(
            pose_error.t[0],
            pose_error.t[1],
            pose_error.R.to_angle()
        )
```

### 3.3 IMU 바이어스 보정 팩터

IMU 바이어스의 시간에 따른 변화를 모델링하는 팩터입니다.

```python
# symforce_factors/imu_bias_factor.py
@codegen.with_codegen_namespace("cc_slam_sym")
class IMUBiasEvolutionFactor:
    """
    IMU 바이어스의 랜덤 워크 모델
    """
    
    @staticmethod
    def residual(
        bias_i: sf.V6,  # [accel_bias, gyro_bias]
        bias_j: sf.V6,
        dt: sf.Scalar,
        accel_bias_sigma: sf.V3,
        gyro_bias_sigma: sf.V3
    ) -> sf.V6:
        """
        바이어스 변화율 제약
        """
        # 예상 바이어스 변화 (랜덤 워크)
        expected_change = sf.V6.zero()
        
        # 실제 변화
        actual_change = bias_j - bias_i
        
        # 정규화된 잔차
        residual = sf.V6()
        
        # 가속도계 바이어스 (0-2)
        for i in range(3):
            residual[i] = actual_change[i] / (
                accel_bias_sigma[i] * sf.sqrt(dt)
            )
        
        # 자이로 바이어스 (3-5)
        for i in range(3):
            residual[i+3] = actual_change[i+3] / (
                gyro_bias_sigma[i] * sf.sqrt(dt)
            )
        
        return residual
```

### 3.4 랜드마크 기하 제약 팩터

트랙 형상에 대한 사전 지식을 활용한 팩터입니다.

```python
# symforce_factors/track_geometry_factor.py
@codegen.with_codegen_namespace("cc_slam_sym")
class TrackGeometryFactor:
    """
    Formula Student 트랙의 기하학적 제약
    - 파란색과 노란색 콘이 평행선 형성
    - 트랙 폭이 일정 범위 내
    """
    
    @staticmethod
    def residual(
        blue_cone_pos: sf.V2,
        yellow_cone_pos: sf.V2,
        nominal_track_width: sf.Scalar,
        width_tolerance: sf.Scalar,
        epsilon: sf.Scalar = sf.numeric_epsilon
    ) -> sf.V2:
        """
        Returns:
            [width_error, parallelism_error]
        """
        # 콘 간 거리
        cone_distance = (yellow_cone_pos - blue_cone_pos).norm()
        
        # 트랙 폭 오차
        width_error = (cone_distance - nominal_track_width) / width_tolerance
        
        # 평행성 오차 (간단히 거리 변화율로 근사)
        # 실제로는 이웃 콘들과의 관계도 고려해야 함
        parallelism_error = 0.0  # TODO: 구현 필요
        
        return sf.V2(width_error, parallelism_error)
```

## 4. GTSAM 통합

### 4.1 생성된 코드 래핑

Symforce로 생성된 C++ 코드를 GTSAM 팩터로 래핑합니다.

```cpp
// factors/symforce_cone_color_factor.h
#include <gtsam/nonlinear/NonlinearFactor.h>
#include "generated/cone_color_factor/cone_color_factor.h"

namespace cc_slam_sym {

class SymforceConeColorFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2> {
private:
    Eigen::Vector2d observation_;
    int observed_color_;
    int landmark_color_;
    double color_weight_;
    
public:
    SymforceConeColorFactor(
        gtsam::Key pose_key,
        gtsam::Key landmark_key,
        const Eigen::Vector2d& observation,
        int observed_color,
        int landmark_color,
        double color_weight,
        const gtsam::SharedNoiseModel& model
    ) : NoiseModelFactor2<gtsam::Pose2, gtsam::Point2>(model, pose_key, landmark_key),
        observation_(observation),
        observed_color_(observed_color),
        landmark_color_(landmark_color),
        color_weight_(color_weight) {}
    
    gtsam::Vector evaluateError(
        const gtsam::Pose2& pose,
        const gtsam::Point2& landmark,
        boost::optional<gtsam::Matrix&> H1 = boost::none,
        boost::optional<gtsam::Matrix&> H2 = boost::none
    ) const override {
        // Symforce 생성 함수 호출
        Eigen::Matrix<double, 2, 1> landmark_pos;
        landmark_pos << landmark.x(), landmark.y();
        
        Eigen::Matrix<double, 3, 1> residual;
        Eigen::Matrix<double, 3, 3> J_pose;
        Eigen::Matrix<double, 3, 2> J_landmark;
        
        // Symforce 생성 코드 호출
        cone_color_factor(
            pose.x(), pose.y(), pose.theta(),  // robot_pose
            landmark_pos.data(),                // landmark_pos
            observation_.data(),                // observation
            static_cast<double>(observed_color_),
            static_cast<double>(landmark_color_),
            color_weight_,
            1e-9,                              // epsilon
            residual.data(),                   // output
            H1 ? J_pose.data() : nullptr,
            H2 ? J_landmark.data() : nullptr
        );
        
        // 야코비안 설정
        if (H1) *H1 = J_pose;
        if (H2) *H2 = J_landmark;
        
        return residual;
    }
};

} // namespace cc_slam_sym
```

### 4.2 팩터 사용 예시

```cpp
// SLAM 백엔드에서 사용
void addConeObservation(
    gtsam::NonlinearFactorGraph& graph,
    int pose_id,
    int landmark_id,
    const ConeCluster& observation,
    const Landmark& landmark
) {
    // 색상이 다른 경우 높은 불확실성
    double color_weight = (observation.color == landmark.color) ? 0.1 : 10.0;
    
    // 노이즈 모델 (위치 2D + 색상 1D)
    gtsam::Vector3 sigmas;
    sigmas << 0.1, 0.1, 1.0;  // 색상 차원은 큰 시그마
    auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    
    // 팩터 추가
    graph.add(SymforceConeColorFactor(
        gtsam::Symbol('x', pose_id),
        gtsam::Symbol('l', landmark_id),
        observation.position.head<2>(),
        colorToInt(observation.color),
        colorToInt(landmark.color),
        color_weight,
        noise
    ));
}
```

## 5. 코드 생성 파이프라인

### 5.1 빌드 시스템 통합

```cmake
# CMakeLists.txt
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Symforce 코드 생성 커스텀 명령
add_custom_command(
    OUTPUT 
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/cone_color_factor/cone_color_factor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/generated/cone_color_factor/cone_color_factor.cpp
    COMMAND ${Python3_EXECUTABLE} 
        ${CMAKE_CURRENT_SOURCE_DIR}/symforce_factors/generate_all.py
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/symforce_factors/cone_color_factor.py
    COMMENT "Generating Symforce factors..."
)

# 생성된 소스를 타겟에 추가
add_library(symforce_factors
    generated/cone_color_factor/cone_color_factor.cpp
    # ... 다른 생성된 파일들
)
```

### 5.2 자동 생성 스크립트

```python
# symforce_factors/generate_all.py
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 각 팩터 모듈 임포트
from cone_color_factor import ConeColorFactor
from ackermann_motion_factor import AckermannMotionFactor
from imu_bias_factor import IMUBiasEvolutionFactor
from track_geometry_factor import TrackGeometryFactor

def generate_all_factors():
    """모든 Symforce 팩터 코드 생성"""
    
    base_dir = Path(__file__).parent.parent / "generated"
    base_dir.mkdir(exist_ok=True)
    
    factors = [
        ConeColorFactor,
        AckermannMotionFactor,
        IMUBiasEvolutionFactor,
        TrackGeometryFactor
    ]
    
    for factor_class in factors:
        print(f"Generating {factor_class.__name__}...")
        try:
            factor_class.generate_code()
            print(f"✓ {factor_class.__name__} generated successfully")
        except Exception as e:
            print(f"✗ Failed to generate {factor_class.__name__}: {e}")
            sys.exit(1)
    
    print("\nAll factors generated successfully!")

if __name__ == "__main__":
    generate_all_factors()
```

## 6. 성능 최적화

### 6.1 심볼릭 단순화

Symforce는 자동으로 심볼릭 표현을 단순화합니다:

```python
# 단순화 예시
# 원래: sin(theta) * cos(theta) * 2
# 단순화: sin(2*theta)

# 설정으로 제어 가능
config = codegen.CppConfig(
    use_eigen_types=True,
    zero_epsilon_behavior=codegen.ZeroEpsilonBehavior.PASSTHROUGH,
    normalize_results=True
)
```

### 6.2 수치 안정성

```python
@staticmethod
def safe_normalize(v: sf.V3, epsilon: sf.Scalar) -> sf.V3:
    """안전한 정규화"""
    norm = v.norm()
    return sf.where(
        norm > epsilon,
        v / norm,
        sf.V3.unit_x()  # 기본값
    )
```

## 7. 테스트 및 검증

### 7.1 단위 테스트

```python
# tests/test_symforce_factors.py
import numpy as np
from symforce.test_util import TestCase
from symforce import sf

class TestConeColorFactor(TestCase):
    def test_color_matching(self):
        """같은 색상일 때 색상 잔차가 0인지 확인"""
        robot_pose = sf.Pose2()
        landmark_pos = sf.V2(5.0, 0.0)
        observation = sf.V2(5.0, 0.0)
        
        residual = ConeColorFactor.residual(
            robot_pose, landmark_pos, observation,
            observed_color=0.0,  # yellow
            landmark_color=0.0,  # yellow
            color_weight=1.0
        )
        
        self.assertAlmostEqual(residual[2], 0.0)
    
    def test_jacobian_correctness(self):
        """자동 생성된 야코비안 검증"""
        # 수치 미분과 비교
        # ...
```

### 7.2 벤치마크

```cpp
// benchmarks/symforce_vs_manual.cpp
#include <benchmark/benchmark.h>

static void BM_SymforceJacobian(benchmark::State& state) {
    // Symforce 생성 코드 벤치마크
    for (auto _ : state) {
        // cone_color_factor() 호출
    }
}

static void BM_ManualJacobian(benchmark::State& state) {
    // 수동 구현 야코비안 벤치마크
    for (auto _ : state) {
        // 수동 야코비안 계산
    }
}

BENCHMARK(BM_SymforceJacobian);
BENCHMARK(BM_ManualJacobian);
```

## 8. 고급 활용

### 8.1 조건부 팩터

```python
@staticmethod
def conditional_residual(
    condition: sf.Scalar,
    true_residual: sf.V3,
    false_residual: sf.V3
) -> sf.V3:
    """조건에 따라 다른 잔차 반환"""
    return sf.where(
        condition > 0.5,
        true_residual,
        false_residual
    )
```

### 8.2 다중 센서 융합

```python
class MultiSensorFactor:
    """카메라 + 라이다 융합 팩터"""
    
    @staticmethod
    def residual(
        pose: sf.Pose2,
        landmark: sf.V2,
        camera_obs: sf.V2,
        lidar_obs: sf.V2,
        camera_weight: sf.Scalar,
        lidar_weight: sf.Scalar
    ) -> sf.V4:
        # 각 센서의 잔차 계산
        camera_residual = compute_camera_residual(pose, landmark, camera_obs)
        lidar_residual = compute_lidar_residual(pose, landmark, lidar_obs)
        
        # 가중 결합
        return sf.V4(
            camera_weight * camera_residual[0],
            camera_weight * camera_residual[1],
            lidar_weight * lidar_residual[0],
            lidar_weight * lidar_residual[1]
        )
```

## 9. 트러블슈팅

### 9.1 일반적인 문제

1. **컴파일 에러**: 생성된 코드의 include 경로 확인
2. **수치 불안정**: epsilon 값 조정
3. **성능 저하**: 심볼릭 표현 복잡도 확인

### 9.2 디버깅 팁

```python
# 중간 값 출력을 위한 디버그 모드
config = codegen.CppConfig(
    debug_mode=True,  # 중간 계산 출력
    print_generated_code=True  # 생성 코드 출력
)
```

## 10. 결론

Symforce를 통해:
1. 복잡한 야코비안 자동 생성
2. 수치적으로 안정적인 구현
3. 빠른 프로토타이핑 가능
4. GTSAM과의 원활한 통합

이를 통해 CC-SLAM-SYM은 더 정확하고 강건한 SLAM 시스템을 구축할 수 있습니다.