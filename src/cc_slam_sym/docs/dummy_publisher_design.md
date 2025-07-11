# 더미 퍼블리셔 설계 문서

## 1. 개요

더미 퍼블리셔는 SLAM 시스템 개발 및 테스트를 위한 시뮬레이션 환경을 제공합니다. 기존 cc_slam_graph 프로젝트에서 구현된 더미 퍼블리셔를 기반으로 하되, cc_slam_sym의 요구사항에 맞게 개선합니다.

**중요**: Ground Truth 콘 배치는 Formula Student Driverless 대회 환경을 정확히 모사하도록 신중하게 설계되었으므로, 절대 변경하지 않습니다.

## 2. 시나리오 정의

### 2.1 시나리오 1: 가속 및 브레이킹 테스트
- **목적**: 직선 구간에서의 SLAM 성능 및 AEB(자동 긴급 제동) 테스트
- **트랙 구성**:
  - 0-100m: 일반 구간 (파란색 왼쪽, 노란색 오른쪽)
  - 100-150m: AEB 구간 (빨간색 콘)
  - 150m: 빨간색 콘 벽
- **콘 간격**: 2.5m
- **차선 폭**: 5.0m

### 2.2 시나리오 2: 실제 대회 트랙 (중요)
- **목적**: 실제 Formula Student 트랙 환경 시뮬레이션
- **트랙 구성**: 
  - 복잡한 곡선 경로 (S자 커브 포함)
  - 내측: 파란색 콘 (반경 7m)
  - 외측: 노란색 콘 (반경 12m)
  - 다중 랩 지원 (기본 1바퀴, 최대 2바퀴)
- **시작 위치**: (30.0, 12.5) - 설정 가능

## 3. 핵심 기능

### 3.1 데이터 발행

#### 오도메트리 (/odom_sim)
- **주파수**: 100Hz
- **프레임**: odom → odom_sim
- **노이즈 모델**:
  - 선속도: σ_x = 0.02 m/s
  - 측면속도: σ_y = 0.01 m/s  
  - 각속도: σ_θ = 0.01 rad/s
- **메시지 타입**: nav_msgs/Odometry

#### 콘 관측 (/fused_sorted_cones_ukf_sim)
- **주파수**: ~19Hz
- **프레임**: odom_sim (로봇 로컬 좌표계)
- **관측 노이즈**:
  - 위치: σ_x = σ_y = 0.05m
- **시야각**: ±45도
- **최대 감지 거리**: 15m
- **메시지 타입**: custom_interface/TrackedConeArray

#### Ground Truth 맵 (/ground_truth_map_cones)
- **발행**: 초기 1회 (TRANSIENT_LOCAL QoS)
- **프레임**: map
- **메시지 타입**: visualization_msgs/MarkerArray

### 3.2 상태 기반 Track ID 관리

기존 구현의 핵심 로직 유지:
```python
# Track ID 할당 로직
- GT ID와 Track ID 분리 (Track ID는 1000번부터 시작)
- 시야에 들어올 때: 새 Track ID 할당
- 시야에서 벗어날 때: Track ID 해제
- 재진입 시: 새로운 Track ID 할당 (이전 ID 재사용 안함)
```

### 3.3 로봇 제어 (간소화)

#### 단순 중앙선 추종
- 트랙 중앙선을 따라 이동하는 단순한 제어
- 시나리오별 경로:
  - 시나리오 1: 직선 주행 → 정지
  - 시나리오 2: 파란색/노란색 콘의 중점 연결선 추종
- 제어 파라미터:
  - 일정 속도: 5 m/s (기본값)
  - 곡선 구간 감속: 3 m/s
  - 단순 P 제어기로 방향 조정

## 4. 개선사항 (cc_slam_sym용)

### 4.1 센서 퓨전 지원
```python
# 옵션 A를 위한 추가 토픽
/ouster/imu_sim (sensor_msgs/Imu) - 6축 IMU 시뮬레이션
/ublox_gps_node/fix_sim (sensor_msgs/NavSatFix) - RTK-GPS 시뮬레이션
/ublox_gps_node/fix_velocity_sim (geometry_msgs/TwistWithCovarianceStamped)
```

### 4.2 설정 구조 개선
```yaml
# config/dummy_publisher_config.yaml
scenario:
  id: 2
  num_laps_s2: 1
  s2_start_x: 30.0
  s2_start_y: 12.5

publish:
  odom: true
  cones: true
  gt_map_markers: true
  gt_tf: true
  odom_tf: true
  # 새로 추가
  imu: true
  gps: true

simulation:
  loop_rate: 100.0
  odom_rate: 100.0
  observation_rate: 19.0
  # 새로 추가
  imu_rate: 100.0
  gps_rate: 8.0

# 새로 추가: 센서 노이즈 설정
sensors:
  imu:
    accel_noise_density: 0.01  # m/s^2/√Hz
    gyro_noise_density: 0.001  # rad/s/√Hz
    accel_bias_stability: 0.1  # m/s^2
    gyro_bias_stability: 0.01  # rad/s
  gps:
    position_stddev: [0.014, 0.014, 0.015]  # m (RTK 정밀도)
    velocity_stddev: [0.085, 0.085, 0.085]  # m/s
```

### 4.3 코드 구조 개선

기존 파일 구조 유지하되 패키지 분리:
```
cc_slam_sym/
├── cc_slam_sym/
│   ├── __init__.py
│   ├── slam_node.py              # 메인 SLAM 노드
│   └── modules/                  # SLAM 모듈들
└── dummy_publisher/              # 독립된 서브패키지
    ├── __init__.py
    ├── cone_definitions.py       # 기존 콘 정의 (변경 없음!)
    ├── simple_path_follower.py   # 간소화된 경로 추종
    ├── ros_utils.py             # ROS 유틸리티
    ├── dummy_publisher_node.py   # 메인 노드 (개선)
    └── sensor_simulators.py      # 새로 추가: IMU/GPS 시뮬레이터
```

### 4.4 간소화된 경로 추종

```python
# simple_path_follower.py
class SimplePathFollower:
    """트랙 중앙선 추종을 위한 단순 제어기"""
    
    def __init__(self, scenario_id):
        self.scenario_id = scenario_id
        self.centerline_points = self._generate_centerline()
        self.current_segment = 0
        
    def _generate_centerline(self):
        """파란색/노란색 콘의 중점으로 중앙선 생성"""
        if self.scenario_id == 1:
            # 직선 경로
            return [(x, 0.0) for x in np.arange(0, 150, 5.0)]
        else:
            # 시나리오 2: 콘 정의에서 중앙선 계산
            # 각 구간별로 파란색/노란색 콘 쌍의 중점 계산
            pass
    
    def get_control_command(self, current_pose):
        """현재 위치에서 목표점까지의 단순 P 제어"""
        target_point = self._get_lookahead_point(current_pose)
        
        # 방향 오차 계산
        dx = target_point[0] - current_pose[0]
        dy = target_point[1] - current_pose[1]
        heading_error = np.arctan2(dy, dx) - current_pose[2]
        
        # 단순 P 제어
        angular_vel = 2.0 * np.sin(heading_error)  # P gain = 2.0
        linear_vel = 5.0 if abs(heading_error) < 0.3 else 3.0
        
        return linear_vel, angular_vel
```

### 4.5 IMU/GPS 시뮬레이터 추가

```python
# sensor_simulators.py
class IMUSimulator:
    """6축 IMU 시뮬레이션 (Ouster OS1 내장 IMU 모사)"""
    def generate_imu_data(self, linear_vel, angular_vel, dt):
        # 간단한 노이즈 모델
        accel = np.array([linear_vel/dt, 0, 0])  # 차체 좌표계
        accel += np.random.normal(0, 0.01, 3)    # 가속도 노이즈
        
        gyro = np.array([0, 0, angular_vel])
        gyro += np.random.normal(0, 0.001, 3)    # 자이로 노이즈
        
        return accel, gyro

class GPSSimulator:
    """RTK-GPS 시뮬레이션 (u-blox F9P 모사)"""
    def __init__(self, origin_lat=37.541383, origin_lon=127.077763):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
    def generate_gps_fix(self, x, y):
        # 간단한 미터 → GPS 좌표 변환
        lat = self.origin_lat + (y / 111111.0)
        lon = self.origin_lon + (x / (111111.0 * np.cos(np.radians(self.origin_lat))))
        
        # RTK 수준 노이즈 추가
        lat += np.random.normal(0, 0.00000014)  # ~1.5cm
        lon += np.random.normal(0, 0.00000014)
        
        return lat, lon
```

## 5. 인터페이스

### 5.1 ROS2 토픽 (출력)

| 토픽명 | 메시지 타입 | 주파수 | 설명 |
|--------|-------------|---------|------|
| /odom_sim | nav_msgs/Odometry | 100Hz | 노이즈가 추가된 오도메트리 |
| /fused_sorted_cones_ukf_sim | custom_interface/TrackedConeArray | ~19Hz | 관측된 콘 (로컬 좌표계) |
| /ground_truth_map_cones | visualization_msgs/MarkerArray | 1회 | Ground Truth 맵 |
| /ouster/imu_sim | sensor_msgs/Imu | 100Hz | 6축 IMU 데이터 |
| /ublox_gps_node/fix_sim | sensor_msgs/NavSatFix | 8Hz | RTK-GPS 위치 |
| /ublox_gps_node/fix_velocity_sim | geometry_msgs/TwistWithCovarianceStamped | 8Hz | GPS 속도 |

### 5.2 TF 변환

- `odom` → `odom_sim`: 오도메트리 프레임
- `map` → `base_link_gt_sim`: Ground Truth 시각화용

## 6. 사용 방법

```bash
# 기본 실행 (시나리오 2, 1바퀴)
ros2 run cc_slam_sym dummy_publisher_node

# 파라미터 오버라이드
ros2 run cc_slam_sym dummy_publisher_node --ros-args \
  -p scenario.id:=1 \
  -p scenario.num_laps_s2:=2 \
  -p sensors.gps.position_stddev:=[0.02,0.02,0.03]

# 설정 파일 사용
ros2 run cc_slam_sym dummy_publisher_node --ros-args \
  --params-file src/cc_slam_sym/config/dummy_publisher_config.yaml
```

## 7. 주의사항

1. **콘 배치 보존**: cone_definitions.py의 좌표는 절대 수정하지 않음
2. **Track ID 로직**: 기존 상태 기반 관리 방식 유지
3. **좌표계 일관성**: 모든 데이터는 정의된 좌표계 규칙 준수
4. **시간 동기화**: 모든 센서 데이터는 동일한 시계 기준 사용

## 8. 테스트 계획

1. **단위 테스트**
   - 각 시나리오별 콘 배치 검증
   - 노이즈 모델 통계적 검증
   - Track ID 할당/해제 로직 테스트

2. **통합 테스트**
   - SLAM 시스템과 연동 테스트
   - 다중 랩 시나리오 테스트
   - 센서 퓨전 데이터 일관성 검증

3. **성능 테스트**
   - CPU/메모리 사용량 모니터링
   - 발행 주파수 정확도 검증
   - 장시간 실행 안정성 테스트