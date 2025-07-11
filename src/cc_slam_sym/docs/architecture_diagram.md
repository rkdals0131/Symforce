# CC-SLAM-SYM 모듈 아키텍처 설계

## 1. 시스템 개요

CC-SLAM-SYM은 콘 클러스터 기반의 그래프 SLAM 시스템으로, GTSAM과 Symforce를 활용하여 최적화를 수행합니다. 시스템은 두 가지 센서 퓨전 옵션을 제공합니다.

## 2. 아키텍처 옵션

### 옵션 A: 내부 센서 퓨전 (권장)
SLAM 시스템이 직접 IMU와 RTK-GPS 데이터를 받아서 내부적으로 융합

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              외부 ROS2 토픽                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  /ouster/imu          /ublox_gps_node/fix      /fused_sorted_cones_ukf          │
│  (IMU 6축)            /ublox_gps_node/fix_velocity  (콘 클러스터)                  │
└────────┬──────────────────────┬─────────────────────────┬───────────────────────┘
         │                      │                         │
         ▼                      ▼                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  IMU 전처리      │    │  GPS 전처리        │    │  콘 전처리         │
│  모듈            │    │  모듈             │    │  모듈             │
├─────────────────┤    ├──────────────────┤    ├──────────────────┤
│ • 좌표계 변환      │   │ • UTM 변환         │    │ • 좌표계 변환      │
│ • 타임스탬프       │   │ • 공분산 처리       │    │ • 노이즈 필터링     │
│ • 바이어스 추정    │   │ • 상태 검증         │    │ • 색상 검증        │
└────────┬────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                      │                         │
         └──────────────────────┴─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    센서 퓨전 모듈                                  │
├─────────────────────────────────────────────────────────────────┤
│ • IMU 사전적분 (GTSAM PreintegratedImuMeasurements)               │
│ • GPS Factor 생성                                                │
│ • 시간 동기화                                                     │
│ • 센서 캘리브레이션                                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    프론트엔드 모듈                                 │
├─────────────────────────────────────────────────────────────────┤
│ • 키프레임 선정 (거리/회전 기반)                               │
│ • 초기 포즈 예측                                               │
│ • 콘 데이터 연관 (색상 + 거리 기반)                           │
│ • 새 랜드마크 초기화                                           │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    백엔드 최적화 모듈                           │
├─────────────────────────────────────────────────────────────────┤
│ • GTSAM Factor Graph 구성                                       │
│   - Prior Factor (초기 포즈)                                   │
│   - IMU Factor (IMU 사전적분)                                  │
│   - GPS Factor (절대 위치)                                     │
│   - Landmark Factor (콘 관측)                                  │
│   - Loop Closure Factor                                        │
│ • ISAM2 증분 최적화                                            │
│ • Symforce 커스텀 팩터                                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
        ┌──────────────────┐        ┌──────────────────┐
        │  맵 관리 모듈    │        │  시각화 모듈     │
        ├──────────────────┤        ├──────────────────┤
        │ • 랜드마크 DB    │        │ • RViz2 마커     │
        │ • 맵 저장/로드   │        │ • TF2 퍼블리싱   │
        │ • 중복 제거      │        │ • 경로 표시      │
        └──────────────────┘        └──────────────────┘
```

### 옵션 B: 외부 오도메트리 사용
robot_localization 등에서 생성된 퓨전 오도메트리 사용

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              외부 ROS2 토픽                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  /odometry/filtered    /fused_sorted_cones_ukf    /ouster/imu (선택적)          │
│  (EKF 퓨전 결과)       (콘 클러스터)              (추가 제약용)                  │
└────────┬───────────────────────┬──────────────────────┬─────────────────────────┘
         │                       │                      │
         ▼                       ▼                      ▼
┌─────────────────┐    ┌──────────────────┐   ┌──────────────────┐
│ 오도메트리 전처리│    │  콘 전처리       │   │  IMU 전처리      │
│  모듈           │    │  모듈            │   │  (선택적)        │
├─────────────────┤    ├──────────────────┤   ├──────────────────┤
│ • 공분산 검증   │    │ • 좌표계 변환    │   │ • 추가 제약 생성 │
│ • 이상치 검출   │    │ • 노이즈 필터링  │   │                  │
└────────┬────────┘    └────────┬─────────┘   └────────┬─────────┘
         │                      │                        │
         └──────────────────────┴────────────────────────┘
                                │
                                ▼
                    (이후 프론트엔드부터 동일)
```

## 3. 핵심 모듈 상세 설계

### 3.1 센서 퓨전 모듈 (옵션 A 전용)
```cpp
class SensorFusionModule {
public:
    // IMU 데이터 처리
    void processIMU(const sensor_msgs::msg::Imu& imu_msg);
    
    // GPS 데이터 처리
    void processGPS(const sensor_msgs::msg::NavSatFix& gps_fix,
                    const geometry_msgs::msg::TwistWithCovarianceStamped& gps_vel);
    
    // 퓨전된 상태 반환
    FusedState getFusedState();
    
private:
    gtsam::PreintegratedImuMeasurements::Params imu_params_;
    std::unique_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegrator_;
    UTMConverter utm_converter_;
};
```

### 3.2 프론트엔드 모듈
```cpp
class FrontendModule {
public:
    // 키프레임 처리
    void processKeyframe(const FusedState& state, 
                        const std::vector<ConeCluster>& cones);
    
    // 데이터 연관
    DataAssociationResult associateCones(
        const std::vector<ConeCluster>& observed_cones,
        const gtsam::Pose2& current_pose);
    
private:
    // 키프레임 선정 기준
    double keyframe_translation_threshold_ = 1.0;  // meters
    double keyframe_rotation_threshold_ = 0.2;     // radians
    
    // 데이터 연관 파라미터
    double association_distance_threshold_ = 2.0;  // meters
    std::unique_ptr<KDTree> landmark_kdtree_;
};
```

### 3.3 백엔드 최적화 모듈
```cpp
class BackendOptimizer {
public:
    // 팩터 추가
    void addIMUFactor(const gtsam::PreintegratedImuMeasurements& pim);
    void addGPSFactor(const GPSMeasurement& gps);
    void addLandmarkFactor(const LandmarkObservation& obs);
    void addLoopClosureFactor(const LoopConstraint& loop);
    
    // 최적화 수행
    OptimizationResult optimize();
    
private:
    gtsam::ISAM2 isam2_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values values_;
};
```

## 4. 데이터 흐름

### 4.1 타이밍 다이어그램
```
시간 →
IMU:     |--100Hz--|--100Hz--|--100Hz--|--100Hz--|
GPS:     |--------10Hz--------|--------10Hz--------|
Cones:   |------20Hz------|------20Hz------|
SLAM:    |----------20Hz----------|----------20Hz----------|
```

### 4.2 스레드 구조
```
Main Thread
    ├── IMU Thread (100Hz)
    ├── GPS Thread (10Hz) 
    ├── Cone Thread (20Hz)
    ├── Frontend Thread (20Hz)
    ├── Backend Thread (5-10Hz)
    └── Visualization Thread (10Hz)
```

## 5. 인터페이스 정의

### 5.1 ROS2 토픽 인터페이스

**입력 토픽 (옵션 A):**
- `/ouster/imu` (sensor_msgs/Imu): 6축 IMU 데이터
- `/ublox_gps_node/fix` (sensor_msgs/NavSatFix): GPS 위치
- `/ublox_gps_node/fix_velocity` (geometry_msgs/TwistWithCovarianceStamped): GPS 속도
- `/fused_sorted_cones_ukf` (custom_interface/TrackedConeArray): 추적된 콘

**입력 토픽 (옵션 B):**
- `/odometry/filtered` (nav_msgs/Odometry): EKF 퓨전 오도메트리
- `/fused_sorted_cones_ukf` (custom_interface/TrackedConeArray): 추적된 콘

**출력 토픽:**
- `/slam/pose` (geometry_msgs/PoseStamped): 최적화된 로봇 포즈
- `/slam/path` (nav_msgs/Path): 로봇 경로
- `/slam/landmarks` (visualization_msgs/MarkerArray): 랜드마크 맵
- `/tf` (tf2_msgs/TFMessage): 좌표계 변환

### 5.2 서비스 인터페이스
- `/slam/save_map` (std_srvs/Trigger): 맵 저장
- `/slam/load_map` (std_srvs/Trigger): 맵 로드
- `/slam/reset` (std_srvs/Trigger): SLAM 초기화

## 6. 성능 고려사항

### 6.1 병렬 처리
- 센서 데이터 수신: 별도 스레드
- 전처리: 병렬 파이프라인
- 백엔드 최적화: 별도 스레드에서 주기적 실행

### 6.2 메모리 관리
- 슬라이딩 윈도우: 최근 N개 키프레임만 활성 유지
- 랜드마크 프루닝: 오래된/보이지 않는 랜드마크 제거
- 공유 포인터 사용: 데이터 복사 최소화

## 7. 권장사항

**옵션 A (내부 센서 퓨전) 장점:**
- SLAM에 최적화된 타이트한 센서 통합
- GPS 아웃라이어를 SLAM 레벨에서 처리
- 더 정확한 불확실성 전파

**옵션 B (외부 오도메트리) 장점:**
- 모듈화된 시스템 구조
- 기존 robot_localization 활용
- 개발 및 디버깅 용이

**추천:** 옵션 A로 시작하되, 모듈을 잘 분리해서 나중에 옵션 B로도 전환 가능하도록 설계

## 8. 더미 퍼블리셔 통합

### 8.1 시뮬레이션 환경
```
[더미 퍼블리셔 노드]
    ├── 시나리오 1: 직선 + AEB 테스트
    └── 시나리오 2: Formula Student 트랙 (2바퀴)
         ├── /odom_sim (100Hz)
         ├── /fused_sorted_cones_ukf_sim (~19Hz)
         ├── /ouster/imu_sim (100Hz) 
         ├── /ublox_gps_node/fix_sim (8Hz)
         └── /ublox_gps_node/fix_velocity_sim (8Hz)
                    ↓
              [CC-SLAM-SYM]
```

### 8.2 테스트 모드 전환
```yaml
# 더미 퍼블리셔 사용 시
use_simulation: true
input_topics:
  imu: "/ouster/imu_sim"
  gps_fix: "/ublox_gps_node/fix_sim"
  gps_vel: "/ublox_gps_node/fix_velocity_sim"
  cones: "/fused_sorted_cones_ukf_sim"
  
# 실제 센서 사용 시
use_simulation: false
input_topics:
  imu: "/ouster/imu"
  gps_fix: "/ublox_gps_node/fix"
  gps_vel: "/ublox_gps_node/fix_velocity"
  cones: "/fused_sorted_cones_ukf"
```

## 9. 다음 단계
1. 데이터 구조 상세 정의
2. 각 모듈의 구체적인 인터페이스 정의
3. GTSAM 팩터 구성 설계
4. Symforce 커스텀 팩터 설계