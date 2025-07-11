# CC-SLAM-SYM 데이터 구조 상세 명세

## 1. 개요

본 문서는 CC-SLAM-SYM 프로젝트에서 사용되는 모든 데이터 구조의 상세 명세를 포함합니다. GTSAM과 Symforce를 활용한 Factor Graph SLAM 구현에 필요한 핵심 데이터 타입들을 정의합니다.

## 2. 기본 데이터 구조

### 2.1 ConeCluster

콘 감지 모듈에서 생성되는 원시 관측 데이터입니다.

```cpp
struct ConeCluster {
    // 기본 속성
    double timestamp;              // ROS 타임스탬프
    Eigen::Vector3d position;      // 로봇 기준 3D 위치 (x, y, z)
    std::string color;             // 콘 색상: "yellow", "blue", "red", "orange"
    double confidence;             // 감지 신뢰도 [0.0, 1.0]
    
    // 추가 속성
    int track_id;                  // 추적 ID (ukf 출력용)
    Eigen::Matrix3d covariance;    // 위치 불확실성 공분산
    
    // 메서드
    bool isValid() const;          // 유효성 검사
    double distanceTo(const ConeCluster& other) const;
};
```

### 2.2 Landmark

맵에 등록된 콘 랜드마크입니다.

```cpp
struct Landmark {
    // 식별자
    int id;                        // 고유 랜드마크 ID
    gtsam::Symbol symbol;          // GTSAM 심볼 (예: L0, L1, ...)
    
    // 속성
    Eigen::Vector2d position;      // 맵 기준 2D 위치 (x, y)
    std::string color;             // 콘 색상
    LandmarkType type;             // 랜드마크 타입
    
    // 통계 정보
    int observation_count;         // 총 관측 횟수
    double first_seen_timestamp;   // 최초 관측 시간
    double last_seen_timestamp;    // 최근 관측 시간
    double confidence;             // 랜드마크 신뢰도
    
    // 불확실성
    Eigen::Matrix2d covariance;    // 위치 공분산
    
    // 메서드
    void updateWithObservation(const ConeCluster& obs);
    bool shouldRemove(double current_time) const;
    gtsam::Point2 toGTSAM() const;
};

enum class LandmarkType {
    CONE_YELLOW,
    CONE_BLUE,
    CONE_RED,
    CONE_ORANGE,
    START_FINISH_LINE
};
```

### 2.3 Keyframe

SLAM 백엔드에서 사용하는 키프레임입니다.

```cpp
struct Keyframe {
    // 식별자
    int id;                              // 키프레임 ID
    double timestamp;                    // 타임스탬프
    gtsam::Symbol pose_symbol;           // GTSAM 포즈 심볼 (X0, X1, ...)
    
    // 상태
    gtsam::Pose2 pose;                   // SE(2) 포즈 (x, y, theta)
    Eigen::Vector3d velocity;            // 속도 (vx, vy, vtheta)
    
    // 센서 데이터
    std::vector<ConeCluster> observations;     // 콘 관측
    std::shared_ptr<ImuData> imu_data;         // IMU 데이터
    std::shared_ptr<GpsData> gps_data;         // GPS 데이터
    
    // 연결 정보
    std::vector<int> connected_keyframes;      // 연결된 키프레임 ID
    std::vector<int> observed_landmarks;       // 관측된 랜드마크 ID
    
    // 메서드
    bool shouldBeKeyframe(const Keyframe& last_kf) const;
    gtsam::Pose2 predictNextPose(double dt) const;
};
```

## 3. GTSAM 관련 데이터 구조

### 3.1 Factor Graph 구성 요소

```cpp
// Factor Graph 관리 클래스
class SlamFactorGraph {
private:
    gtsam::NonlinearFactorGraph graph;     // 팩터 그래프
    gtsam::Values initial_values;          // 초기값
    gtsam::Values optimized_values;        // 최적화된 값
    gtsam::ISAM2 isam2;                   // 증분 최적화기
    
public:
    // 팩터 추가 메서드
    void addPriorFactor(const gtsam::Symbol& key, const gtsam::Pose2& prior);
    void addOdometryFactor(const gtsam::Symbol& key1, const gtsam::Symbol& key2, 
                          const gtsam::Pose2& odometry);
    void addLandmarkFactor(const gtsam::Symbol& pose_key, const gtsam::Symbol& landmark_key,
                          const Eigen::Vector2d& observation);
    void addGPSFactor(const gtsam::Symbol& key, const GpsData& gps);
    void addIMUFactor(const gtsam::Symbol& key1, const gtsam::Symbol& key2,
                     const ImuPreintegration& preint);
    
    // 최적화
    void optimize();
    void updateISAM2(const gtsam::NonlinearFactorGraph& new_factors,
                    const gtsam::Values& new_values);
};
```

### 3.2 Noise Models

```cpp
namespace NoiseModels {
    // 오도메트리 노이즈 모델
    struct OdometryNoise {
        static gtsam::SharedNoiseModel create() {
            return gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector3() << 0.1, 0.1, 0.05).finished()  // x, y, theta
            );
        }
    };
    
    // 랜드마크 관측 노이즈
    struct LandmarkNoise {
        static gtsam::SharedNoiseModel create(double distance) {
            double sigma = 0.05 + 0.01 * distance;  // 거리에 비례
            return gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector2() << sigma, sigma).finished()
            );
        }
    };
    
    // GPS 노이즈 (RTK 정밀도)
    struct GPSNoise {
        static gtsam::SharedNoiseModel create() {
            return gtsam::noiseModel::Diagonal::Sigmas(
                (gtsam::Vector2() << 0.02, 0.02).finished()  // RTK: 2cm
            );
        }
    };
}
```

## 4. IMU 관련 데이터 구조

### 4.1 IMU 데이터

```cpp
struct ImuData {
    double timestamp;
    Eigen::Vector3d linear_acceleration;    // 가속도 (ax, ay, az)
    Eigen::Vector3d angular_velocity;       // 각속도 (wx, wy, wz)
    
    // 공분산
    Eigen::Matrix3d accel_covariance;
    Eigen::Matrix3d gyro_covariance;
};

// IMU 사전적분
class ImuPreintegration {
private:
    gtsam::PreintegratedImuMeasurements preintegrated;
    std::vector<ImuData> measurements;
    
public:
    void addMeasurement(const ImuData& imu);
    gtsam::ImuFactor createFactor(const gtsam::Symbol& pose_i,
                                 const gtsam::Symbol& vel_i,
                                 const gtsam::Symbol& pose_j,
                                 const gtsam::Symbol& vel_j,
                                 const gtsam::Symbol& bias) const;
};

// IMU 파라미터
struct ImuParameters {
    // 노이즈 밀도
    double accel_noise_density = 0.01;      // m/s^2/√Hz
    double gyro_noise_density = 0.001;      // rad/s/√Hz
    
    // 바이어스 안정성
    double accel_bias_stability = 0.1;      // m/s^2
    double gyro_bias_stability = 0.01;      // rad/s
    
    // 중력
    Eigen::Vector3d gravity = {0, 0, -9.81};
    
    // GTSAM 파라미터로 변환
    gtsam::PreintegrationParams toGTSAM() const;
};
```

## 5. GPS 관련 데이터 구조

### 5.1 GPS 데이터

```cpp
struct GpsData {
    double timestamp;
    
    // 위치 (위경도)
    double latitude;
    double longitude;
    double altitude;
    
    // UTM 좌표
    double utm_x;
    double utm_y;
    std::string utm_zone;
    
    // 속도
    Eigen::Vector3d velocity_enu;  // East-North-Up
    
    // 정밀도
    Eigen::Matrix3d position_covariance;
    Eigen::Matrix3d velocity_covariance;
    
    // 상태
    enum FixType {
        NO_FIX = 0,
        SINGLE = 1,
        DIFFERENTIAL = 2,
        RTK_FIXED = 4,
        RTK_FLOAT = 5
    } fix_type;
    
    // 메서드
    Eigen::Vector2d toLocal(double origin_lat, double origin_lon) const;
    bool isValid() const { return fix_type >= RTK_FLOAT; }
};
```

## 6. 데이터 연관 (Data Association)

### 6.1 연관 결과

```cpp
struct DataAssociation {
    struct Match {
        int landmark_id;
        int observation_idx;
        double distance;
        double mahalanobis_distance;
        double color_match_score;
    };
    
    std::vector<Match> matches;              // 매칭된 쌍
    std::vector<int> new_landmark_indices;   // 새 랜드마크가 될 관측
    std::vector<int> outlier_indices;        // 아웃라이어 관측
    
    // 통계
    double average_match_distance;
    int num_color_mismatches;
};

// 데이터 연관 설정
struct DataAssociationConfig {
    double max_distance = 2.0;               // 최대 매칭 거리 (m)
    double max_mahalanobis = 5.991;         // χ²(2, 0.95) = 5.991
    bool enforce_color_match = true;         // 색상 일치 강제
    double new_landmark_threshold = 3.0;     // 새 랜드마크 거리 임계값
};
```

## 7. 루프 클로저

### 7.1 루프 클로저 데이터

```cpp
struct LoopClosure {
    // 루프 정보
    int query_keyframe_id;
    int match_keyframe_id;
    double timestamp;
    
    // 변환
    gtsam::Pose2 relative_pose;             // 상대 포즈
    Eigen::Matrix3d covariance;             // 불확실성
    
    // 검증 정보
    double score;                           // 매칭 점수
    int num_matched_landmarks;              // 매칭된 랜드마크 수
    std::vector<std::pair<int, int>> landmark_matches;  // 랜드마크 쌍
    
    // 메서드
    bool isValid() const { return score > 0.8 && num_matched_landmarks >= 5; }
    gtsam::BetweenFactor<gtsam::Pose2> toFactor() const;
};

// 루프 감지기 설정
struct LoopDetectorConfig {
    double min_time_diff = 30.0;            // 최소 시간 차이 (초)
    double search_radius = 5.0;             // 검색 반경 (m)
    int min_common_landmarks = 5;           // 최소 공통 랜드마크 수
    double geometric_consistency_threshold = 0.5;  // 기하학적 일관성 임계값
};
```

## 8. Symforce 관련 데이터 구조

### 8.1 Symbolic 표현

```cpp
// Symforce를 위한 심볼릭 타입 정의
namespace sym {
    using Pose2 = sym::Pose2<double>;
    using Vector2 = sym::Vector2<double>;
    using Scalar = sym::Scalar<double>;
    
    // 커스텀 팩터를 위한 심볼릭 함수
    struct ConeLandmarkResidual {
        static sym::Vector2 compute(
            const sym::Pose2& robot_pose,
            const sym::Vector2& landmark_pos,
            const sym::Vector2& observation,
            const sym::Scalar& epsilon = 1e-9
        );
    };
}
```

## 9. 시스템 상태

### 9.1 SLAM 상태

```cpp
struct SlamState {
    // 현재 상태
    gtsam::Pose2 current_pose;
    Eigen::Vector3d current_velocity;
    double current_timestamp;
    
    // 맵 정보
    std::unordered_map<int, Landmark> landmarks;
    std::unordered_map<int, Keyframe> keyframes;
    
    // 그래프 정보
    int num_factors;
    int num_variables;
    double optimization_error;
    
    // 통계
    struct Statistics {
        int total_keyframes;
        int total_landmarks;
        int active_landmarks;
        int loop_closures;
        double mapping_time;
        double optimization_time;
    } stats;
    
    // 메서드
    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
    void reset();
};
```

## 10. 사용 예시

### 10.1 기본 사용 예시

```cpp
// 키프레임 생성
Keyframe kf;
kf.id = keyframe_counter++;
kf.timestamp = ros::Time::now().toSec();
kf.pose = current_pose;
kf.observations = cone_observations;

// 랜드마크 업데이트
for (const auto& obs : kf.observations) {
    Landmark& lm = landmarks[obs.track_id];
    lm.updateWithObservation(obs);
}

// Factor Graph에 추가
factor_graph.addOdometryFactor(
    gtsam::Symbol('x', kf.id - 1),
    gtsam::Symbol('x', kf.id),
    odometry_measurement
);

// 최적화
factor_graph.optimize();
```

## 11. 메모리 관리 지침

- 모든 대용량 데이터는 `std::shared_ptr`로 관리
- 순환 참조 방지를 위해 약한 포인터(`std::weak_ptr`) 사용
- 오래된 키프레임과 보이지 않는 랜드마크는 주기적으로 제거
- Factor Graph 크기 제한을 위한 마지널라이제이션 적용