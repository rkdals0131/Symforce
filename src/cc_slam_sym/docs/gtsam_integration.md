# GTSAM 통합 상세 설계

## 1. 개요

본 문서는 CC-SLAM-SYM에서 GTSAM(Georgia Tech Smoothing and Mapping) 라이브러리를 활용한 Factor Graph 기반 최적화의 상세 설계를 다룹니다.

## 2. Factor Graph 구조 설계

### 2.1 전체 그래프 구조

```
Variables (노드):
- X_i: 로봇 포즈 (SE(2)) at time i
- L_j: 랜드마크 위치 (R²) for landmark j  
- V_i: 로봇 속도 (R³) at time i (IMU 사용 시)
- B_i: IMU 바이어스 at time i

Factors (엣지):
- Prior factors: 초기 상태
- Odometry factors: 연속 포즈 간 제약
- Landmark factors: 포즈-랜드마크 관측
- IMU factors: IMU 사전적분
- GPS factors: 절대 위치 제약
- Loop closure factors: 루프 제약
```

### 2.2 Variable 명명 규칙

```cpp
// Symbol 정의
namespace sym {
    // 포즈: X0, X1, X2, ...
    gtsam::Symbol pose(int idx) { 
        return gtsam::Symbol('x', idx); 
    }
    
    // 랜드마크: L0, L1, L2, ...
    gtsam::Symbol landmark(int idx) { 
        return gtsam::Symbol('l', idx); 
    }
    
    // 속도: V0, V1, V2, ...
    gtsam::Symbol velocity(int idx) { 
        return gtsam::Symbol('v', idx); 
    }
    
    // IMU 바이어스: B0, B1, B2, ...
    gtsam::Symbol bias(int idx) { 
        return gtsam::Symbol('b', idx); 
    }
}
```

## 3. Factor 구현 상세

### 3.1 Prior Factor

초기 포즈 또는 고정된 랜드마크에 대한 절대적 제약입니다.

```cpp
// 초기 포즈 Prior
void addInitialPosePrior(gtsam::NonlinearFactorGraph& graph,
                        const gtsam::Pose2& initial_pose,
                        const Eigen::Vector3d& sigmas = {0.1, 0.1, 0.05}) {
    auto prior_noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(
        sym::pose(0), initial_pose, prior_noise
    ));
}

// 랜드마크 Prior (시작/종료 라인 등)
void addLandmarkPrior(gtsam::NonlinearFactorGraph& graph,
                     int landmark_id,
                     const gtsam::Point2& position,
                     double sigma = 0.05) {
    auto prior_noise = gtsam::noiseModel::Isotropic::Sigma(2, sigma);
    graph.add(gtsam::PriorFactor<gtsam::Point2>(
        sym::landmark(landmark_id), position, prior_noise
    ));
}
```

### 3.2 Odometry Factor

연속된 포즈 간의 상대적 움직임 제약입니다.

```cpp
class OdometryFactor {
public:
    static void add(gtsam::NonlinearFactorGraph& graph,
                   int from_idx, int to_idx,
                   const gtsam::Pose2& odometry,
                   const OdometryNoise& noise_params) {
        // 적응적 노이즈 모델 (이동 거리에 비례)
        double distance = odometry.translation().norm();
        double rotation = std::abs(odometry.theta());
        
        Eigen::Vector3d sigmas;
        sigmas << noise_params.sigma_x * (1.0 + noise_params.scale_x * distance),
                 noise_params.sigma_y * (1.0 + noise_params.scale_y * distance),
                 noise_params.sigma_theta * (1.0 + noise_params.scale_theta * rotation);
        
        auto noise = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        
        graph.add(gtsam::BetweenFactor<gtsam::Pose2>(
            sym::pose(from_idx), sym::pose(to_idx),
            odometry, noise
        ));
    }
};

struct OdometryNoise {
    double sigma_x = 0.1;       // 기본 x 노이즈
    double sigma_y = 0.05;      // 기본 y 노이즈  
    double sigma_theta = 0.05;  // 기본 회전 노이즈
    double scale_x = 0.01;      // 거리 비례 계수
    double scale_y = 0.005;
    double scale_theta = 0.01;
};
```

### 3.3 Landmark Observation Factor

로봇 포즈에서 랜드마크 관측에 대한 제약입니다.

```cpp
// 2D 랜드마크 관측 팩터
class LandmarkObservationFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Point2> {
private:
    gtsam::Point2 measured_;  // 로봇 좌표계 관측값
    
public:
    LandmarkObservationFactor(gtsam::Key pose_key, gtsam::Key landmark_key,
                             const gtsam::Point2& measured,
                             const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor2(model, pose_key, landmark_key), measured_(measured) {}
    
    gtsam::Vector evaluateError(const gtsam::Pose2& pose,
                               const gtsam::Point2& landmark,
                               boost::optional<gtsam::Matrix&> H1 = boost::none,
                               boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        // 예측된 관측값 계산
        gtsam::Matrix2 H_transform;
        gtsam::Point2 predicted = pose.transformTo(landmark, H_transform, H2);
        
        if (H1) {
            // 포즈에 대한 야코비안
            *H1 = (gtsam::Matrix23() << H_transform, 
                   -predicted.y(), predicted.x()).finished();
        }
        
        // 잔차 계산
        return predicted - measured_;
    }
    
    // 거리 기반 적응적 노이즈
    static gtsam::SharedNoiseModel createNoise(double distance, 
                                              double base_sigma = 0.05) {
        double sigma = base_sigma * (1.0 + 0.01 * distance);
        return gtsam::noiseModel::Isotropic::Sigma(2, sigma);
    }
};
```

### 3.4 IMU Factor

IMU 사전적분을 사용한 연속 상태 간 제약입니다.

```cpp
class ImuIntegration {
private:
    std::shared_ptr<gtsam::PreintegratedImuMeasurements> preintegrated_;
    gtsam::imuBias::ConstantBias current_bias_;
    
public:
    ImuIntegration(const ImuParameters& params) {
        auto p = gtsam::PreintegrationParams::MakeSharedU(params.gravity);
        
        // IMU 노이즈 설정
        p->accelerometerCovariance = gtsam::I_3x3 * pow(params.accel_noise_density, 2);
        p->gyroscopeCovariance = gtsam::I_3x3 * pow(params.gyro_noise_density, 2);
        p->integrationCovariance = gtsam::I_3x3 * 1e-8;
        
        // 바이어스 모델
        p->biasAccCovariance = gtsam::I_3x3 * pow(params.accel_bias_stability, 2);
        p->biasOmegaCovariance = gtsam::I_3x3 * pow(params.gyro_bias_stability, 2);
        
        preintegrated_ = std::make_shared<gtsam::PreintegratedImuMeasurements>(p, current_bias_);
    }
    
    void addMeasurement(const ImuData& imu, double dt) {
        preintegrated_->integrateMeasurement(
            imu.linear_acceleration,
            imu.angular_velocity,
            dt
        );
    }
    
    void addToGraph(gtsam::NonlinearFactorGraph& graph,
                   int from_idx, int to_idx) {
        // 2D SLAM을 위한 IMU 팩터 (z축 회전만 사용)
        graph.add(gtsam::ImuFactor(
            sym::pose(from_idx), sym::velocity(from_idx),
            sym::pose(to_idx), sym::velocity(to_idx),
            sym::bias(from_idx),
            *preintegrated_
        ));
        
        // 바이어스 변화 제약
        auto bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
        graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(
            sym::bias(from_idx), sym::bias(to_idx),
            gtsam::imuBias::ConstantBias(), bias_noise
        ));
    }
};
```

### 3.5 GPS Factor

RTK-GPS의 절대 위치 제약입니다.

```cpp
class GpsFactor : public gtsam::NoiseModelFactor1<gtsam::Pose2> {
private:
    gtsam::Point2 gps_position_;
    
public:
    GpsFactor(gtsam::Key pose_key, 
             const gtsam::Point2& gps_position,
             const gtsam::SharedNoiseModel& model)
        : NoiseModelFactor1(model, pose_key), gps_position_(gps_position) {}
    
    gtsam::Vector evaluateError(const gtsam::Pose2& pose,
                               boost::optional<gtsam::Matrix&> H = boost::none) const override {
        if (H) {
            *H = (gtsam::Matrix23() << 1, 0, 0, 
                                       0, 1, 0).finished();
        }
        return pose.translation() - gps_position_;
    }
    
    // RTK 상태에 따른 노이즈 모델
    static gtsam::SharedNoiseModel createNoise(const GpsData& gps) {
        Eigen::Vector2d sigmas;
        
        switch (gps.fix_type) {
            case GpsData::RTK_FIXED:
                sigmas << 0.02, 0.02;  // 2cm
                break;
            case GpsData::RTK_FLOAT:
                sigmas << 0.10, 0.10;  // 10cm
                break;
            default:
                sigmas << 1.0, 1.0;    // 1m (사용 안함)
        }
        
        return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }
};
```

## 4. ISAM2 증분 최적화

### 4.1 ISAM2 설정

```cpp
class SlamOptimizer {
private:
    gtsam::ISAM2 isam2_;
    gtsam::Values current_estimate_;
    
public:
    SlamOptimizer() {
        gtsam::ISAM2Params params;
        
        // 재선형화 설정
        params.relinearizeThreshold = 0.01;
        params.relinearizeSkip = 10;
        
        // 최적화 설정
        params.optimizationParams = gtsam::ISAM2DoglegParams();
        params.enableDetailedResults = true;
        
        // 팩터 제거 허용 (루프 클로저 재검증용)
        params.enablePartialRelinearizationCheck = true;
        
        isam2_ = gtsam::ISAM2(params);
    }
    
    void update(const gtsam::NonlinearFactorGraph& new_factors,
               const gtsam::Values& new_values,
               const std::vector<size_t>& remove_indices = {}) {
        // ISAM2 업데이트
        gtsam::ISAM2Result result = isam2_.update(
            new_factors, new_values,
            gtsam::FactorIndices(remove_indices.begin(), remove_indices.end())
        );
        
        // 통계 로깅
        if (result.errorBefore && result.errorAfter) {
            spdlog::debug("Optimization error: {:.6f} -> {:.6f}",
                         *result.errorBefore, *result.errorAfter);
        }
        
        // 현재 추정값 업데이트
        current_estimate_ = isam2_.calculateEstimate();
    }
};
```

### 4.2 키프레임 기반 업데이트 전략

```cpp
class KeyframeManager {
private:
    struct KeyframeData {
        int id;
        gtsam::Pose2 pose;
        std::vector<int> observed_landmarks;
        double timestamp;
    };
    
    std::deque<KeyframeData> active_keyframes_;
    const size_t max_active_keyframes_ = 20;
    
public:
    bool shouldAddKeyframe(const gtsam::Pose2& current_pose,
                          const gtsam::Pose2& last_keyframe_pose) {
        double trans_dist = (current_pose.translation() - 
                           last_keyframe_pose.translation()).norm();
        double rot_dist = std::abs(current_pose.theta() - 
                                  last_keyframe_pose.theta());
        
        return trans_dist > 1.0 || rot_dist > 0.2;  // 1m 또는 ~11도
    }
    
    void addKeyframe(const KeyframeData& kf,
                    gtsam::NonlinearFactorGraph& graph,
                    gtsam::Values& values) {
        active_keyframes_.push_back(kf);
        
        // 슬라이딩 윈도우 관리
        if (active_keyframes_.size() > max_active_keyframes_) {
            marginalizeOldest(graph, values);
        }
    }
    
private:
    void marginalizeOldest(gtsam::NonlinearFactorGraph& graph,
                          gtsam::Values& values) {
        // 가장 오래된 키프레임을 마지널라이즈
        // (구현 상세는 GTSAM marginalization 참고)
    }
};
```

## 5. 강건성을 위한 기법

### 5.1 Robust Kernel

아웃라이어에 강건한 최적화를 위한 Huber/Cauchy 커널 사용:

```cpp
// Huber robust kernel
auto huber = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Huber::Create(1.345),
    base_noise_model
);

// Cauchy robust kernel (더 강한 아웃라이어 제거)
auto cauchy = gtsam::noiseModel::Robust::Create(
    gtsam::noiseModel::mEstimator::Cauchy::Create(0.1),
    base_noise_model
);
```

### 5.2 Chi-square 테스트

팩터의 유효성 검증:

```cpp
bool validateFactor(const gtsam::NonlinearFactor& factor,
                   const gtsam::Values& values,
                   double chi2_threshold = 5.991) {  // 95% for 2-DOF
    double error = factor.error(values);
    return error < chi2_threshold;
}
```

## 6. 구현 예시

### 6.1 전체 파이프라인

```cpp
class GTSAMBackend {
private:
    SlamOptimizer optimizer_;
    KeyframeManager keyframe_manager_;
    gtsam::NonlinearFactorGraph new_factors_;
    gtsam::Values new_values_;
    
public:
    void processKeyframe(const Keyframe& kf,
                        const std::vector<DataAssociation::Match>& matches) {
        // 1. 오도메트리 팩터 추가
        if (kf.id > 0) {
            OdometryFactor::add(new_factors_, kf.id-1, kf.id,
                              kf.relative_odometry, OdometryNoise());
        }
        
        // 2. 랜드마크 관측 팩터 추가
        for (const auto& match : matches) {
            double distance = kf.observations[match.observation_idx]
                            .position.norm();
            
            auto noise = LandmarkObservationFactor::createNoise(distance);
            new_factors_.add(LandmarkObservationFactor(
                sym::pose(kf.id),
                sym::landmark(match.landmark_id),
                kf.observations[match.observation_idx].position,
                noise
            ));
        }
        
        // 3. GPS 팩터 추가 (있을 경우)
        if (kf.gps_data && kf.gps_data->isValid()) {
            auto noise = GpsFactor::createNoise(*kf.gps_data);
            new_factors_.add(GpsFactor(
                sym::pose(kf.id),
                kf.gps_data->toLocal(origin_lat_, origin_lon_),
                noise
            ));
        }
        
        // 4. 초기값 설정
        new_values_.insert(sym::pose(kf.id), kf.pose);
        
        // 5. ISAM2 업데이트
        if (keyframe_manager_.shouldAddKeyframe(kf.pose, last_kf_pose_)) {
            optimizer_.update(new_factors_, new_values_);
            new_factors_.resize(0);
            new_values_.clear();
        }
    }
};
```

## 7. 디버깅 및 시각화

### 7.1 Factor Graph 시각화

```cpp
void visualizeFactorGraph(const gtsam::NonlinearFactorGraph& graph,
                         const gtsam::Values& values) {
    // GraphViz dot 파일 생성
    graph.saveGraph("factor_graph.dot", values);
    
    // 각 팩터의 에러 출력
    for (size_t i = 0; i < graph.size(); ++i) {
        double error = graph[i]->error(values);
        spdlog::debug("Factor {}: error = {:.6f}", i, error);
    }
}
```

### 7.2 공분산 추출

```cpp
gtsam::Matrix extractPoseCovariance(const gtsam::ISAM2& isam2,
                                   gtsam::Key pose_key) {
    try {
        return isam2.marginalCovariance(pose_key);
    } catch (const gtsam::IndeterminantLinearSystemException& e) {
        spdlog::warn("Cannot compute marginal covariance");
        return gtsam::Matrix::Identity(3, 3) * 1e6;
    }
}
```

## 8. 성능 최적화 팁

1. **배치 업데이트**: 매 프레임마다 최적화하지 않고 키프레임 단위로
2. **변수 순서**: 시간 순서대로 변수 추가 (ISAM2 효율성)
3. **스파스성 활용**: 불필요한 팩터 연결 최소화
4. **병렬 처리**: 팩터 생성은 병렬화 가능

## 9. 주의사항

1. **좌표계 일관성**: 모든 데이터가 동일한 좌표계 사용 확인
2. **시간 동기화**: 센서 간 정확한 시간 동기화 필수
3. **초기값 품질**: GTSAM은 비선형 최적화이므로 좋은 초기값 중요
4. **수치 안정성**: 매우 작거나 큰 값 피하기