# 프로젝트 요구사항 명세서 (PRD)

**프로젝트 이름:** cc_slam_sym

**버전:** 1.1

**최종 업데이트:** 2025년 1월 12일

## 1. 프로젝트 개요 및 목표

### 1.1 개요

본 프로젝트는 Formula Student Driverless 대회 참가를 위한 자율주행 차량의 SLAM(Simultaneous Localization and Mapping) 시스템 개발을 목표로 합니다. 특히, 포인트 클라우드 전체를 사용하는 대신, 콘 클러스터(위치 + 색상) 정보를 활용하여 효율적인 SLAM을 구현하고자 합니다. GTSAM과 Symforce를 사용하여 최적화된 그래프 기반 SLAM을 구축하며, GLIM의 모듈형 아키텍처 철학을 따르되, 플러그인 시스템을 제외하여 시스템 복잡도를 줄이고 유지보수성을 높입니다. 테스트 및 검증을 위해 더미 퍼블리셔를 포함하고, 실제 rosbag 데이터 처리도 지원합니다.

### 1.2 목표

- **정확하고 안정적인 SLAM:** 콘 클러스터 기반으로 실시간으로 정확하고 안정적인 차량의 위치 추정 및 맵 생성을 제공합니다.
- **최적화된 성능:** GTSAM과 Symforce를 활용하여 연산량을 최소화하고 실시간 성능을 확보합니다.
- **모듈화된 아키텍처:** GLIM의 아키텍처 철학을 따라 유지보수 및 확장성이 용이한 모듈형 구조를 구축합니다.
- **실용적인 시스템:** Formula Student Driverless 대회의 실제 환경에서 사용 가능한 실용적인 SLAM 시스템을 개발합니다.
- **테스트 용이성:** 더미 퍼블리셔를 통해 개발 및 테스트를 용이하게 하고, 실제 rosbag 데이터 처리도 지원합니다.

## 2. 핵심 기능 요구사항

### 2.1 SLAM 핵심 기능

- **콘 클러스터 기반 특징점 추출:** 입력된 센서 데이터로부터 콘의 위치 및 색상 정보를 추출합니다. (외부 모듈 연동)
- **오도메트리 처리:** IMU, 휠 엔코더 등으로부터 얻은 오도메트리 데이터를 처리하여 차량의 움직임을 추정합니다.
- **백엔드 최적화:** GTSAM과 Symforce를 사용하여 추출된 특징점과 오도메트리 정보를 기반으로 그래프 기반 SLAM을 수행하고, 맵 및 차량 위치를 최적화합니다.
- **루프 클로저 감지 및 처리:** 이전에 방문했던 지역을 재방문했을 때 루프 클로저를 감지하고, 이를 SLAM 백엔드에 반영하여 맵의 정확도를 향상시킵니다.
- **맵 시각화:** 생성된 맵을 시각화하여 사용자에게 제공합니다. (RViz2 연동)

### 2.2 시뮬레이션 모듈 (현재 구현 완료)

- **현실적 센서 시뮬레이션:** TrackedConeArray 메시지 기반 콘 데이터 생성
  - 시나리오 1: 직선 가속 및 AEB 테스트 (150m 트랙)
  - 시나리오 2: Formula Student 실제 트랙 환경 (타원형 트랙)
  - CALICO 스타일 track ID 관리 (ROI 진입 시 새 ID 할당)

- **고급 노이즈 모델링:**
  - IMU drift/bias: 시간 의존적 바이어스 누적
  - Odometry drift: 거리/회전 비례 누적 오차  
  - Temperature effects: 온도 변화에 따른 센서 바이어스 변동
  - Vibration effects: 고가속 시 노이즈 증가

- **Detection Error Simulation:**
  - False negative: 콘 미감지 (기본 4%)
  - False positive: 가짜 콘 생성 (기본 0.5%)
  - Color misclassification: 잘못된 색상 분류 (기본 5%)
  - Unknown color: 센서 융합 실패 (기본 3%)

- **Formula Student 규칙 준수:**
  - Orange 콘 완전 제거 (FS 규칙 준수)
  - Yellow/Blue/Red 콘만 지원
  - 실제 트랙 레이아웃 반영

## 3. 시스템 아키텍처

본 시스템은 GLIM의 모듈형 아키텍처 철학을 따르되, 플러그인 시스템은 제외하여 다음과 같은 구조를 가집니다. 상세한 아키텍처 다이어그램은 [architecture_diagram.md](docs/architecture_diagram.md)를 참조하세요.

### 3.1 센서 퓨전 옵션

시스템은 두 가지 센서 퓨전 방식을 지원합니다:

**옵션 A: 내부 센서 퓨전 (권장)**
- SLAM 시스템이 IMU와 RTK-GPS 데이터를 직접 받아 내부적으로 융합
- GTSAM의 IMU 사전적분과 GPS 팩터를 활용
- 더 타이트한 센서 통합과 정확한 불확실성 전파

**옵션 B: 외부 오도메트리 사용**
- robot_localization 등에서 생성된 EKF 퓨전 오도메트리 활용
- 모듈화된 시스템 구조로 개발 및 디버깅 용이
- 기존 시스템과의 통합 간편

### 3.2 현재 모듈 구성 (Python 기반)

**현재 구현 완료:**
- **simulation/**: 비침습적 테스트 애드온
  - dummy_publisher_node.py: 현실적 센서 시뮬레이션
  - cone_definitions.py: 트��� 시나리오 정의
- **slam_core/**: SLAM 알고리즘 구현체
  - cone_color_factor.py: SymForce 기반 색상 팩터
- **utils/**: 공통 데이터 구조
  - data_structures.py: SLAM 핵심 데이터 타입
- **ros_bridge/**: ROS2 인터페이스

**향후 구현 예정 (Python):**
- **센서 전처리 모듈:** IMU, GPS, 콘 데이터의 전처리 및 좌표계 변환
- **프론트엔드 모듈:** 키프레임 선정, 초기 자세 예측, 강건한 데이터 연관
- **백엔드 최적화 모듈:** GTSAM/Symforce 기반 그래프 최적화
- **맵 관리 모듈:** 랜드마크 데이터베이스 관리 및 맵 업데이트
- **시각화 모듈:** RViz2를 통한 실시간 시각화

### 3.3 비동기 처리 구조

GLIM과 유사하게 각 모듈은 비동기적으로 동작하여 실시간 성능을 보장합니다:

- 각 모듈은 별도의 스레드에서 실행
- 스레드 안전 큐를 통한 데이터 교환
- 백프레셔 제어를 통한 시스템 안정성 확보
- 센서별 처리 주기: IMU(100Hz), GPS(8Hz), 콘(~19Hz), SLAM(~18Hz)

## 4. 주요 모듈 상세 설명

### 4.1 전처리 모듈

**입력:** 콘 클러스터 데이터 (위치, 색상), 오도메트리 데이터

**기능:**
- 데이터 형식 변환 (ROS2 메시지 → 내부 데이터 구조)
- 노이즈 필터링
- 좌표계 변환 (센서 좌표계 → 로봇 좌표계)
- 타임스탬프 동기화

### 4.2 프론트엔드 추정 모듈

**입력:** 오도메트리 데이터 (IMU, 휠 엔코더)

**기능:**
- IMU 사전적분 (GTSAM PreintegratedIMU 사용)
- 휠 오도메트리 처리
- 초기 자세 예측
- 키프레임 선정 (이동 거리/회전각 기반)

### 4.3 데이터 연관 모듈

**입력:** 관측된 콘 클러스터, 현재 자세 추정치, 기존 맵

**기능:**
- 최근접 이웃 탐색 (KD-Tree 사용)
- 색상 기반 연관 검증
- 새로운 랜드마크 초기화
- 아웃라이어 제거

### 4.4 백엔드 최적화 모듈

**입력:** 키프레임, 랜드마크 관측, 루프 클로저

**기능:**
- Factor Graph 구성
  - Prior Factor: 초기 자세
  - Odometry Factor: 연속 자세 간 제약
  - Landmark Factor: 자세-랜드마크 간 관측
  - Loop Closure Factor: 루프 제약
- GTSAM ISAM2를 사용한 증분 최적화
- Symforce를 사용한 커스텀 팩터 정의
- 아웃라이어 검출 및 제거

### 4.5 맵 관리 모듈

**입력:** 최적화된 랜드마크 위치

**기능:**
- 랜드마크 데이터베이스 관리
- 중복 랜드마크 병합
- 오래된 랜드마크 제거
- 맵 저장/로드 기능

### 4.6 시각화 모듈

**입력:** 로봇 자세, 랜드마크 맵, 원시 관측

**기능:**
- RViz2 마커 퍼블리싱
- TF2 브로드캐스팅
- 실시간 궤적 표시
- 콘 색상별 시각화

## 5. 데이터 구조

핵심 데이터 구조의 요약입니다. 상세한 정의와 GTSAM 관련 구조는 [data_structures.md](docs/data_structures.md)를 참조하세요.

### 5.1 핵심 데이터 타입

- **ConeCluster**: 콘 감지 모듈의 원시 관측 데이터
- **Landmark**: 맵에 등록된 콘 랜드마크 
- **Keyframe**: SLAM 백엔드의 키프레임
- **SlamFactorGraph**: GTSAM Factor Graph 관리
- **ImuData/GpsData**: 센서 데이터 구조체

각 데이터 구조는 GTSAM과의 원활한 통합을 위해 변환 메서드를 포함합니다.

## 6. 기술 스택 (Python-Only 접근법)

### 6.1 전체 시스템 구현 (Python 기반)
- **프로그래밍 언어:** Python 3.10
- **ROS 버전:** ROS2 Humble
- **SLAM 라이브러리:** 
  - GTSAM Python wrapper (내부적으로 C++ 최적화)
  - SymForce (Python 기반 정의 및 코드 생성)
- **주요 라이브러리:** NumPy, SciPy, Matplotlib
- **메시지 인터페이스:** custom_interface.msg.TrackedConeArray

### 6.2 성능 최적화 전략
- **SymForce codegen**: Python으로 정의된 팩터로부터 최적화된 Python/C++ 코드 생성
- **GTSAM Python wrapper**: C++로 구현된 핵심 최적화 알고리즘 활용
- **프로파일링 기반**: `cProfile`, `line_profiler` 등으로 병목 지점 분석
- **Numba/Cython**: 필요시 순수 Python 함수 가속화

### 6.3 개발 우선순위 (Formula Student 최적화)
1. **빠른 프로토타이핑**: Python의 높은 생산성을 활용하여 핵심 기능 빠르게 구현
2. **실제 데이터 검증**: 조기 센서 통합 및 테��트를 통해 알고리즘 검증
3. **성능 프로파일링**: 실제 병목 지점을 정확히 파악
4. **선택적 최적화**: Symforce 코드 생성 및 Numba/Cython을 활용하여 필요한 부분만 성능 최적화

## 7. 성능 요구사항

- **실시간 성능:** 20Hz 이상의 업데이트 속도
- **위치 정확도:** 10cm 이내 (트랙 환경에서)
- **메모리 사용량:** 2GB 이내
- **CPU 사용량:** 4코어 기준 50% 이내
- **초기화 시간:** 5초 이내
- **최대 랜드마크 수:** 1000개

## 8. 개발 마일스톤 (Python 우선 접근법)

### ✅ Phase 0: 시뮬레이션 기반 구축 (완료)
- 패키지 구조 리팩터링 (simulation, slam_core, ros_bridge, utils)
- 현실적 센서 노이즈 시뮬레이션 (IMU drift/bias, odometry 누적 드리프트)
- Detection error simulation (false positive/negative, color errors)
- Formula Student 규칙 준수 (orange 콘 제거)
- SymForce cone color factor 기본 구현

### 🔄 Phase 1: Python SLAM 프레임워크 구축 (2-3주)
- GTSAM Python wrapper 설치 및 기본 사용법 학습
- 간단한 프론트엔드 (nearest neighbor data association) Python 구현
- 기본 백엔드 (GTSAM factor graph) Python 구현
- Python simulation과 Python SLAM 통합

### 📋 Phase 2: 프론트엔드 고도화 (3-4주)
- Robust data association (Mahalanobis distance, validation gates)
- Detection error 처리 (false positive/negative 대응)
- Color inconsistency 처리 및 unknown cone 대응
- Keyframe 선정 전략 구현

### 📋 Phase 3: 백엔드 최적화 (3-4주)
- SymForce cone color factor 통합 및 코드 생성
- IMU 사전적분 (GTSAM PreintegratedImuMeasurements)
- GTSAM ISAM2 증분 최적화
- Outlier rejection 및 robustness 개선

### 📋 Phase 4: 실제 센서 통합 & 프로파일링 (2-3주)
- ROS2 Python bridge를 통해 실제 센서 데이터 처리
- 실제 트랙 데이터 테스트
- **성능 프로파일링**: 병목 지점 정확한 측정
- 필요시 선택적 Python 최적화 (Numba, Cython 등)

### 📋 Phase 5: 고급 기능 및 대회 준비 (2-3주)
- Loop closure detection 구현
- 전체 시스템 성능 최적화 (Python + Symforce codegen)
- Formula Student 대회 준비

**현재 진행률:** Phase 0 완료, Phase 1 준비 중
**총 예상 잔여 개발 기간:** 약 12-17주 (Python 개발 속도 이점)

## 9. 테스트 계획

### 9.1 단위 테스트
- 각 모듈별 기능 테스트
- 데이터 구조 및 알고리즘 검증

### 9.2 통합 테스트
- 더미 퍼블리셔를 사용한 전체 시스템 테스트
- 다양한 시나리오 (직선, 곡선, 루프) 테스트

### 9.3 성능 테스트
- 실시간 성능 측정
- 메모리 사용량 프로파일링
- 정확도 평가 (ground truth 대비)

### 9.4 실제 데이터 테스트
- Formula Student 트랙 rosbag 데이터
- 다양한 날씨/조명 조건 테스트

## 10. 리스크 및 대응 방안

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| 실시간 성능 미달 | 높음 | 병렬 처리 최적화, 계산량 감소 |
| 루프 클로저 오검출 | 중간 | 강건한 검증 알고리즘 추가 |
| 센서 노이즈 | 중간 | 아웃라이어 제거 강화 |
| 메모리 누수 | 높음 | 정적 분석 도구 사용, 주기적 프로파일링 |

## 11. 향후 확장 계획 (Gemini 조언 반영)

### 11.1 단기 계획 (Formula Student 대회 준비) - Python 우선
- **빠른 프로토타이핑**: Python의 개발 속도 최대 활용
- **실제 센서 통합 최우선**: 최대한 빨리 실제 센서 데이터로 테스트
- **성능 프로파일링**: Python 병목 지점 정확한 측정 및 분석
- **선택적 최적화**: 실제 필요한 부분만 C++/Numba로 가속화
- **Real-world validation**: 시뮬레이션과 실제 환경 간 성능 차이 분석

### 11.2 중장기 확장 계획
- **GPU 가속 지원** (CUDA) - 실시간 성능 개선
- **다중 센서 융합** (Camera + LiDAR) - 정확도 향상
- **동적 물체 추적** - 레이싱 환경 대응
- **클라우드 기반 맵 공유** - 팀 간 데이터 공유
- **다중 로봇 SLAM** - 분산 SLAM 연구

### 11.3 연구 개발 방향
- **Machine Learning 통합**: 학습 기반 data association
- **Edge computing**: 임베디드 환경 최적화
- **Formula Student 커뮤니티 기여**: 오픈소스 SLAM 솔루션 제공

---

**참고:** 본 PRD는 개발 진행 상황에 따라 업데이트될 예정입니다.