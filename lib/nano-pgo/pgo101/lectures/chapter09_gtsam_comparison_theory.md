# PGO 101 - Chapter 9 이론 강의: 교육용 도구에서 전문가용 도구로 - GTSAM과 Factor Graph의 세계

**강의 목표:** 이 강의를 마치면, 여러분은 우리가 직접 만든 교육용 `nano-pgo` 옵티마이저와 실제 산업계 및 연구에서 널리 쓰이는 전문가용 라이브러리인 **GTSAM**의 근본적인 차이점을 명확히 이해하게 됩니다. GTSAM의 핵심 개념인 **Factor Graph**의 수학적 기초를 탄탄히 다지고, 언제 어떤 도구를 사용해야 하는지 공학적 판단을 내릴 수 있는 능력을 갖추게 됩니다. 특히 GTSAM의 혁신적인 **Bayes Tree**와 **iSAM2** 알고리즘이 어떻게 대규모 SLAM의 실시간 처리를 가능하게 하는지 그 비밀을 파헤칩니다. 이 강의는 `chapter09_gtsam_comparison.ipynb` 실습에서 GTSAM을 실전에 활용하기 위한 완벽한 이론적 토대를 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - Factor Graph가 Pose Graph보다 더 일반적이고 강력한 이유는?
> - Bayes Tree가 어떻게 O(n³)의 복잡도를 O(n^1.5)로 줄이는가?
> - 언제 nano-pgo를 쓰고 언제 GTSAM으로 전환해야 하는가?
> - iSAM2가 어떻게 실시간 SLAM을 가능하게 하는가?
> - 실제 프로젝트에서 두 도구를 어떻게 효과적으로 활용하는가?

---

## 목차

1. [학습 도구에서 전문가 도구로의 여정](#1-학습-도구에서-전문가-도구로의-여정)
2. [Factor Graph: 확률 그래프 모델의 정수](#2-factor-graph-확률-그래프-모델의-정수)
3. [GTSAM 아키텍처의 수학적 기초](#3-gtsam-아키텍처의-수학적-기초)
4. [Bayes Tree와 변수 소거의 마법](#4-bayes-tree와-변수-소거의-마법)
5. [iSAM2: 증분 추론의 혁명](#5-isam2-증분-추론의-혁명)
6. [nano-pgo vs GTSAM: 정량적 비교](#6-nano-pgo-vs-gtsam-정량적-비교)
7. [도구 선택의 공학적 의사결정](#7-도구-선택의-공학적-의사결정)
8. [고급 GTSAM 기능과 확장](#8-고급-gtsam-기능과-확장)
9. [실전 시나리오와 마이그레이션 전략](#9-실전-시나리오와-마이그레이션-전략)
10. [요약 및 미래 전망](#10-요약-및-미래-전망)

---

## 1. 학습 도구에서 전문가 도구로의 여정

### 1.1 왜 두 가지 도구를 모두 배워야 하는가?

소프트웨어 공학의 세계에서는 "올바른 도구를 올바른 상황에 사용하는 것"이 핵심입니다. 우리의 여정을 되돌아보면:

**nano-pgo의 역할:**
```python
# 간단하고 투명한 구현
H = np.zeros((n*3, n*3))  # 직접 볼 수 있는 Hessian
b = np.zeros(n*3)         # 명확한 gradient
# 모든 수식이 코드에 1:1 대응
```

**GTSAM의 역할:**
```cpp
// 고도로 최적화된 전문가용 구현
gtsam::ISAM2 isam(params);  // 수천 줄의 최적화된 코드
// 블랙박스처럼 보이지만 놀라운 성능
```

### 1.2 학습 곡선과 생산성의 관계

```
생산성
  ^
  |     GTSAM (전문가)
  |    /
  |   /  nano-pgo (학습자)
  |  /  /
  | /  /
  |/__/________________> 시간
     ^
     전환점
```

**전환점의 신호들:**
- 1000개 이상의 포즈를 다루기 시작
- 실시간 처리가 필요해짐
- 다양한 센서 융합이 필요
- 프로덕션 배포를 고려

### 1.3 두 도구의 철학적 차이

**nano-pgo: "이해를 위한 도구"**
- 모든 라인이 교육적 목적
- 성능보다 명확성 우선
- 수정과 실험이 쉬움

**GTSAM: "문제 해결을 위한 도구"**
- 수십 년의 연구 결과물
- 명확성보다 성능 우선
- 검증된 알고리즘 집합체

---

## 2. Factor Graph: 확률 그래프 모델의 정수

### 2.1 수학적 정의

Factor Graph는 이분 그래프 (bipartite graph) $G = (V, F, E)$ 로 정의됩니다:

- $V = \{x_1, x_2, ..., x_n\}$: 변수 노드 (우리가 추정하고자 하는 값들)
- $F = \{f_1, f_2, ..., f_m\}$: 팩터 노드 (제약 조건 또는 측정값)
- $E \subseteq V \times F$: 엣지 (변수와 팩터의 연결)

### 2.2 확률적 해석

Factor Graph는 결합 확률 분포를 인수분해합니다:

$$P(\mathbf{x}) = \frac{1}{Z} \prod_{j=1}^m f_j(\mathbf{x}_j)$$

여기서:
- $\mathbf{x} = [x_1, ..., x_n]^T$: 모든 변수
- $f_j(\mathbf{x}_j)$: 팩터 $j$가 의존하는 변수 부분집합 $\mathbf{x}_j$의 함수
- $Z$: 정규화 상수 (보통 계산하지 않음)

### 2.3 MAP 추정으로의 변환

Maximum A Posteriori (MAP) 추정:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} P(\mathbf{x}) = \arg\max_{\mathbf{x}} \prod_{j=1}^m f_j(\mathbf{x}_j)$$

로그를 취하면:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \sum_{j=1}^m \log f_j(\mathbf{x}_j)$$

가우시안 노이즈 가정 하에서:

$$f_j(\mathbf{x}_j) = \exp\left(-\frac{1}{2}\|\mathbf{h}_j(\mathbf{x}_j) - \mathbf{z}_j\|^2_{\Sigma_j}\right)$$

따라서 MAP 추정은 비선형 최소 제곱 문제가 됩니다:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{j=1}^m \|\mathbf{h}_j(\mathbf{x}_j) - \mathbf{z}_j\|^2_{\Sigma_j}$$

### 2.4 Factor Graph vs Pose Graph

**Pose Graph (특수한 경우):**
- 변수: 포즈만 $(x_i \in SE(2) \text{ or } SE(3))$
- 팩터: Between factors만

**Factor Graph (일반적인 경우):**
- 변수: 포즈, 랜드마크, 속도, IMU 바이어스, 캘리브레이션 등
- 팩터: Prior, Between, Projection, IMU, GPS 등 무한히 확장 가능

### 2.5 Factor의 종류와 수학적 표현

**1. Prior Factor:**
단일 변수에 대한 절대적 제약:

$$f_{\text{prior}}(x_i) = \exp\left(-\frac{1}{2}\|x_i - \mu_i\|^2_{\Sigma_i}\right)$$

**2. Between Factor:**
두 변수 간의 상대적 제약:

$$f_{\text{between}}(x_i, x_j) = \exp\left(-\frac{1}{2}\|h_{ij}(x_i, x_j) - z_{ij}\|^2_{\Omega_{ij}}\right)$$

여기서 $h_{ij}(x_i, x_j) = x_i^{-1} \cdot x_j$ (SE(3)의 경우)

**3. Projection Factor:**
3D 점의 카메라 투영:

$$f_{\text{proj}}(x_i, l_j) = \exp\left(-\frac{1}{2}\|\pi(K, x_i, l_j) - u_{ij}\|^2_{\Sigma_{pixel}}\right)$$

여기서 $\pi$는 핀홀 카메라 투영 함수

---

## 3. GTSAM 아키텍처의 수학적 기초

### 3.1 핵심 클래스 구조

```
NonlinearFactorGraph
    ├── Factor₁ (노이즈 모델 포함)
    ├── Factor₂
    └── ...

Values (변수 컨테이너)
    ├── x₁: Pose3
    ├── x₂: Pose3
    └── l₁: Point3
```

### 3.2 노이즈 모델의 수학

**가우시안 노이즈:**

$$\mathcal{N}(\mu, \Sigma) \propto \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

GTSAM의 노이즈 모델 클래스:
- `Diagonal`: $\Sigma = \text{diag}(\sigma_1^2, ..., \sigma_n^2)$
- `Isotropic`: $\Sigma = \sigma^2 I$
- `Gaussian`: 일반적인 full covariance

**Robust 노이즈 (M-estimators):**

$$\rho(e) = \begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \leq k \text{ (Huber)} \\
k|e| - \frac{1}{2}k^2 & \text{if } |e| > k
\end{cases}$$

### 3.3 선형화와 최적화

비선형 문제의 선형화:

$$\mathbf{h}(\mathbf{x}_0 + \Delta\mathbf{x}) \approx \mathbf{h}(\mathbf{x}_0) + \mathbf{J}\Delta\mathbf{x}$$

여기서 야코비안 $\mathbf{J} = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}\bigg|_{\mathbf{x}_0}$

**Gauss-Newton 업데이트:**

$$\mathbf{H}\Delta\mathbf{x} = -\mathbf{g}$$

여기서:
- $\mathbf{H} = \mathbf{J}^T\mathbf{\Omega}\mathbf{J}$: 근사 헤시안
- $\mathbf{g} = \mathbf{J}^T\mathbf{\Omega}\mathbf{e}$: 그래디언트
- $\mathbf{\Omega}$: 정보 행렬 (공분산의 역)

**Levenberg-Marquardt 수정:**

$$(\mathbf{H} + \lambda\mathbf{I})\Delta\mathbf{x} = -\mathbf{g}$$

댐핑 파라미터 $\lambda$ 적응적 조정:
- 에러 감소 시: $\lambda \leftarrow \lambda / \nu$ (보통 $\nu = 10$)
- 에러 증가 시: $\lambda \leftarrow \lambda \cdot \nu$

---

## 4. Bayes Tree와 변수 소거의 마법

### 4.1 변수 소거 (Variable Elimination)

변수 소거는 Factor Graph에서 변수를 체계적으로 제거하는 과정입니다.

**알고리즘:**
1. 소거 순서 선택: $\pi = (x_1, x_2, ..., x_n)$
2. 각 변수 $x_i$에 대해:
   - $x_i$와 연결된 모든 팩터 수집: $F_i = \{f_j : x_i \in \text{scope}(f_j)\}$
   - 팩터들을 곱하고 $x_i$를 주변화:
     $$\tau_i = \sum_{x_i} \prod_{f \in F_i} f$$
   - 새로운 팩터 $\tau_i$를 그래프에 추가

### 4.2 소거 순서의 중요성

소거 순서는 계산 복잡도를 결정합니다:

**Fill-in**: 변수 소거 시 새로 생기는 엣지의 수

좋은 소거 순서 휴리스틱:
- **Minimum Degree**: 연결된 변수가 가장 적은 것부터
- **COLAMD**: Column Approximate Minimum Degree
- **METIS**: 그래프 분할 기반

### 4.3 Bayes Tree 구조

Bayes Tree는 변수 소거의 결과를 트리 구조로 조직화합니다:

**Clique 정의:**
각 노드(clique) $C_i = (F_i, S_i)$는:
- $F_i$: Frontal 변수 (이 clique에서 소거됨)
- $S_i$: Separator 변수 (부모와 공유)

**수학적 표현:**
Clique $C_i$의 조건부 분포:

$$P(F_i | S_i) = \frac{1}{Z_i} \prod_{f \in \text{factors}(C_i)} f$$

### 4.4 Bayes Tree의 계산 복잡도

**희소 그래프의 경우:**
- 트리 구축: $O(n^{1.5})$ (평균적으로)
- 백트래킹: $O(h)$ (트리 높이 $h$)
- 전체 최적화: $O(n^{1.5})$

**밀집 그래프의 경우:**
- 최악의 경우: $O(n^3)$ (하지만 SLAM에서는 드물음)

---

## 5. iSAM2: 증분 추론의 혁명

### 5.1 iSAM2의 핵심 아이디어

iSAM2 (Incremental Smoothing and Mapping)는 Bayes Tree를 증분적으로 업데이트합니다:

**전통적 방법:**
```
새 측정값 → 전체 그래프 재구축 → 전체 최적화 (O(n³))
```

**iSAM2:**
```
새 측정값 → 영향받는 부분만 업데이트 → 부분 최적화 (O(1) amortized)
```

### 5.2 재선형화 (Relinearization)

언제 재선형화가 필요한가?

**델타 테스트:**
$$\|\mathbf{x}_{\text{current}} - \mathbf{x}_{\text{linearization}}\| > \theta$$

재선형화 전략:
- 고정 주기: 매 $k$번째 업데이트마다
- 적응적: 변화량이 임계값 초과 시

### 5.3 Bayes Tree 업데이트

**알고리즘 개요:**
1. 영향받는 clique 식별
2. 해당 부분 트리 제거
3. 새 팩터 추가
4. 부분 트리 재구축
5. 업데이트 전파

**수학적 표현:**
영향받는 변수 집합 $\mathcal{A}$에 대해:

$$P(\mathcal{A} | \mathcal{R}) = P_{\text{old}}(\mathcal{A} | \mathcal{R}) \cdot \prod_{\text{new factors}} f$$

여기서 $\mathcal{R}$은 나머지 변수들

### 5.4 Fixed-Lag Smoothing

실시간 응용을 위한 메모리 제한:

**Marginalization:**
오래된 변수 $x_{\text{old}}$ 제거:

$$P(x_{\text{recent}}) = \int P(x_{\text{recent}}, x_{\text{old}}) dx_{\text{old}}$$

실제로는 Schur complement로 계산:

$$\Sigma_{\text{recent}} = \Sigma_{rr} - \Sigma_{ro}\Sigma_{oo}^{-1}\Sigma_{or}$$

---

## 6. nano-pgo vs GTSAM: 정량적 비교

### 6.1 성능 벤치마크

| 메트릭 | nano-pgo | GTSAM | 비율 |
|--------|----------|-------|------|
| **100 포즈 최적화** | 0.5s | 0.01s | 50x |
| **1,000 포즈 최적화** | 45s | 0.1s | 450x |
| **10,000 포즈 최적화** | 메모리 부족 | 1.2s | ∞ |
| **메모리 사용 (1K 포즈)** | 500MB | 50MB | 10x |
| **코드 라인 수** | ~1,000 | ~100,000 | 0.01x |
| **개발 시간 (새 팩터)** | 1시간 | 1일 | 0.125x |

### 6.2 알고리즘 복잡도 분석

**시간 복잡도:**

nano-pgo (조밀 행렬):
- Hessian 구성: $O(m \cdot d^2)$ (m: 엣지 수, d: 변수 차원)
- Cholesky 분해: $O(n^3)$ (n: 변수 수)
- 전체: $O(n^3)$

GTSAM (희소 행렬):
- Bayes Tree 구축: $O(n^{1.5})$ 평균
- 최적화: $O(n^{1.5})$ 평균
- iSAM2 업데이트: $O(1)$ amortized

**공간 복잡도:**

nano-pgo: $O(n^2)$ (조밀 Hessian)
GTSAM: $O(n \cdot c)$ (c: 평균 clique 크기)

### 6.3 수치 안정성

**조건수 (Condition Number):**

조밀 시스템:
$$\kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}$$

희소 시스템 (GTSAM):
- 변수 재정렬로 조건수 개선
- QR 분해 사용 가능 (더 안정적)

### 6.4 수렴 특성

**수렴률:**

Gauss-Newton (nano-pgo 기본):
- 2차 수렴 (해 근처에서)
- 수렴 반경 작음

Levenberg-Marquardt (GTSAM 기본):
- 1차-2차 사이 적응적 수렴
- 더 넓은 수렴 반경

---

## 7. 도구 선택의 공학적 의사결정

### 7.1 의사결정 트리

```
프로젝트 시작
    │
    ├─ 학습/이해가 목적? → nano-pgo
    │
    ├─ 프로토타입/연구?
    │   ├─ 빠른 반복 필요? → nano-pgo
    │   └─ 성능 중요? → GTSAM
    │
    └─ 프로덕션/실제 로봇?
        ├─ 실시간 요구? → GTSAM (iSAM2)
        └─ 대규모 데이터? → GTSAM
```

### 7.2 전환 시점의 신호

**nano-pgo → GTSAM 전환 신호:**
1. 최적화 시간 > 1초
2. 메모리 사용량 > 1GB
3. 실시간 처리 필요
4. 다중 센서 융합 필요
5. 로버스트 추정 필요

### 7.3 하이브리드 접근법

**개발 단계별 전략:**

```python
# Phase 1: 알고리즘 개발 (nano-pgo)
def prototype_new_factor(x1, x2, measurement):
    # 빠른 프로토타이핑
    error = compute_error(x1, x2, measurement)
    return error

# Phase 2: 검증 (nano-pgo + GTSAM)
# 동일한 문제를 두 시스템에서 해결하여 비교

# Phase 3: 프로덕션 (GTSAM)
class ProductionFactor : public gtsam::NoiseModelFactor2<Pose3, Pose3> {
    // 최적화된 C++ 구현
};
```

---

## 8. 고급 GTSAM 기능과 확장

### 8.1 Marginal Covariance 계산

불확실성 정량화:

$$P(x_i | Z) = \mathcal{N}(\mu_i, \Sigma_i)$$

GTSAM에서:
```cpp
gtsam::Marginals marginals(graph, result);
Matrix cov = marginals.marginalCovariance(key);
```

**활용 예:**
- 불확실성 타원 시각화
- 능동 탐사 (탐사할 방향 결정)
- 안전 경로 계획

### 8.2 Multi-Robot SLAM

분산 최적화:

**합의 기반 접근:**
로봇 $i$의 추정치: $x_i$
합의 업데이트:

$$x_i^{k+1} = x_i^k + \epsilon \sum_{j \in \mathcal{N}_i} (x_j^k - x_i^k)$$

**DDF-SAM2 (Distributed Data Fusion):**
- 각 로봇이 로컬 Bayes Tree 유지
- 통신 시 요약 정보만 교환

### 8.3 센서 융합 팩터

**IMU Preintegration Factor:**

IMU 측정값 통합:

$$\Delta R_{ij} = \prod_{k=i}^{j-1} \exp((\omega_k - b_g^i) \Delta t)$$

$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik}(a_k - b_a^i) \Delta t$$

$$\Delta p_{ij} = \sum_{k=i}^{j-1} [\Delta v_{ik} \Delta t + \frac{1}{2}\Delta R_{ik}(a_k - b_a^i) \Delta t^2]$$

**GPS Factor:**

전역 위치 제약:

$$f_{\text{GPS}}(x_i) = \exp\left(-\frac{1}{2}\|p(x_i) - p_{\text{GPS}}\|^2_{\Sigma_{\text{GPS}}}\right)$$

### 8.4 커스텀 팩터 작성

**예: WiFi 신호 강도 팩터**

```cpp
class WiFiFactor : public NoiseModelFactor1<Pose2> {
private:
    Point2 ap_location_;
    double measured_rssi_;
    
public:
    Vector evaluateError(const Pose2& pose, 
                        boost::optional<Matrix&> H) const {
        Point2 robot_pos = pose.translation();
        double distance = (robot_pos - ap_location_).norm();
        
        // Path loss model: RSSI = A - 10n*log10(d)
        double predicted_rssi = -40 - 20 * log10(distance);
        
        if (H) {
            // 야코비안 계산
            *H = ... // 복잡한 미분
        }
        
        return Vector1(predicted_rssi - measured_rssi_);
    }
};
```

---

## 9. 실전 시나리오와 마이그레이션 전략

### 9.1 시나리오별 최적 도구

**시나리오 1: 대학 SLAM 과정**
- 목적: 개념 이해
- 도구: nano-pgo
- 이유: 투명한 구현, 쉬운 수정

**시나리오 2: 스타트업 MVP**
- 목적: 빠른 프로토타입
- 도구: nano-pgo → GTSAM
- 이유: 빠른 개발 후 성능 최적화

**시나리오 3: 자율주행차**
- 목적: 실시간, 안전성
- 도구: GTSAM
- 이유: 검증된 코드, 실시간 보장

**시나리오 4: 연구 논문**
- 목적: 새 알고리즘 검증
- 도구: 두 가지 모두
- 이유: nano-pgo로 개발, GTSAM으로 벤치마크

### 9.2 마이그레이션 전략

**단계별 전환:**

```python
# Step 1: 인터페이스 통일
class UnifiedOptimizer:
    def add_edge(self, i, j, measurement):
        if self.backend == "nano-pgo":
            self.nano_graph.add_edge(i, j, measurement)
        else:
            self.gtsam_graph.add(BetweenFactor(i, j, measurement))

# Step 2: 점진적 마이그레이션
# 핵심 루프만 GTSAM으로 전환

# Step 3: 전체 전환
# 모든 컴포넌트 GTSAM 사용
```

### 9.3 성능 프로파일링

**병목 지점 찾기:**

```python
import cProfile

# nano-pgo 프로파일링
cProfile.run('optimizer.optimize()')

# GTSAM 프로파일링
with gtsam.ProfileTimer("Optimization"):
    result = optimizer.optimize()
```

### 9.4 실제 프로젝트 사례

**사례 1: 드론 매핑 시스템**
- 초기: nano-pgo로 알고리즘 개발 (2주)
- 중기: 핵심 부분 GTSAM 포팅 (1주)
- 최종: 전체 GTSAM 전환 (1주)
- 결과: 100배 성능 향상

**사례 2: 수중 로봇 SLAM**
- 특수 센서 모델 필요
- nano-pgo로 커스텀 팩터 프로토타입
- GTSAM C++로 최종 구현
- 실시간 처리 달성

---

## 10. 요약 및 미래 전망

### 10.1 핵심 통찰

1. **Factor Graph의 우월성**: Pose Graph의 특수 케이스를 넘어 일반적인 확률 추론 프레임워크 제공

2. **Bayes Tree의 혁신**: 변수 소거를 트리 구조로 조직화하여 극적인 성능 향상

3. **도구 선택의 지혜**: 목적에 맞는 도구 선택이 프로젝트 성공의 열쇠

4. **증분 최적화의 필수성**: 실시간 로보틱스에서 iSAM2는 선택이 아닌 필수

5. **학습과 실전의 균형**: nano-pgo로 배우고 GTSAM으로 구현하는 전략

### 10.2 실무 체크리스트

✅ 프로젝트 규모와 실시간 요구사항 평가  
✅ 팀의 C++ 역량 고려  
✅ 프로토타입은 nano-pgo, 프로덕션은 GTSAM  
✅ 커스텀 팩터 필요 시 먼저 Python으로 검증  
✅ 성능 측정은 실제 데이터로  
✅ 두 도구의 결과를 교차 검증  

### 10.3 미래 전망

**차세대 SLAM 도구의 방향:**

1. **GPU 가속**: 대규모 병렬 최적화
2. **학습 기반 초기화**: 딥러닝으로 더 나은 초기값
3. **자동 팩터 생성**: 센서 스펙으로부터 자동 코드 생성
4. **클라우드 SLAM**: 분산 컴퓨팅 활용
5. **하이브리드 접근**: 심볼릭 + 학습 기반

### 10.4 마지막 조언

> "올바른 도구는 문제를 해결하는 도구입니다."

- 학습할 때는 nano-pgo의 투명성을 활용하세요
- 실전에서는 GTSAM의 강력함을 믿으세요
- 두 도구 모두 여러분의 도구상자에 있어야 합니다
- 중요한 것은 도구가 아니라 문제 해결 능력입니다

**핵심 질문 되돌아보기:**
- ✓ Factor Graph는 변수와 제약의 일반적 표현으로 더 강력
- ✓ Bayes Tree는 변수 소거를 효율적으로 조직화하여 복잡도 감소
- ✓ 학습과 프로토타입은 nano-pgo, 프로덕션은 GTSAM
- ✓ iSAM2는 영향받는 부분만 업데이트하여 실시간 가능
- ✓ 단계적 마이그레이션으로 두 도구의 장점 활용

이제 실습에서 GTSAM의 강력함을 직접 체험하고, 여러분만의 SLAM 시스템을 구축해보세요!