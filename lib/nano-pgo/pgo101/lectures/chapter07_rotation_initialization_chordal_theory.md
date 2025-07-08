# PGO 101 - Chapter 7 이론 강의: 최적화의 첫 단추 - 회전 초기화

**강의 목표:** 이 강의를 마치면, 여러분은 Pose Graph Optimization의 성공 여부를 결정하는 매우 중요한 첫 단계인 **초기값 설정 (Initialization)** 의 중요성을 이해하게 됩니다. 특히, 위치보다 회전 오차에 더 민감한 최적화 문제의 특성을 파악하고, **회전 동기화 (Rotation Synchronization)** 라는 문제를 푸는 강력한 기법인 **코달 완화 (Chordal Relaxation)** 의 원리를 설명할 수 있게 됩니다. 이 강의는 `chapter07_rotation_initialization_chordal.ipynb` 실습에서 좋은 초기값을 구하는 알고리즘을 구현하기 위한 이론적 기반을 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 초기화가 PGO의 성패를 좌우하는가?
> - SO(3) 매니폴드의 어떤 특성이 최적화를 어렵게 만드는가?
> - 코달 거리와 측지 거리의 차이는 무엇인가?
> - 어떻게 비볼록 문제를 볼록 문제로 변환할 수 있는가?
> - 분산 환경에서 회전 동기화를 어떻게 수행하는가?

---

## 목차

1. [초기화의 중요성 - 성공과 실패의 갈림길](#1-초기화의-중요성---성공과-실패의-갈림길)
2. [회전 동기화 문제의 수학적 정의](#2-회전-동기화-문제의-수학적-정의)
3. [SO(3) 매니폴드와 최적화의 도전](#3-so3-매니폴드와-최적화의-도전)
4. [코달 거리 vs 측지 거리](#4-코달-거리-vs-측지-거리)
5. [코달 완화의 수학적 유도](#5-코달-완화의-수학적-유도)
6. [스펙트럴 방법과 회전 라플라시안](#6-스펙트럴-방법과-회전-라플라시안)
7. [SVD 기반 SO(3) 투영](#7-svd-기반-so3-투영)
8. [반복적 회전 평균화](#8-반복적-회전-평균화)
9. [고급 주제들](#9-고급-주제들)
10. [요약 및 실전 가이드](#10-요약-및-실전-가이드)

---

## 1. 초기화의 중요성 - 성공과 실패의 갈림길

### 1.1 비선형 최적화의 본질

Pose Graph Optimization은 다음과 같은 비선형 최소 제곱 문제입니다:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{e}_{ij}(\mathbf{x})\|^2_{\Omega_{ij}}$$

이 문제의 핵심 특징:
- **비볼록성**: 여러 개의 국소 최소값 존재
- **고차원성**: 수백~수천 개의 변수
- **비선형성**: 회전으로 인한 삼각함수

### 1.2 초기값이 결정하는 수렴 운명

**정리 (Basin of Attraction):** 
비선형 최적화에서 각 국소 최소값 $\mathbf{x}^*_i$ 는 수렴 영역 $\mathcal{B}_i$ 를 가지며, 초기값 $\mathbf{x}_0 \in \mathcal{B}_i$ 일 때만 해당 최소값으로 수렴한다.

**실제 SLAM에서의 의미:**
```
좋은 초기값: |x₀ - x_global| < δ_good
→ 10-20회 반복으로 전역 최적해 도달

나쁜 초기값: |x₀ - x_global| > δ_bad  
→ 100+회 반복 후에도 국소 최적해 갇힘
```

### 1.3 회전이 주도하는 비선형성

**위치 vs 회전의 영향력 분석:**

간단한 2D 예시를 통해 보면:
- 위치 오차 $\Delta p$ 의 영향: $O(\Delta p)$ (선형)
- 회전 오차 $\Delta \theta$ 의 영향: $O(L \sin(\Delta \theta))$ (비선형)

여기서 $L$ 은 레버 암(lever arm) 길이입니다. 작은 회전 오차도 먼 거리에서는 큰 위치 오차를 유발합니다.

**수치 예시:**
- 1도 회전 오차 + 10m 거리 = 17.5cm 위치 오차
- 5도 회전 오차 + 10m 거리 = 87.2cm 위치 오차

---

## 2. 회전 동기화 문제의 수학적 정의

### 2.1 문제 설정

**Given:**
- 노드 집합: $V = \{1, 2, ..., n\}$
- 엣지 집합: $\mathcal{E} \subseteq V \times V$
- 상대 회전 측정값: $R_{ij} \in SO(3)$ for $(i,j) \in \mathcal{E}$
- 가중치: $w_{ij} > 0$ (측정 신뢰도)

**Find:**
절대 회전: $R_1, R_2, ..., R_n \in SO(3)$

### 2.2 최적화 문제 공식화

$$\min_{R_1,...,R_n \in SO(3)} \sum_{(i,j) \in \mathcal{E}} w_{ij} \|R_i^T R_j - R_{ij}\|_F^2$$

**등가 공식 (더 일반적):**
$$\min_{R_1,...,R_n \in SO(3)} \sum_{(i,j) \in \mathcal{E}} w_{ij} \|R_j - R_i R_{ij}\|_F^2$$

### 2.3 응용 분야

1. **SLAM**: 로봇의 자세 초기화
2. **Structure from Motion**: 카메라 방향 추정
3. **센서 네트워크**: 분산 센서 캘리브레이션
4. **분자 구조 재구성**: 원자 방향 결정

---

## 3. SO(3) 매니폴드와 최적화의 도전

### 3.1 SO(3)의 기하학적 구조

**정의 (Special Orthogonal Group):**
$$SO(3) = \{R \in \mathbb{R}^{3 \times 3} : R^T R = I, \det(R) = 1\}$$

**핵심 속성:**
- **차원**: 3 (9개 원소 - 6개 직교 제약)
- **위상**: 컴팩트, 연결된 리 군
- **접선 공간**: $T_R SO(3) = \{R\Omega : \Omega^T = -\Omega\}$

### 3.2 왜 SO(3) 최적화가 어려운가?

**1. 비유클리드 구조:**
```python
# 이것은 틀렸다!
R_avg = (R1 + R2) / 2  # 일반적으로 회전 행렬이 아님

# 올바른 평균은 측지선 상에서 계산
R_avg = exp(log(R1) + log(R2)) / 2)
```

**2. 다중 매개변수화의 문제:**
- **오일러 각**: 짐벌 락 특이점
- **축-각도**: $\pi$ 근처에서 불연속
- **쿼터니언**: 이중 커버 ($q$ 와 $-q$ 가 같은 회전)

**3. 제약 조건의 비볼록성:**
$$f(R) = \|R^T R - I\|_F^2 + (\det(R) - 1)^2$$
이 제약 함수는 비볼록하여 국소 최소값을 가집니다.

---

## 4. 코달 거리 vs 측지 거리

### 4.1 측지 거리 (Geodesic Distance)

SO(3) 매니폴드 상의 최단 경로 길이:
$$d_g(R_1, R_2) = \|\log(R_1^T R_2)\|_F = |\theta|$$

여기서 $\theta$ 는 $R_1$ 에서 $R_2$ 로의 회전 각도입니다.

### 4.2 코달 거리 (Chordal Distance)

유클리드 공간에서의 직선 거리:
$$d_c(R_1, R_2) = \|R_1 - R_2\|_F$$

### 4.3 두 거리의 관계

**정리 (거리 관계):**
$$2\sqrt{2}\sin\left(\frac{\theta}{2}\right) \leq d_c(R_1, R_2) \leq 2\sin\left(\frac{\theta}{2}\right)\sqrt{4 - 2\cos(\theta)}$$

**근사 오차 분석:**
Taylor 전개를 통해:
$$d_c^2 = 4 - 4\cos(\theta) = 2\theta^2 - \frac{\theta^4}{12} + O(\theta^6)$$

따라서 작은 각도에서:
$$d_c \approx \sqrt{2}\theta + O(\theta^3)$$

**실용적 함의:**
- $\theta < 30°$: 오차 < 2%
- $\theta < 60°$: 오차 < 10%
- $\theta > 90°$: 심각한 오차

---

## 5. 코달 완화의 수학적 유도

### 5.1 원래 문제에서 완화 문제로

**Step 1: 제약 완화**
$$\min_{R_i \in SO(3)} \sum_{(i,j)} w_{ij} \|R_j - R_i R_{ij}\|_F^2$$
↓
$$\min_{X_i \in \mathbb{R}^{3 \times 3}} \sum_{(i,j)} w_{ij} \|X_j - X_i R_{ij}\|_F^2$$

**Step 2: 벡터화**
$\mathbf{x}_i = \text{vec}(X_i) \in \mathbb{R}^9$ 로 정의하면:
$$\|X_j - X_i R_{ij}\|_F^2 = \|\mathbf{x}_j - (I_3 \otimes R_{ij}^T) \mathbf{x}_i\|^2$$

### 5.2 이차 형식으로 변환

전체 비용 함수는:
$$f(\mathbf{x}) = \mathbf{x}^T Q \mathbf{x}$$

여기서 $\mathbf{x} = [\mathbf{x}_1^T, ..., \mathbf{x}_n^T]^T \in \mathbb{R}^{9n}$ 이고, $Q$ 는 블록 행렬:

$$Q_{ii} = \sum_{j:(i,j) \in \mathcal{E}} w_{ij} (I_9 + (I_3 \otimes R_{ij})(I_3 \otimes R_{ij})^T)$$

$$Q_{ij} = \begin{cases}
-w_{ij}(I_9 + (I_3 \otimes R_{ij})) & \text{if } (i,j) \in \mathcal{E} \\
0 & \text{otherwise}
\end{cases}$$

### 5.3 고유값 문제로의 변환

**정리 (Rayleigh-Ritz):**
제약 없는 최소화 문제 $\min_{\mathbf{x}} \mathbf{x}^T Q \mathbf{x}$ 의 해는 $Q$ 의 최소 고유값에 대응하는 고유벡터입니다.

**Gauge Freedom 처리:**
회전 동기화는 전역 회전에 대한 자유도를 가집니다. 이를 고정하기 위해:
1. 한 회전을 고정: $R_1 = I$
2. 또는 평균 제약: $\sum_i R_i = 0$

---

## 6. 스펙트럴 방법과 회전 라플라시안

### 6.1 회전 라플라시안 구성

그래프 라플라시안과 유사하게:

**가중 인접 행렬:**
$$W_{ij} = \begin{cases}
w_{ij} & \text{if } (i,j) \in \mathcal{E} \\
0 & \text{otherwise}
\end{cases}$$

**차수 행렬:**
$$D_{ii} = \sum_j W_{ij}$$

**회전 라플라시안:**
$$L = D \otimes I_9 - W \otimes I_9$$

### 6.2 스펙트럴 완화

**정리:** 회전 동기화 문제의 스펙트럴 완화는 다음과 같이 표현됩니다:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{s.t.} \quad \|\mathbf{x}\|^2 = n$$

**해법:**
1. $L$ 의 가장 작은 9개 고유값에 대응하는 고유벡터 계산
2. 이론적으로 최소 3개 고유값은 0 (또는 매우 작음)
3. 다음 3개 고유벡터가 회전의 3개 열을 형성

### 6.3 알고리즘

```python
def spectral_rotation_sync(edges, weights, n_nodes):
    # 1. 라플라시안 구성
    L = build_rotation_laplacian(edges, weights, n_nodes)
    
    # 2. 고유값 분해
    eigenvalues, eigenvectors = eigsh(L, k=9, which='SM')
    
    # 3. 회전 행렬 재구성
    rotations = []
    for i in range(n_nodes):
        R_vec = eigenvectors[9*i:9*(i+1), 3:6]
        R = R_vec.reshape(3, 3)
        rotations.append(project_to_SO3(R))
    
    return rotations
```

---

## 7. SVD 기반 SO(3) 투영

### 7.1 최근접 회전 행렬 문제

**문제:** 주어진 행렬 $M \in \mathbb{R}^{3 \times 3}$ 에 대해, 가장 가까운 회전 행렬 찾기:
$$R^* = \arg\min_{R \in SO(3)} \|R - M\|_F^2$$

### 7.2 SVD 해법

**정리 (Kabsch):** $M = U\Sigma V^T$ 가 $M$ 의 SVD일 때:
$$R^* = UV^T \cdot \text{sign}(\det(UV^T))$$

더 정확히는:
$$R^* = U \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(UV^T) \end{bmatrix} V^T$$

### 7.3 증명

**Step 1:** 문제를 다시 쓰면:
$$\max_{R \in SO(3)} \text{tr}(R^T M)$$

**Step 2:** $M = U\Sigma V^T$ 를 대입:
$$\text{tr}(R^T U\Sigma V^T) = \text{tr}(\Sigma V^T R^T U)$$

**Step 3:** $A = V^T R^T U$ 는 직교 행렬이고:
$$\text{tr}(\Sigma A) = \sum_i \sigma_i A_{ii} \leq \sum_i \sigma_i$$

등호는 $A = I$ 일 때 성립하므로 $R = UV^T$.

---

## 8. 반복적 회전 평균화

### 8.1 접선 공간에서의 업데이트

코달 완화의 대안으로, 반복적으로 회전을 개선하는 방법:

**업데이트 규칙:**
$$R_i^{(k+1)} = R_i^{(k)} \exp\left(\sum_{j \in \mathcal{N}(i)} w_{ij} \log(R_i^{(k)T} R_j^{(k)} R_{ij}^T)\right)$$

### 8.2 수렴성 분석

**정리:** 충분히 작은 스텝 크기 $\alpha$ 에 대해, 반복 알고리즘:
$$R_i^{(k+1)} = R_i^{(k)} \exp\left(\alpha \sum_{j} w_{ij} \log(R_i^{(k)T} R_j^{(k)} R_{ij}^T)\right)$$

는 국소 최소값으로 수렴합니다.

### 8.3 분산 구현

**합의 기반 알고리즘:**
```python
def distributed_rotation_averaging(neighbors, measurements, max_iter):
    # 각 노드가 독립적으로 실행
    for iteration in range(max_iter):
        # 이웃으로부터 정보 수집
        neighbor_rotations = gather_from_neighbors()
        
        # 로컬 업데이트
        tangent_sum = np.zeros(3)
        for j, R_ij in neighbors:
            tangent = log_SO3(R_j @ R_ij.T @ R_i.T)
            tangent_sum += weight[j] * tangent
        
        # 회전 업데이트
        R_i = R_i @ exp_SO3(alpha * tangent_sum)
```

---

## 9. 고급 주제들

### 9.1 반정부호 계획법 (SDP) 완화

더 정확한 완화를 위해 SDP 사용:

**SDP 공식화:**
$$\min \text{tr}(CX) \quad \text{s.t.} \quad X \succeq 0, \quad X_{ii} = I_3$$

여기서 $X \in \mathbb{R}^{3n \times 3n}$ 는 모든 회전의 그람 행렬입니다.

### 9.2 불확실성을 고려한 회전 평균화

각 측정값의 공분산 $\Sigma_{ij}$ 를 고려:

$$\min \sum_{(i,j)} (R_j - R_i R_{ij})^T \Sigma_{ij}^{-1} (R_j - R_i R_{ij})$$

### 9.3 로버스트 회전 동기화

이상치에 강건한 비용 함수 사용:

$$\min \sum_{(i,j)} \rho(\|R_j - R_i R_{ij}\|_F)$$

여기서 $\rho$ 는 Huber 또는 Cauchy 함수입니다.

### 9.4 계산 복잡도 분석

**코달 완화:**
- 라플라시안 구성: $O(|\mathcal{E}|)$
- 고유값 분해: $O(n^3)$ (전체), $O(kn^2)$ (sparse, k 반복)
- SVD 투영: $O(n)$

**반복 방법:**
- 각 반복: $O(|\mathcal{E}|)$
- 총 복잡도: $O(k|\mathcal{E}|)$, k는 반복 횟수

---

## 10. 요약 및 실전 가이드

### 10.1 핵심 개념 정리

1. **초기화는 PGO의 성패를 결정**: 좋은 초기값 → 전역 최적해, 나쁜 초기값 → 국소 최적해

2. **SO(3)의 비유클리드 구조**: 직접 최적화가 어려워 완화 필요

3. **코달 완화의 천재성**: 비볼록 → 볼록 → 투영의 3단계 전략

4. **스펙트럴 방법의 효율성**: 고유값 분해로 전역 해 획득

5. **실용적 고려사항**: 
   - 밀집 그래프: 코달 완화
   - 희소 그래프: 반복 방법
   - 이상치 존재: 로버스트 방법

### 10.2 구현 체크리스트

✅ 상대 회전의 일관성 검사  
✅ 가중치 정규화 (수치 안정성)  
✅ Gauge freedom 처리 (한 회전 고정)  
✅ SVD에서 반사 처리  
✅ 수렴 판정 기준 설정  

### 10.3 실전 팁

1. **초기 스크리닝**: 명백히 잘못된 측정값 제거
2. **다단계 접근**: 코달 → 반복 정제
3. **병렬화**: 고유값 분해와 SVD 투영 병렬 처리
4. **메모리 효율**: 희소 행렬 자료구조 활용

### 10.4 다음 장으로

이제 좋은 회전 초기값을 얻었으니, Chapter 8에서는 **루프 클로저**와 **전역 최적화**를 다룹니다. 초기값이 좋아도 루프 클로저가 잘못되면 전체 지도가 왜곡될 수 있습니다.

**핵심 질문 되돌아보기:**
- ✓ 초기화는 수렴 영역을 결정하여 성패 좌우
- ✓ SO(3)는 비유클리드 매니폴드로 제약이 복잡
- ✓ 코달은 직선 거리, 측지는 곡면 거리
- ✓ 완화 → 고유값 분해 → 투영의 3단계
- ✓ 분산 환경에서는 합의 기반 반복 알고리즘

이제 실습에서 이 이론을 코드로 구현하고 그 효과를 직접 확인해보세요!