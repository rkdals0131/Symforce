# PGO 101 - Chapter 6 이론 강의: 현실의 노이즈와 싸우기 - 강인한 최적화

**강의 목표:** 이 강의를 마치면, 여러분은 실제 SLAM 상황에서 왜 표준 최소 제곱법이 실패할 수 있는지, 그리고 **이상치 (Outlier)** 문제를 해결하기 위한 **강인한 최적화 (Robust Optimization)** 기법이 왜 필수적인지 이해하게 됩니다. 다양한 강인한 비용 함수 (Robust Kernel) 의 원리를 배우고, 이를 **반복적 재가중 최소 제곱법 (IRLS)** 을 통해 구현하는 방법을 설명할 수 있게 됩니다. 이 강의는 `chapter06_robust_optimization_with_cauchy.ipynb` 실습에서 이상치에 강건한 옵티마이저를 구현하기 위한 이론적 기반을 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 실제 SLAM에서 이상치가 불가피한가?
> - M-estimator의 통계학적 기반은 무엇인가?
> - IRLS가 어떻게 비볼록 문제를 효율적으로 푸는가?
> - 각 robust kernel의 확률 분포적 해석은?
> - 수렴성을 어떻게 보장할 수 있는가?

---

## 목차

1. [이상치 문제의 본질](#1-이상치-문제의-본질)
2. [M-추정자의 통계학적 기초](#2-m-추정자의-통계학적-기초)
3. [강인한 비용 함수의 수학적 분석](#3-강인한-비용-함수의-수학적-분석)
4. [IRLS 알고리즘의 유도와 수렴성](#4-irls-알고리즘의-유도와-수렴성)
5. [주요 Robust Kernel의 상세 분석](#5-주요-robust-kernel의-상세-분석)
6. [확률 분포와의 연결](#6-확률-분포와의-연결)
7. [Chi-squared 검정을 통한 이상치 탐지](#7-chi-squared-검정을-통한-이상치-탐지)
8. [고급 기법들](#8-고급-기법들)
9. [실전 구현 전략](#9-실전-구현-전략)
10. [요약 및 다음 장 예고](#10-요약-및-다음-장-예고)

---

## 1. 이상치 문제의 본질

### 1.1 최대 우도 추정과 가우시안 가정

표준 최소 제곱법은 **최대 우도 추정 (Maximum Likelihood Estimation, MLE)** 의 특수한 경우입니다. 측정 오차가 평균 0, 공분산 $\Sigma$ 인 가우시안 분포를 따른다고 가정하면:

$$p(\mathbf{z} | \mathbf{x}) = \prod_i \frac{1}{\sqrt{(2\pi)^n |\Sigma_i|}} \exp\left(-\frac{1}{2} \mathbf{e}_i^T \Sigma_i^{-1} \mathbf{e}_i\right)$$

여기서 $\mathbf{e}_i = \mathbf{z}_i - h_i(\mathbf{x})$ 는 잔차입니다. 로그 우도를 최대화하면:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} \log p(\mathbf{z} | \mathbf{x}) = \arg\min_{\mathbf{x}} \sum_i \mathbf{e}_i^T \Sigma_i^{-1} \mathbf{e}_i$$

이것이 바로 우리가 익숙한 최소 제곱 문제입니다.

### 1.2 이상치의 통계학적 모델

실제 데이터는 두 가지 분포의 혼합으로 모델링할 수 있습니다:

$$p(\mathbf{z}) = (1-\epsilon) \cdot p_{\text{inlier}}(\mathbf{z}) + \epsilon \cdot p_{\text{outlier}}(\mathbf{z})$$

- $p_{\text{inlier}}$: 정상 측정값의 분포 (좁은 가우시안)
- $p_{\text{outlier}}$: 이상치의 분포 (넓은 가우시안 또는 균등 분포)
- $\epsilon$: 이상치 비율

### 1.3 SLAM에서의 이상치 원인

**1. 잘못된 루프 클로저 (Perceptual Aliasing)**
```
실제: 로봇이 위치 A에서 B로 이동
잘못된 인식: B를 이전에 방문한 C로 착각
결과: |A-C| >> |A-B| 인 거대한 오차 발생
```

**2. 센서 오류의 유형**
- **LiDAR**: 유리 반사, 빗방울, 안개
- **카메라**: 모션 블러, 조명 변화, 렌즈 플레어
- **IMU**: 자기장 간섭, 진동 충격

**3. 동적 환경**
- 움직이는 물체를 정적 랜드마크로 오인
- 시간차 측정으로 인한 불일치

---

## 2. M-추정자의 통계학적 기초

### 2.1 M-추정자의 정의

**M-추정자 (Maximum likelihood-type estimator)** 는 MLE를 일반화한 것으로, 다음을 최소화합니다:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_i \rho(e_i(\mathbf{x}))$$

여기서 $\rho(\cdot)$ 는 robust loss function입니다.

### 2.2 영향 함수 (Influence Function)

영향 함수 $\psi(e) = \frac{d\rho(e)}{de}$ 는 한 측정값이 추정에 미치는 영향을 나타냅니다:

$$\sum_i \psi(e_i) \frac{\partial e_i}{\partial \mathbf{x}} = 0$$

**핵심 속성:**
- **Bounded**: $|\psi(e)| < M$ for some $M$
- **Redescending**: $\lim_{|e| \to \infty} \psi(e) = 0$

### 2.3 붕괴점 (Breakdown Point)

붕괴점은 추정자가 무의미해지기 전까지 견딜 수 있는 최대 이상치 비율입니다:

$$\epsilon^* = \sup\{\epsilon : \sup_{\mathbf{z}_{\text{corrupted}}} ||\hat{\mathbf{x}}(\mathbf{z}_{\text{corrupted}}) - \mathbf{x}_{\text{true}}|| < \infty\}$$

**예시:**
- 평균: $\epsilon^* = 0$ (단 하나의 이상치도 치명적)
- 중앙값: $\epsilon^* = 0.5$ (50%까지 견딤)
- Huber M-estimator: $\epsilon^* = 0$ (하지만 영향은 제한됨)

---

## 3. 강인한 비용 함수의 수학적 분석

### 3.1 볼록성과 최적화

비용 함수의 볼록성은 최적화의 수렴성을 결정합니다:

**정의:** $\rho(e)$ 가 볼록하려면:
$$\rho(\lambda e_1 + (1-\lambda) e_2) \leq \lambda \rho(e_1) + (1-\lambda) \rho(e_2)$$

모든 $e_1, e_2$ 와 $\lambda \in [0,1]$ 에 대해 성립해야 합니다.

### 3.2 점근적 효율성 (Asymptotic Efficiency)

가우시안 노이즈 하에서 M-추정자의 상대적 효율성:

$$\text{ARE} = \frac{\text{Var}[\hat{\mathbf{x}}_{\text{MLE}}]}{\text{Var}[\hat{\mathbf{x}}_{\text{M-estimator}}]}$$

**Trade-off**: 이상치에 강건할수록 정상 데이터에서의 효율성은 감소합니다.

---

## 4. IRLS 알고리즘의 유도와 수렴성

### 4.1 수학적 유도

강건한 비용 함수 최소화 문제:

$$\min_{\mathbf{x}} F(\mathbf{x}) = \sum_i \rho(e_i(\mathbf{x}))$$

최적성 조건 (1차 필요조건):

$$\nabla F = \sum_i \psi(e_i) \nabla e_i = 0$$

여기서 $\psi(e) = \rho'(e)$ 는 영향 함수입니다.

### 4.2 가중치 함수 도입

$w(e) = \frac{\psi(e)}{e}$ for $e \neq 0$ 로 정의하면:

$$\sum_i w(e_i) e_i \nabla e_i = 0$$

이는 가중 최소 제곱 문제의 정규 방정식과 동일합니다!

### 4.3 IRLS 알고리즘

**반복 $k$ 에서:**

1. **잔차 계산**: $e_i^{(k)} = z_i - h_i(\mathbf{x}^{(k)})$

2. **가중치 업데이트**: 
   $$w_i^{(k)} = \frac{\psi(||e_i^{(k)}||)}{||e_i^{(k)}||}$$

3. **선형화**: $e_i(\mathbf{x}) \approx e_i^{(k)} + J_i^{(k)} \Delta\mathbf{x}$

4. **가중 정규 방정식 풀이**:
   $$\left(\sum_i J_i^T W_i^{(k)} J_i\right) \Delta\mathbf{x} = -\sum_i J_i^T W_i^{(k)} e_i^{(k)}$$
   
   여기서 $W_i^{(k)} = w_i^{(k)} \Omega_i$

5. **업데이트**: $\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \Delta\mathbf{x}$

### 4.4 수렴성 증명 (볼록 커널의 경우)

**정리:** $\rho(e)$ 가 볼록하고 미분가능하면, IRLS는 단조 감소 성질을 만족합니다:

$$F(\mathbf{x}^{(k+1)}) \leq F(\mathbf{x}^{(k)})$$

**증명 스케치:**

1. Taylor 전개:
   $$\rho(e^{(k+1)}) \approx \rho(e^{(k)}) + \psi(e^{(k)})(e^{(k+1)} - e^{(k)})$$

2. 가중 최소 제곱 해는 다음을 최소화:
   $$\sum_i w_i^{(k)} (e_i^{(k+1)})^2$$

3. 볼록성과 Jensen 부등식을 이용하여 감소 성질 증명

---

## 5. 주요 Robust Kernel의 상세 분석

### 5.1 L2 (Gaussian) Kernel

$$\rho(e) = \frac{1}{2}e^2, \quad \psi(e) = e, \quad w(e) = 1$$

**특성:**
- 볼록성: ✓ (강볼록)
- 영향 함수: 무한대 (unbounded)
- 최적성: 가우시안 노이즈에 최적

### 5.2 Huber Kernel

$$\rho(e) = \begin{cases}
\frac{1}{2}e^2 & |e| \leq k \\
k|e| - \frac{1}{2}k^2 & |e| > k
\end{cases}$$

$$\psi(e) = \begin{cases}
e & |e| \leq k \\
k \cdot \text{sign}(e) & |e| > k
\end{cases}$$

$$w(e) = \begin{cases}
1 & |e| \leq k \\
k/|e| & |e| > k
\end{cases}$$

**특성:**
- 볼록성: ✓
- 영향 함수: 제한됨 (bounded)
- 매개변수 선택: $k = 1.345\sigma$ 는 95% 효율성 제공

### 5.3 Cauchy (Lorentzian) Kernel

$$\rho(e) = \frac{c^2}{2} \log\left(1 + \left(\frac{e}{c}\right)^2\right)$$

$$\psi(e) = \frac{e}{1 + (e/c)^2}$$

$$w(e) = \frac{1}{1 + (e/c)^2}$$

**특성:**
- 볼록성: ✗ (비볼록)
- 영향 함수: Redescending
- 점근적 행동: $\rho(e) \sim c^2 \log|e|$ as $|e| \to \infty$

### 5.4 Tukey Biweight Kernel

$$\rho(e) = \begin{cases}
\frac{c^2}{6}\left[1 - \left(1 - \left(\frac{e}{c}\right)^2\right)^3\right] & |e| \leq c \\
\frac{c^2}{6} & |e| > c
\end{cases}$$

$$\psi(e) = \begin{cases}
e\left(1 - \left(\frac{e}{c}\right)^2\right)^2 & |e| \leq c \\
0 & |e| > c
\end{cases}$$

**특성:**
- 볼록성: ✗ (비볼록)
- 영향 함수: Hard redescending
- 완전 거부: $|e| > c$ 인 측정값은 무시

### 5.5 비교 분석

```
에러 크기에 따른 가중치 변화:
e/c    | L2   | Huber | Cauchy | Tukey
-------|------|-------|--------|-------
0.5    | 1.00 | 1.00  | 0.80   | 0.94
1.0    | 1.00 | 1.00  | 0.50   | 0.00
2.0    | 1.00 | 0.50  | 0.20   | 0.00
5.0    | 1.00 | 0.20  | 0.04   | 0.00
```

---

## 6. 확률 분포와의 연결

### 6.1 M-추정자와 확률 분포

각 robust kernel은 특정 오차 분포에 대한 MLE로 해석될 수 있습니다:

$$\rho(e) = -\log p(e) + \text{const}$$

### 6.2 Cauchy 분포

Cauchy (Lorentzian) 분포의 확률 밀도 함수:

$$p(e; \gamma) = \frac{1}{\pi\gamma \left(1 + \left(\frac{e}{\gamma}\right)^2\right)}$$

이는 Cauchy kernel과 직접 연결됩니다:

$$-\log p(e) \propto \log\left(1 + \left(\frac{e}{\gamma}\right)^2\right) = \frac{2}{c^2}\rho_{\text{Cauchy}}(e)$$

**특징:**
- Heavy-tailed 분포 (꼬리가 $1/e^2$ 로 감소)
- 평균과 분산이 정의되지 않음
- 이상치에 매우 강건

### 6.3 Student's t-분포

자유도 $\nu$ 인 t-분포:

$$p(e; \nu, \sigma) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}} \left(1 + \frac{e^2}{\nu\sigma^2}\right)^{-\frac{\nu+1}{2}}$$

- $\nu \to \infty$: 가우시안 분포로 수렴
- $\nu = 1$: Cauchy 분포
- $\nu = 4-6$: 실용적인 선택

---

## 7. Chi-squared 검정을 통한 이상치 탐지

### 7.1 정규화된 잔차

최적화 후 잔차의 통계적 검정:

$$r_i = \frac{e_i}{\sqrt{\sigma_i^2}} = \frac{e_i}{\sqrt{(J_i \Sigma_x J_i^T + R_i)_{ii}}}$$

여기서:
- $\Sigma_x = (J^T \Omega J)^{-1}$: 상태 추정의 공분산
- $R_i$: 측정 노이즈 공분산

### 7.2 Chi-squared 검정

가우시안 가정 하에서 $r_i^2 \sim \chi^2(m)$, 여기서 $m$ 은 측정값의 차원입니다.

**이상치 판별:**
$$r_i^2 > \chi^2_{m,1-\alpha} \Rightarrow \text{이상치로 판별}$$

여기서 $\alpha$ 는 유의수준 (예: 0.05)

### 7.3 Mahalanobis 거리

다변량 측정값의 경우:

$$d_i^2 = \mathbf{e}_i^T \Sigma_i^{-1} \mathbf{e}_i$$

이는 $\chi^2(m)$ 분포를 따르며, 다차원 이상치 검정에 사용됩니다.

---

## 8. 고급 기법들

### 8.1 Graduated Non-Convexity (GNC)

비볼록 문제를 점진적으로 해결하는 전략:

**알고리즘:**
1. 초기: 볼록 근사 (예: Huber with large $k$)
2. 반복적으로 비볼록성 증가:
   $$\rho_\mu(e) = (1-\mu)\rho_{\text{convex}}(e) + \mu\rho_{\text{non-convex}}(e)$$
3. $\mu: 0 \to 1$ 로 점진적 증가

**수렴성 개선:**
- 나쁜 국소 최솟값 회피
- 초기값 민감도 감소

### 8.2 Switchable Constraints

이진 변수를 사용한 명시적 이상치 거부:

$$\min_{\mathbf{x}, \mathbf{s}} \sum_i s_i \rho(e_i(\mathbf{x})) + \lambda \sum_i (1-s_i)$$

여기서 $s_i \in \{0,1\}$ 는 측정값 $i$ 의 활성화 여부

**완화 기법:**
- 연속 완화: $s_i \in [0,1]$
- 교대 최적화: $\mathbf{x}$ 와 $\mathbf{s}$ 를 번갈아 최적화

### 8.3 Dynamic Covariance Scaling (DCS)

적응적 정보 행렬 조정:

$$\Omega_i^{(k)} = \gamma_i^{(k)} \Omega_i^{(0)}$$

여기서 스케일 팩터:
$$\gamma_i^{(k)} = \frac{\psi(||e_i^{(k)}||)}{||e_i^{(k)}||}$$

### 8.4 Maximum Correntropy Criterion (MCC)

정보 이론적 접근:

$$\max_{\mathbf{x}} \sum_i \exp\left(-\frac{e_i(\mathbf{x})^2}{2\sigma^2}\right)$$

이는 Gaussian kernel을 사용한 correntropy 최대화와 동일합니다.

---

## 9. 실전 구현 전략

### 9.1 커널 매개변수 선택

**1. MAD (Median Absolute Deviation) 기반:**
$$\hat{\sigma} = 1.4826 \times \text{MAD} = 1.4826 \times \text{median}(|e_i - \text{median}(e_i)|)$$

**2. 센서별 튜닝:**
```python
# LiDAR: 작은 threshold (정밀 센서)
cauchy_lidar = CauchyKernel(delta=0.1)  # 10cm

# 카메라: 중간 threshold
cauchy_camera = CauchyKernel(delta=2.0)  # 2 pixels

# 휠 오도메트리: 큰 threshold (슬립 고려)
cauchy_wheel = CauchyKernel(delta=0.05)  # 5% of distance
```

### 9.2 다단계 최적화 전략

```python
def multi_stage_optimization(optimizer):
    # Stage 1: Conservative (Huber)
    optimizer.kernel = HuberKernel(delta=3.0*sigma)
    optimizer.optimize(max_iter=10)
    
    # Stage 2: Moderate (Cauchy)
    optimizer.kernel = CauchyKernel(delta=1.5*sigma)
    optimizer.optimize(max_iter=10)
    
    # Stage 3: Aggressive (Tukey)
    optimizer.kernel = TukeyKernel(delta=3.0*sigma)
    optimizer.optimize(max_iter=5)
```

### 9.3 수렴 가속 기법

**1. Damping (Levenberg-Marquardt):**
$$(J^T W J + \lambda I) \Delta\mathbf{x} = -J^T W \mathbf{e}$$

**2. Line Search:**
$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + \alpha^* \Delta\mathbf{x}$$

여기서 $\alpha^* = \arg\min_\alpha F(\mathbf{x}^{(k)} + \alpha \Delta\mathbf{x})$

### 9.4 실시간 SLAM을 위한 최적화

**1. Incremental 업데이트:**
- 새 측정값만 처리
- 이전 가중치 재사용

**2. 병렬화:**
```python
# 잔차와 가중치 계산 병렬화
parallel_for(edges) {
    compute_error_and_weight(edge)
}

# 희소 행렬 조립 병렬화
parallel_reduce(H_matrix, b_vector)
```

### 9.5 하이브리드 접근법

```python
class HybridRobustOptimizer:
    def optimize(self):
        # 1. RANSAC으로 명백한 이상치 제거
        inliers = ransac_filter(measurements)
        
        # 2. Robust optimization으로 정제
        result = irls_optimize(inliers, CauchyKernel())
        
        # 3. Chi-squared test로 추가 이상치 검출
        final_inliers = chi_squared_filter(result)
        
        # 4. 최종 최적화
        return optimize(final_inliers)
```

---

## 10. 요약 및 다음 장 예고

### 10.1 핵심 개념 정리

1. **이상치의 불가피성**: 실제 SLAM에서 센서 오류, 동적 환경, 데이터 연관 실패로 인한 이상치는 피할 수 없습니다.

2. **M-추정자의 힘**: 영향 함수를 제한하거나 redescending하게 만들어 이상치의 영향을 통제합니다.

3. **IRLS의 우아함**: 복잡한 robust 최적화를 반복적인 가중 최소 제곱 문제로 변환합니다.

4. **커널 선택의 중요성**:
   - **Huber**: 안전한 첫 선택, 볼록성 보장
   - **Cauchy**: 강력한 이상치 처리, 적당한 비볼록성
   - **Tukey**: 극단적 이상치 완전 거부

5. **수렴성과 안정성**: 볼록 커널은 수렴 보장, 비볼록 커널은 GNC나 좋은 초기값 필요

### 10.2 실무 체크리스트

✅ 센서별 노이즈 특성 파악  
✅ MAD 기반 robust scale 추정  
✅ 다단계 최적화 전략 적용  
✅ 수렴성 모니터링  
✅ 이상치 비율 추적  

### 10.3 다음 장 예고

Chapter 7에서는 회전 행렬의 특수한 구조를 활용한 **Chordal 초기화**를 배웁니다. 비선형 최적화의 좋은 초기값을 구하는 것이 왜 중요한지, 그리고 회전의 기하학적 성질을 어떻게 활용하는지 살펴볼 것입니다.

**핵심 질문 되돌아보기:**
- ✓ 실제 SLAM의 이상치는 센서와 환경의 본질적 한계
- ✓ M-estimator는 MLE의 robust 일반화
- ✓ IRLS는 가중치 업데이트로 비볼록성 처리
- ✓ 각 kernel은 특정 heavy-tailed 분포에 대응
- ✓ 볼록 커널 + 좋은 초기값 = 수렴성 보장

이제 실습에서 이상치가 포함된 데이터로 각 방법의 효과를 직접 확인해보세요!