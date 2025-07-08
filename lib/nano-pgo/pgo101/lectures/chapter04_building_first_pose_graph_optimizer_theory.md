# PGO 101 - Chapter 4 이론 강의: 나만의 첫 Pose Graph Optimizer 제작하기 - 비선형 최적화의 심장부

**강의 목표:** 이 강의를 마치면, 여러분은 Pose Graph Optimization의 핵심 엔진이 어떻게 작동하는지 그 내부 원리를 완벽히 이해하게 됩니다. 비선형 최적화 문제를 푸는 표준적인 접근법인 **Gauss-Newton 알고리즘**과 그 개선 버전인 **Levenberg-Marquardt 알고리즘**의 수학적 기초를 탄탄히 다지고, 이를 구현하기 위해 **Hessian 행렬 (H)** 과 **gradient 벡터 (b)** 를 어떻게 구축하는지, 그리고 왜 **희소 행렬 (Sparse Matrix)** 의 개념이 대규모 SLAM에서 필수적인지를 설명할 수 있게 됩니다. 특히 초기값이 나쁠 때의 수렴 문제와 이를 해결하는 적응적 댐핑 전략까지 실무에 즉시 적용 가능한 지식을 습득합니다. 이 강의는 `chapter04_building_first_pose_graph_optimizer.ipynb` 실습에서 직접 옵티마이저를 코딩하기 위한 모든 이론적 배경을 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 SLAM을 비선형 최소 제곱 문제로 공식화하는가?
> - Gauss-Newton이 Newton 방법보다 SLAM에 더 적합한 이유는?
> - Levenberg-Marquardt의 댐핑 파라미터가 어떻게 안정성을 보장하는가?
> - 희소 행렬이 없다면 대규모 SLAM이 불가능한 이유는?
> - 초기값이 나쁠 때 최적화가 실패하는 수학적 원인은?

---

## 목차

1. [SLAM의 비선형 최소 제곱 공식화](#1-slam의-비선형-최소-제곱-공식화)
2. [Newton 방법에서 Gauss-Newton으로](#2-newton-방법에서-gauss-newton으로)
3. [정규 방정식과 선형 시스템](#3-정규-방정식과-선형-시스템)
4. [Levenberg-Marquardt - 적응적 최적화](#4-levenberg-marquardt---적응적-최적화)
5. [희소 행렬의 마법](#5-희소-행렬의-마법)
6. [선형 대수 솔버의 선택](#6-선형-대수-솔버의-선택)
7. [수치적 안정성과 조건수](#7-수치적-안정성과-조건수)
8. [실전 구현 고려사항](#8-실전-구현-고려사항)
9. [성능 최적화 전략](#9-성능-최적화-전략)
10. [요약 및 다음 장 예고](#10-요약-및-다음-장-예고)

---

## 1. SLAM의 비선형 최소 제곱 공식화

### 1.1 왜 최소 제곱인가?

SLAM (Simultaneous Localization and Mapping)은 본질적으로 **불확실성 하에서의 추정 문제**입니다. 센서는 노이즈를 포함하고, 로봇의 움직임은 부정확하며, 환경은 변할 수 있습니다. 이런 상황에서 우리의 목표는 모든 측정값을 **가장 잘 설명하는** 로봇의 궤적과 지도를 찾는 것입니다.

**비용 함수 (Cost Function):**

$$F(\mathbf{x}) = \frac{1}{2} \sum_{(i,j) \in \mathcal{C}} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j)^T \Omega_{ij} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j)$$

여기서:
- $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n]^T$ : 모든 로봇 포즈의 상태 벡터
- $\mathbf{e}_{ij}$ : 포즈 $i$ 와 $j$ 사이의 **잔차 (residual)**
- $\Omega_{ij} = \Sigma_{ij}^{-1}$ : **정보 행렬 (Information matrix)**, 측정 불확실성의 역

### 1.2 잔차의 의미

잔차는 "예측과 측정의 차이"를 나타냅니다:

$$\mathbf{e}_{ij} = \mathbf{z}_{ij} - \mathbf{h}_{ij}(\mathbf{x}_i, \mathbf{x}_j)$$

- $\mathbf{z}_{ij}$ : 센서가 측정한 상대 변환
- $\mathbf{h}_{ij}$ : 현재 포즈 추정값으로 예측한 상대 변환

**SE(2) 예시:**
```python
# 측정값: 로봇이 1m 직진, 10도 회전
z_ij = [1.0, 0.0, 0.174]  # [x, y, theta in rad]

# 예측값: 현재 포즈로 계산
h_ij = inverse(x_i) * x_j  # 상대 변환

# 잔차
e_ij = z_ij - h_ij
```

### 1.3 왜 비선형인가?

SLAM의 비선형성은 주로 **회전 변환**에서 발생합니다:

1. **회전의 합성**: $R_1 \cdot R_2 \neq R_1 + R_2$
2. **삼각함수**: $\cos(\theta_1 + \theta_2) \neq \cos(\theta_1) + \cos(\theta_2)$
3. **SE(3)의 매니폴드 구조**: 유클리드 공간이 아닌 곡면

**구체적 예시:**
```python
# 비선형 변환
def relative_transform(x_i, x_j):
    # 회전 행렬 (비선형!)
    R_i = [[cos(x_i[2]), -sin(x_i[2])],
           [sin(x_i[2]),  cos(x_i[2])]]
    
    # 상대 위치 (회전이 곱해져서 비선형)
    t_ij = R_i.T @ (x_j[:2] - x_i[:2])
    
    return t_ij
```

### 1.4 국소 최소값의 함정

비선형성의 결과로 비용 함수는 **여러 개의 최소값**을 가질 수 있습니다:

<div style="text-align: center;">
<pre>
비용 함수의 지형
     ^
 F(x)|     *     
     |    / \    *
     |   /   \  / \
     |  /     \/   \
     | /            \
     +----------------> x
      국소    전역
      최소    최소
</pre>
</div>

> 🎯 **핵심 통찰**: 좋은 초기값이 없으면 알고리즘이 국소 최소값에 갇힐 수 있습니다!

---

## 2. Newton 방법에서 Gauss-Newton으로

### 2.1 Newton 방법의 기본 아이디어

Newton 방법은 함수를 2차 테일러 급수로 근사합니다:

$$F(\mathbf{x} + \Delta\mathbf{x}) \approx F(\mathbf{x}) + \mathbf{g}^T \Delta\mathbf{x} + \frac{1}{2} \Delta\mathbf{x}^T \mathbf{H} \Delta\mathbf{x}$$

여기서:
- $\mathbf{g} = \nabla F(\mathbf{x})$ : Gradient
- $\mathbf{H} = \nabla^2 F(\mathbf{x})$ : Hessian

최소값은 미분이 0인 점에서 발생:
$$\nabla_{\Delta\mathbf{x}} F(\mathbf{x} + \Delta\mathbf{x}) = \mathbf{g} + \mathbf{H} \Delta\mathbf{x} = 0$$

따라서 **Newton 업데이트**:
$$\Delta\mathbf{x} = -\mathbf{H}^{-1} \mathbf{g}$$

### 2.2 최소 제곱 문제의 특별한 구조

SLAM의 비용 함수는 특별한 구조를 가집니다:

$$F(\mathbf{x}) = \frac{1}{2} \mathbf{e}(\mathbf{x})^T \Omega \mathbf{e}(\mathbf{x})$$

이때 gradient와 Hessian은:
- **Gradient**: $\mathbf{g} = \mathbf{J}^T \Omega \mathbf{e}$
- **Hessian**: $\mathbf{H} = \mathbf{J}^T \Omega \mathbf{J} + \sum_k \mathbf{e}_k^T \Omega_k \nabla^2 \mathbf{e}_k$

여기서 $\mathbf{J} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}}$ 는 잔차의 Jacobian입니다.

### 2.3 Gauss-Newton 근사

**핵심 아이디어**: Hessian의 2차 항을 무시합니다!

$$\mathbf{H} \approx \mathbf{J}^T \Omega \mathbf{J}$$

**왜 이 근사가 타당한가?**

1. **해 근처에서**: $\mathbf{e} \approx 0$ 이므로 2차 항이 작음
2. **계산 효율성**: 2차 미분을 계산할 필요 없음
3. **양의 준정부호성**: $\mathbf{J}^T \Omega \mathbf{J}$ 는 항상 PSD

### 2.4 Gauss-Newton vs Newton 비교

| 특성 | Newton 방법 | Gauss-Newton |
|------|------------|--------------|
| Hessian 계산 | 완전한 2차 미분 필요 | 1차 미분만 필요 |
| 계산 복잡도 | $O(n^2)$ Hessian 원소 | $O(nm)$ Jacobian 원소 |
| 수렴 속도 | 2차 수렴 | 거의 2차 수렴 |
| 안정성 | Hessian이 비정부호일 수 있음 | 항상 하강 방향 |
| 메모리 | 더 많음 | 더 적음 |

**수렴 분석:**

Newton 방법의 수렴:
$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C \|\mathbf{x}_k - \mathbf{x}^*\|^2$$

Gauss-Newton의 수렴:
$$\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq C_1 \|\mathbf{x}_k - \mathbf{x}^*\|^2 + C_2 \|\mathbf{e}(\mathbf{x}^*)\|$$

> 💡 **핵심**: Gauss-Newton은 잔차가 0이 아닐 때 약간의 편향을 가지지만, 실용적으로는 충분히 빠릅니다!

---

## 3. 정규 방정식과 선형 시스템

### 3.1 정규 방정식의 유도

각 반복에서 우리는 선형화된 문제를 풉니다:

$$\min_{\Delta\mathbf{x}} \|\mathbf{e} + \mathbf{J}\Delta\mathbf{x}\|^2_\Omega$$

이를 전개하면:
$$(\mathbf{e} + \mathbf{J}\Delta\mathbf{x})^T \Omega (\mathbf{e} + \mathbf{J}\Delta\mathbf{x})$$

미분하여 0으로 놓으면:
$$\mathbf{J}^T \Omega \mathbf{J} \Delta\mathbf{x} = -\mathbf{J}^T \Omega \mathbf{e}$$

이것이 **정규 방정식 (Normal Equations)** 입니다:
$$\mathbf{H} \Delta\mathbf{x} = -\mathbf{b}$$

### 3.2 블록 구조의 활용

SLAM에서 각 제약은 몇 개의 변수에만 영향을 줍니다:

```python
# 엣지 (i,j)가 H와 b에 기여하는 부분
def add_edge_contribution(H, b, i, j, e_ij, J_i, J_j, Omega):
    # H의 블록 업데이트
    H[i,i] += J_i.T @ Omega @ J_i
    H[j,j] += J_j.T @ Omega @ J_j
    H[i,j] += J_i.T @ Omega @ J_j
    H[j,i] += J_j.T @ Omega @ J_i  # 대칭성
    
    # b의 블록 업데이트
    b[i] += J_i.T @ Omega @ e_ij
    b[j] += J_j.T @ Omega @ e_ij
```

### 3.3 정보 행렬의 역할

정보 행렬 $\Omega = \Sigma^{-1}$ 은 측정의 신뢰도를 나타냅니다:

**고신뢰도 측정 (Odometry):**
$$\Omega_{\text{odom}} = \begin{bmatrix}
100 & 0 & 0 \\
0 & 100 & 0 \\
0 & 0 & 100
\end{bmatrix}$$

**저신뢰도 측정 (Loop Closure):**
$$\Omega_{\text{loop}} = \begin{bmatrix}
10 & 0 & 0 \\
0 & 10 & 0 \\
0 & 0 & 10
\end{bmatrix}$$

> 🎯 **실용적 팁**: 정보 행렬의 대각 원소는 각 차원의 정밀도를 나타냅니다. 회전이 위치보다 정확하다면 회전에 더 큰 값을 설정하세요.

---

## 4. Levenberg-Marquardt - 적응적 최적화

### 4.1 Trust Region 해석

Levenberg-Marquardt는 **trust region** 방법으로 이해할 수 있습니다:

$$\min_{\Delta\mathbf{x}} \|\mathbf{e} + \mathbf{J}\Delta\mathbf{x}\|^2 \quad \text{subject to} \quad \|\Delta\mathbf{x}\| \leq \delta$$

라그랑주 승수를 사용하면:
$$(\mathbf{J}^T \Omega \mathbf{J} + \lambda \mathbf{I}) \Delta\mathbf{x} = -\mathbf{J}^T \Omega \mathbf{e}$$

여기서 $\lambda$ 는 trust region 크기와 반비례합니다.

### 4.2 댐핑 파라미터의 기하학적 의미

<div style="text-align: center;">
<pre>
λ가 작을 때 (Gauss-Newton)     λ가 클 때 (Gradient Descent)
     
    등고선                        등고선
   /     \                      /     \
  |   •——→|  큰 스텝            | •→   |  작은 스텝
   \     /                      \     /
</pre>
</div>

### 4.3 적응적 댐핑 전략

**Marquardt의 전략:**
```python
def update_lambda(F_new, F_old, lambda_current):
    if F_new < F_old:  # 개선됨
        lambda_new = lambda_current / 10
        accept_step = True
    else:  # 악화됨
        lambda_new = lambda_current * 10
        accept_step = False
    return lambda_new, accept_step
```

**Nielsen의 전략 (gain ratio):**
```python
def nielsen_update(actual_reduction, predicted_reduction, lambda_current):
    rho = actual_reduction / predicted_reduction
    
    if rho > 0.75:
        lambda_new = lambda_current / 3
    elif rho < 0.25:
        lambda_new = lambda_current * 2
    else:
        lambda_new = lambda_current
        
    accept_step = (rho > 0)
    return lambda_new, accept_step
```

### 4.4 LM의 수학적 해석

댐핑이 알고리즘을 어떻게 변화시키는지 보겠습니다:

**고유값 분해를 통한 분석:**

$\mathbf{H} = \mathbf{V} \Lambda \mathbf{V}^T$ 라고 하면:

- Gauss-Newton: $\Delta\mathbf{x} = -\mathbf{V} \Lambda^{-1} \mathbf{V}^T \mathbf{b}$
- Levenberg-Marquardt: $\Delta\mathbf{x} = -\mathbf{V} (\Lambda + \lambda \mathbf{I})^{-1} \mathbf{V}^T \mathbf{b}$

각 고유 방향에서:
$$\Delta x_i = -\frac{v_i^T \mathbf{b}}{\lambda_i + \lambda}$$

- 큰 고유값 방향 ($\lambda_i \gg \lambda$): 거의 영향 없음
- 작은 고유값 방향 ($\lambda_i \ll \lambda$): 크게 감쇠됨

> 💡 **핵심 통찰**: LM은 불확실한 방향(작은 고유값)의 스텝을 제한합니다!

---

## 5. 희소 행렬의 마법

### 5.1 왜 H는 희소한가?

SLAM 그래프의 연결성이 H의 희소성을 결정합니다:

```
포즈 그래프:          H 행렬 구조:
                      
1 --- 2               [* * . .]
|     |               [* * * .]
|     |               [. * * *]
4 --- 3               [. . * *]

(연결된 포즈만 H에서 비영 블록을 만듦)
```

**수치적 예시:**
- 1000개 포즈, 각 포즈당 평균 3개 연결
- Dense H: 6000 × 6000 = 36,000,000 원소
- Sparse H: ~36,000 비영 원소 (0.1%)
- **메모리 절약**: 1000배!

### 5.2 희소 행렬 저장 형식

**COO (Coordinate) 형식:**
```python
# (row, col, value) 삼중항
H_coo = [
    (0, 0, 5.2),
    (0, 1, 1.3),
    (1, 1, 4.1),
    ...
]
```

**CSR (Compressed Sparse Row) 형식:**
```python
# 행별로 압축 저장
values = [5.2, 1.3, 4.1, ...]      # 비영 값들
col_indices = [0, 1, 1, ...]       # 열 인덱스
row_pointers = [0, 2, 3, ...]      # 각 행의 시작 위치
```

**성능 비교:**

| 연산 | Dense | COO | CSR |
|------|-------|-----|-----|
| 구축 | O(n²) | O(nnz) | O(nnz) |
| 행렬-벡터 곱 | O(n²) | O(nnz) | O(nnz) |
| 원소 접근 | O(1) | O(nnz) | O(log k) |
| 메모리 | n² | 3×nnz | 2×nnz+n |

### 5.3 Fill-in 현상과 순서 변경

Cholesky 분해 중 0이 비영이 되는 **fill-in** 현상:

```
원래 패턴:        분해 후:
[* * . .]        [* * ● ●]
[* * * .]   →    [* * * ●]
[. * * *]        [● * * *]
[. . * *]        [● ● * *]

(●는 fill-in)
```

**AMD (Approximate Minimum Degree) 순서 변경:**
```python
# 순서 변경으로 fill-in 최소화
perm = amd_ordering(H)
H_reordered = H[perm, :][:, perm]
# 이제 H_reordered는 적은 fill-in을 가짐
```

### 5.4 실제 SLAM에서의 희소성 패턴

**순차적 SLAM (Odometry만):**
```
H 패턴:
[■ ■ . . . .]
[■ ■ ■ . . .]
[. ■ ■ ■ . .]
[. . ■ ■ ■ .]
[. . . ■ ■ ■]
[. . . . ■ ■]

(띠 대각 구조)
```

**Loop Closure가 있는 SLAM:**
```
H 패턴:
[■ ■ . . . ●]
[■ ■ ■ . . .]
[. ■ ■ ■ . .]
[. . ■ ■ ■ .]
[. . . ■ ■ ■]
[● . . . ■ ■]

(●는 loop closure로 인한 연결)
```

---

## 6. 선형 대수 솔버의 선택

### 6.1 Cholesky 분해

대칭 양의 정부호 행렬에 최적:

$$\mathbf{H} = \mathbf{L} \mathbf{L}^T$$

**장점:**
- 가장 빠름 (n³/3 flops)
- 메모리 효율적 (L만 저장)
- 희소성 보존 가능

**단점:**
- H가 양의 정부호여야 함
- 수치적으로 덜 안정

**구현:**
```python
from scipy.sparse.linalg import splu
from sksparse.cholmod import cholesky

# Sparse Cholesky
factor = cholesky(H_sparse)
x = factor(b)
```

### 6.2 QR 분해

더 안정적이지만 느림:

$$\mathbf{J} = \mathbf{Q} \mathbf{R}$$

정규 방정식을 거치지 않고 직접:
$$\mathbf{R} \Delta\mathbf{x} = -\mathbf{Q}^T \mathbf{e}$$

**장점:**
- 수치적으로 더 안정
- 조건수가 √κ(H) 대신 κ(J)
- Rank-deficient 경우 처리 가능

**단점:**
- 더 느림 (2n³/3 flops)
- 더 많은 메모리 필요

### 6.3 반복적 솔버

대규모 문제에 적합:

**Conjugate Gradient (CG):**
```python
from scipy.sparse.linalg import cg

# 전처리기 사용
M = create_preconditioner(H)  # e.g., Jacobi, SSOR
x, info = cg(H, b, M=M, tol=1e-6)
```

**수렴 속도:**
$$\|\mathbf{x}_k - \mathbf{x}^*\| \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|$$

### 6.4 솔버 선택 가이드

| 문제 특성 | 추천 솔버 |
|----------|----------|
| 작은 dense 문제 | Cholesky |
| 중간 크기, 잘 조건화됨 | Sparse Cholesky |
| 대규모, 매우 희소 | Conjugate Gradient |
| 조건수 나쁨 | QR 또는 SVD |
| Rank deficient | SVD |
| 실시간 요구사항 | Incremental (iSAM2) |

---

## 7. 수치적 안정성과 조건수

### 7.1 조건수의 정의와 의미

조건수는 행렬의 "민감도"를 나타냅니다:

$$\kappa(\mathbf{H}) = \|\mathbf{H}\| \|\mathbf{H}^{-1}\| = \frac{\lambda_{\max}}{\lambda_{\min}}$$

**의미:**
- $\kappa \approx 1$: 잘 조건화됨
- $\kappa \approx 10^6$: 경계선
- $\kappa \approx 10^{12}$: 심각하게 ill-conditioned

### 7.2 SLAM에서 조건수가 나빠지는 경우

1. **긴 궤적, Loop closure 없음**
   ```
   시작 ← · · · · · · · → 끝
   (불확실성이 누적됨)
   ```

2. **평면 운동만 있는 3D SLAM**
   - Z축 정보 부족
   - Roll, Pitch 관측 불가

3. **센서 정렬 문제**
   - 모든 특징점이 한 직선상에

### 7.3 조건수 개선 전략

**1. Gauge 제약 추가:**
```python
# 첫 포즈 고정
H[0:6, 0:6] += 1e10 * np.eye(6)
```

**2. 전처리기 (Preconditioner):**
```python
# Jacobi 전처리
D = np.diag(np.diag(H))
H_precond = D^(-1/2) @ H @ D^(-1/2)
```

**3. 정규화 (Regularization):**
```python
# Tikhonov 정규화
H_reg = H + alpha * np.eye(n)
```

### 7.4 수치 정밀도 손실 감지

```python
def check_numerical_health(H, b, x):
    # 잔차 계산
    residual = H @ x - b
    
    # 상대 오차
    relative_error = np.linalg.norm(residual) / np.linalg.norm(b)
    
    # 조건수 추정
    condition = np.linalg.cond(H)
    
    # 예상 정밀도 손실
    digits_lost = np.log10(condition)
    
    print(f"조건수: {condition:.2e}")
    print(f"예상 정밀도 손실: {digits_lost:.1f} 자리")
```

---

## 8. 실전 구현 고려사항

### 8.1 메모리 관리

**희소 행렬 구축 최적화:**
```python
# 나쁜 예: 반복적 재할당
H = sparse.csr_matrix((n, n))
for edge in edges:
    H[i, j] += value  # 매번 재구조화!

# 좋은 예: triplet으로 모은 후 한번에
rows, cols, values = [], [], []
for edge in edges:
    rows.append(i)
    cols.append(j)
    values.append(value)
H = sparse.coo_matrix((values, (rows, cols))).tocsr()
```

### 8.2 병렬화 전략

**1. Jacobian 계산 병렬화:**
```python
from multiprocessing import Pool

def compute_edge_contribution(edge):
    e, J_i, J_j = compute_residual_and_jacobian(edge)
    return edge.i, edge.j, J_i.T @ Omega @ J_i, J_i.T @ Omega @ e

with Pool() as pool:
    contributions = pool.map(compute_edge_contribution, edges)
```

**2. 행렬 조립 병렬화:**
```python
# OpenMP 스타일 (C++)
#pragma omp parallel for
for (int k = 0; k < edges.size(); ++k) {
    // 각 스레드가 독립적인 edge 처리
    // lock-free accumulation 사용
}
```

### 8.3 수렴 판정

**복합 수렴 조건:**
```python
def check_convergence(dx, g, F_old, F_new, iteration):
    # 1. Gradient norm
    gradient_converged = np.linalg.norm(g) < 1e-4
    
    # 2. Update norm
    update_converged = np.linalg.norm(dx) < 1e-6
    
    # 3. Relative cost reduction
    relative_reduction = abs(F_old - F_new) / F_old
    cost_converged = relative_reduction < 1e-8
    
    # 4. Maximum iterations
    max_iter_reached = iteration >= max_iterations
    
    return (gradient_converged or update_converged or 
            cost_converged or max_iter_reached)
```

### 8.4 Robust 비용 함수

Outlier에 강건한 최적화:

**Huber 손실:**
$$\rho_\text{Huber}(e) = \begin{cases}
\frac{1}{2}e^2 & \text{if } |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}$$

**구현:**
```python
def huber_weight(residual, delta=1.0):
    norm = np.linalg.norm(residual)
    if norm <= delta:
        return 1.0
    else:
        return delta / norm

# 가중치 적용
w = huber_weight(e)
H += w * J.T @ Omega @ J
b += w * J.T @ Omega @ e
```

---

## 9. 성능 최적화 전략

### 9.1 계산 복잡도 분석

| 연산 | Dense | Sparse |
|------|-------|---------|
| H 구축 | O(n²m) | O(nm) |
| Cholesky | O(n³) | O(n^{3/2}) |
| Forward/Back solve | O(n²) | O(n) |
| 총 메모리 | O(n²) | O(n) |

여기서 n = 변수 개수, m = 제약 개수

### 9.2 프로파일링과 병목 현상

```python
import cProfile

def profile_optimization():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 최적화 실행
    optimizer.optimize()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

**전형적인 병목 지점:**
1. Jacobian 계산 (30-40%)
2. 행렬 조립 (20-30%)
3. 선형 시스템 해결 (30-40%)
4. 기타 (10%)

### 9.3 캐시 효율성

**메모리 접근 패턴 최적화:**
```python
# 나쁜 예: column-major 접근 in row-major 저장
for j in range(n):
    for i in range(n):
        process(H[i, j])  # 캐시 미스!

# 좋은 예: row-major 접근
for i in range(n):
    for j in range(n):
        process(H[i, j])  # 캐시 친화적
```

### 9.4 Incremental vs Batch

**Batch 최적화:**
- 모든 데이터를 한번에 처리
- 최적의 정확도
- 계산 비용 높음

**Incremental 최적화 (iSAM2):**
- 새 측정값만 처리
- Bayes tree 구조 활용
- 실시간 가능

```python
# Incremental 업데이트 의사코드
def incremental_update(new_measurements):
    # 1. 영향받는 변수 식별
    affected = find_affected_variables(new_measurements)
    
    # 2. 부분 선형화
    delta_H, delta_b = linearize_new(new_measurements)
    
    # 3. Bayes tree 업데이트
    bayes_tree.update(affected, delta_H, delta_b)
    
    # 4. 부분 해결
    dx = bayes_tree.solve(affected)
```

---

## 10. 요약 및 다음 장 예고

### 10.1 핵심 내용 정리

이 장에서 배운 내용:

1. **비선형 최소 제곱의 본질**
   - SLAM이 왜 이 형태인지
   - 비선형성의 원인과 결과
   - 국소 최소값의 위험

2. **Gauss-Newton의 우아함**
   - Newton 방법의 단순화
   - 최소 제곱 구조 활용
   - 계산 효율성과 안정성

3. **Levenberg-Marquardt의 지혜**
   - Trust region 개념
   - 적응적 댐핑 전략
   - GN과 GD의 장점 결합

4. **희소성의 힘**
   - 그래프 구조와 행렬 희소성
   - 1000배 성능 향상 가능
   - Fill-in 최소화 전략

5. **수치적 강건성**
   - 조건수와 안정성
   - 적절한 솔버 선택
   - 정밀도 손실 방지

### 10.2 실습 체크리스트

`chapter04` 노트북에서 반드시 실습해야 할 내용:

- [ ] SimplePoseGraphOptimizer 클래스 구현
- [ ] Residual과 Jacobian 계산 구현
- [ ] H 행렬과 b 벡터 조립
- [ ] Gauss-Newton 반복 구현
- [ ] Levenberg-Marquardt 댐핑 추가
- [ ] 희소 행렬 시각화와 분석
- [ ] 수렴 분석과 성능 비교

### 10.3 다음 장 예고

**Chapter 5: 야코비안 - 수동 계산 vs 자동 미분**

다음 장에서는:
- 복잡한 변환의 야코비안 유도
- 수치 미분의 함정
- SymForce를 이용한 자동 미분
- 정확성과 효율성 비교

### 10.4 추가 학습 자료

**논문:**
- Nocedal & Wright, "Numerical Optimization" (2006)
- Dellaert & Kaess, "Factor Graphs for Robot Perception" (2017)
- Kummerle et al., "g2o: A General Framework for Graph Optimization" (2011)

**구현 참고:**
- g2o: https://github.com/RainerKuemmerle/g2o
- Ceres Solver: http://ceres-solver.org
- GTSAM: https://gtsam.org

### 10.5 마지막 조언

> "최적화는 예술이자 과학입니다. 수학적 엄밀함도 중요하지만, 실제 문제에서는 엔지니어링 직관이 똑같이 중요합니다. 작은 예제부터 시작해서 점진적으로 복잡도를 높여가세요. 그리고 항상 시각화하세요 - 숫자보다 그림이 더 많은 것을 알려줍니다!"

이제 여러분은 Pose Graph Optimizer의 심장부를 이해했습니다. 실습을 통해 이론을 코드로 구현하며 진정한 이해를 완성하세요!