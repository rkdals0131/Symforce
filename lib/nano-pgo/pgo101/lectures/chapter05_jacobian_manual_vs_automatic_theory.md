# PGO 101 - Chapter 5 이론 강의: 야코비안 계산 - 수동의 고통 vs. 자동의 희열

**강의 목표:** 이 강의를 마치면, 여러분은 Pose Graph Optimization의 성능과 정확도를 좌우하는 가장 핵심적인 요소인 **야코비안 (Jacobian)** 에 대해 깊이 있게 이해하게 됩니다. 야코비안을 수동으로 유도하고 구현하는 것이 왜 그렇게 어렵고 오류가 발생하기 쉬운지, 그리고 SymForce와 같은 자동 미분 도구가 이 문제를 어떻게 해결하여 우리의 삶을 편하게 만들어주는지 명확히 비교하고 설명할 수 있게 됩니다. 이 강의는 `chapter05_jacobian_manual_vs_automatic.ipynb` 실습에서 두 가지 방식의 차이를 코드로 직접 확인하기 위한 이론적 토대를 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 야코비안이 최적화 알고리즘에서 왜 그렇게 중요한가?
> - 매니폴드 위에서의 미분이 왜 유클리드 공간과 다른가?
> - Storage space와 Tangent space의 차이는 무엇인가?
> - 자동 미분이 어떻게 인간의 실수를 방지하는가?
> - 기호 연산이 수치 연산보다 때로는 더 빠를 수 있는 이유는?

---

## 목차

1. [야코비안의 기초 - 최적화의 열쇠](#1-야코비안의-기초---최적화의-열쇠)
2. [SLAM에서의 야코비안 역할](#2-slam에서의-야코비안-역할)
3. [수동 야코비안 계산의 도전](#3-수동-야코비안-계산의-도전)
4. [리 이론과 야코비안](#4-리-이론과-야코비안)
5. [Storage vs Tangent Space](#5-storage-vs-tangent-space)
6. [자동 미분의 원리](#6-자동-미분의-원리)
7. [SymForce의 기호 연산](#7-symforce의-기호-연산)
8. [수동 vs 자동: 실전 비교](#8-수동-vs-자동-실전-비교)
9. [성능과 정확도 분석](#9-성능과-정확도-분석)
10. [요약 및 최선의 실천법](#10-요약-및-최선의-실천법)

---

## 1. 야코비안의 기초 - 최적화의 열쇠

### 1.1 단변수에서 다변수로

고등학교 미적분에서 배운 미분의 개념을 복습해봅시다. 함수 $f(x)$ 의 미분은:

$$f'(x) = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x}$$

이는 함수의 **순간 변화율**을 나타내며, 작은 입력 변화에 대한 출력 변화를 선형 근사합니다:

$$f(x + \Delta x) \approx f(x) + f'(x) \Delta x$$

### 1.2 다변수 함수와 편미분

이제 입력이 벡터 $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ 이고 출력도 벡터 $\mathbf{f} = [f_1, f_2, ..., f_m]^T$ 인 경우를 생각해봅시다.

**편미분 (Partial Derivative):**

$$\frac{\partial f_i}{\partial x_j} = \lim_{\Delta x_j \to 0} \frac{f_i(x_1, ..., x_j + \Delta x_j, ..., x_n) - f_i(x_1, ..., x_j, ..., x_n)}{\Delta x_j}$$

이는 $x_j$ 만 변할 때 $f_i$ 의 변화율을 나타냅니다.

### 1.3 야코비안 행렬의 정의

**야코비안 행렬 $J$** 는 모든 편미분을 모은 $m \times n$ 행렬입니다:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

### 1.4 선형 근사와 테일러 전개

야코비안을 사용하면 다변수 함수의 1차 테일러 근사가 가능합니다:

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + J(\mathbf{x}) \Delta\mathbf{x}$$

**예시: 2D 회전 변환**
```python
def rotate_2d(x, theta):
    # 입력: x = [x, y], 출력: rotated = [x', y']
    c, s = cos(theta), sin(theta)
    return [c*x[0] - s*x[1], s*x[0] + c*x[1]]

# 야코비안 (x에 대해)
J_x = [[cos(theta), -sin(theta)],
       [sin(theta),  cos(theta)]]

# 야코비안 (theta에 대해)
J_theta = [[-sin(theta)*x[0] - cos(theta)*x[1]],
           [ cos(theta)*x[0] - sin(theta)*x[1]]]
```

---

## 2. SLAM에서의 야코비안 역할

### 2.1 비선형 최적화와 야코비안

SLAM의 비용 함수를 최소화하려면:

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_k \|\mathbf{e}_k(\mathbf{x})\|^2$$

Gauss-Newton 알고리즘은 야코비안을 사용해 업데이트를 계산합니다:

$$\Delta\mathbf{x} = -(J^T J)^{-1} J^T \mathbf{e}$$

여기서 $J$ 는 **모든 에러 항에 대한 모든 변수의 야코비안**입니다.

### 2.2 Between Factor의 야코비안

두 포즈 사이의 상대 변환 에러:

$$\mathbf{e}_{ij} = \log(T_{ij}^{\text{meas}^{-1}} \cdot T_i^{-1} \cdot T_j)$$

이를 $T_i$ 와 $T_j$ 에 대해 미분해야 합니다:

$$J_i = \frac{\partial \mathbf{e}_{ij}}{\partial T_i}, \quad J_j = \frac{\partial \mathbf{e}_{ij}}{\partial T_j}$$

### 2.3 야코비안의 기하학적 의미

야코비안은 **\"작은 포즈 변화가 에러에 미치는 영향\"** 을 나타냅니다:

```
포즈 변화 → 예측 변환 변화 → 에러 변화
Δx        →  Δh(x)         →  Δe = J·Δx
```

**직관적 이해:**
- 큰 야코비안 값: 해당 변수가 에러에 민감함
- 작은 야코비안 값: 해당 변수가 에러에 둔감함
- 0인 야코비안: 해당 변수가 에러에 영향 없음

---

## 3. 수동 야코비안 계산의 도전

### 3.1 복잡한 연쇄 법칙

Between factor의 에러 함수를 다시 보면:

$$\mathbf{e}_{ij} = \log(T_{ij}^{\text{meas}^{-1}} \cdot T_i^{-1} \cdot T_j)$$

이를 미분하려면 여러 단계의 연쇄 법칙이 필요합니다:

1. $\log$ 함수의 미분
2. 행렬 곱의 미분
3. 역행렬의 미분
4. 매니폴드 상의 미분

### 3.2 역행렬의 미분

행렬 $A$ 의 역행렬 $A^{-1}$ 의 미분은:

$$\frac{\partial A^{-1}}{\partial \alpha} = -A^{-1} \frac{\partial A}{\partial \alpha} A^{-1}$$

이를 $SE(3)$ 변환에 적용하면 매우 복잡해집니다.

### 3.3 행렬 곱의 미분

두 행렬의 곱 $C = AB$ 의 미분:

$$\frac{\partial C}{\partial \alpha} = \frac{\partial A}{\partial \alpha} B + A \frac{\partial B}{\partial \alpha}$$

### 3.4 수동 계산의 함정들

1. **부호 오류**: 음수 부호를 빼먹기 쉬움
2. **인덱스 오류**: 행렬 원소의 위치 혼동
3. **좌표계 혼동**: 로컬/글로벌 좌표계 변환
4. **수치 안정성**: 특이점 근처에서의 불안정

**실제 예시 - SE(2)에서의 수동 계산:**
```python
def jacobian_manual_se2(Ti, Tj, z_ij):
    # 1. 상대 변환 계산
    xi, yi, thi = Ti
    xj, yj, thj = Tj
    
    # 2. 회전 행렬과 그 미분
    ci, si = cos(thi), sin(thi)
    dci_dthi, dsi_dthi = -si, ci
    
    # 3. 복잡한 연쇄 법칙 적용
    # ... 수십 줄의 복잡한 수식 ...
    
    # 4. 최종 야코비안 조립
    J_i = # 복잡한 수식
    J_j = # 복잡한 수식
    
    return J_i, J_j
```

---

## 4. 리 이론과 야코비안

### 4.1 매니폴드와 접선 공간

**리 그룹 (Lie Group)** 은 곡면 상의 점들로 이루어진 공간입니다:
- $SO(3)$: 3D 회전의 공간 (3차원 매니폴드)
- $SE(3)$: 3D 자세의 공간 (6차원 매니폴드)

**리 대수 (Lie Algebra)** 는 리 그룹의 접선 공간입니다:
- $\mathfrak{so}(3)$: 3D 각속도 벡터 공간
- $\mathfrak{se}(3)$: 6D 속도 벡터 공간 (선속도 + 각속도)

### 4.2 지수 맵과 로그 맵

리 대수와 리 그룹 사이의 변환:

**지수 맵 (Exponential Map):**
$$T = \exp(\boldsymbol{\xi}^{\wedge})$$

**로그 맵 (Logarithm Map):**
$$\boldsymbol{\xi} = \log(T)^{\vee}$$

여기서 $\wedge$ 와 $\vee$ 는 벡터와 행렬 표현 사이의 변환입니다.

### 4.3 리 그룹 위에서의 미분

매니폴드 위에서의 미분은 특별한 규칙을 따릅니다:

**좌 야코비안 (Left Jacobian):**
$$J_l(\boldsymbol{\xi}) = \frac{\sin(\|\boldsymbol{\xi}\|)}{\|\boldsymbol{\xi}\|} I + \frac{1 - \cos(\|\boldsymbol{\xi}\|)}{\|\boldsymbol{\xi}\|} \boldsymbol{\xi}^{\wedge} + \left(1 - \frac{\sin(\|\boldsymbol{\xi}\|)}{\|\boldsymbol{\xi}\|}\right) \boldsymbol{\xi}\boldsymbol{\xi}^T$$

이는 접선 공간에서의 변화가 매니폴드에서 어떻게 나타나는지를 설명합니다.

### 4.4 Adjoint 표현

리 그룹 원소 $T$ 의 Adjoint 표현:

$$\text{Ad}_T : \mathfrak{g} \to \mathfrak{g}$$

이는 좌표계 변환에서 중요한 역할을 합니다:

$$T \exp(\boldsymbol{\xi}^{\wedge}) T^{-1} = \exp((\text{Ad}_T \boldsymbol{\xi})^{\wedge})$$

---

## 5. Storage vs Tangent Space

### 5.1 과다 매개변수화의 필요성

**Storage Space (저장 공간):**
- 쿼터니언: 4개 파라미터 (단위 제약 조건 포함)
- 회전 행렬: 9개 파라미터 (직교 제약 조건 포함)

**Tangent Space (접선 공간):**
- 회전 벡터: 3개 파라미터 (최소 표현)
- se(3) 벡터: 6개 파라미터 (최소 표현)

### 5.2 왜 두 가지 표현이 필요한가?

**Storage Space의 장점:**
1. **특이점 없음**: 모든 회전을 안정적으로 표현
2. **빠른 합성**: 쿼터니언 곱셈은 효율적
3. **보간 용이**: SLERP 등의 부드러운 보간

**Tangent Space의 장점:**
1. **최소 표현**: 자유도와 일치하는 차원
2. **선형 연산**: 덧셈과 뺄셈이 정의됨
3. **최적화 적합**: 제약 조건 없는 최적화 가능

### 5.3 변환 야코비안

Storage와 Tangent 사이의 변환에도 야코비안이 필요합니다:

$$\frac{\partial \text{storage}}{\partial \text{tangent}}$$

**쿼터니언의 예:**
```python
def quaternion_to_tangent_jacobian(q):
    # q = [qw, qx, qy, qz]
    # 반쿼터니언 공식 사용
    w = q[0]
    v = q[1:4]
    
    if w < 0:  # 표준화
        q = -q
        w = -w
        v = -v
    
    # 야코비안 계산
    if abs(w) < 1e-6:
        # 특이점 근처
        J = 2 * np.eye(3)
    else:
        # 일반적인 경우
        a = 2 / w
        J = a * (np.eye(3) - v.reshape(3,1) @ v.reshape(1,3) / (1 + w))
    
    return J
```

### 5.4 최적화에서의 실제 사용

최적화 업데이트는 Tangent space에서 계산되고 Storage space로 변환됩니다:

```python
# 1. Tangent space에서 업데이트 계산
delta_tangent = solve_normal_equations(H, b)

# 2. Storage space로 변환
for i in range(num_poses):
    # 현재 포즈 (storage)
    T_current = poses[i]
    
    # Tangent 업데이트를 리 그룹 원소로
    delta_T = exp(delta_tangent[i*6:(i+1)*6])
    
    # 포즈 업데이트
    T_new = T_current @ delta_T
    
    # 다시 storage로 저장
    poses[i] = T_new
```

---

## 6. 자동 미분의 원리

### 6.1 계산 그래프

모든 계산은 기본 연산의 조합입니다:

```
x → [×2] → y → [+3] → z → [sin] → w
```

각 노드에서 로컬 미분을 계산하고 연쇄 법칙으로 결합합니다.

### 6.2 Forward Mode vs Reverse Mode

**Forward Mode (전진 모드):**
- 입력에서 출력으로 미분 전파
- 입력 차원이 작을 때 효율적
- 방향 미분 계산

**Reverse Mode (역진 모드):**
- 출력에서 입력으로 미분 전파
- 출력 차원이 작을 때 효율적
- 그래디언트 계산 (SLAM에 적합)

### 6.3 이중수를 이용한 자동 미분

**이중수 (Dual Numbers):**
$$a + b\epsilon, \quad \epsilon^2 = 0$$

이를 사용하면 함수값과 미분을 동시에 계산:

```python
class DualNumber:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual
    
    def __mul__(self, other):
        # (a + b*ε)(c + d*ε) = ac + (ad + bc)ε
        return DualNumber(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )
    
    def sin(self):
        # sin(a + b*ε) = sin(a) + b*cos(a)*ε
        return DualNumber(
            np.sin(self.real),
            self.dual * np.cos(self.real)
        )
```

### 6.4 자동 미분의 정확성

자동 미분은 **기계 정밀도**까지 정확합니다:
- 수치 미분: $O(h)$ 또는 $O(h^2)$ 오차
- 자동 미분: $O(\epsilon_{\text{machine}})$ 오차

---

## 7. SymForce의 기호 연산

### 7.1 기호 표현의 힘

SymForce는 계산을 기호로 표현하고 최적화합니다:

```python
# 기호 변수 정의
x = sf.Symbol('x')
y = sf.Symbol('y')

# 기호 표현식
expr = x**2 + 2*x*y + y**2

# 자동으로 인수분해
simplified = sf.simplify(expr)  # (x + y)^2

# 야코비안 자동 계산
J_x = expr.diff(x)  # 2*x + 2*y
J_y = expr.diff(y)  # 2*x + 2*y
```

### 7.2 공통 부분식 제거

SymForce는 반복 계산을 자동으로 감지하고 제거합니다:

```python
# 원래 코드
a = sin(x) * cos(y)
b = sin(x) * sin(y)
c = sin(x) * (cos(y) + sin(y))

# 최적화된 코드
_tmp1 = sin(x)
a = _tmp1 * cos(y)
b = _tmp1 * sin(y)
c = _tmp1 * (cos(y) + sin(y))
```

### 7.3 코드 생성

SymForce는 최적화된 C++ 코드를 생성합니다:

```cpp
template<typename Scalar>
void ComputeJacobian(
    const Eigen::Matrix<Scalar, 3, 1>& x,
    Eigen::Matrix<Scalar, 2, 3>& J)
{
    // 자동 생성된 최적화 코드
    const Scalar _tmp0 = std::sin(x[2]);
    const Scalar _tmp1 = std::cos(x[2]);
    
    J(0, 0) = _tmp1;
    J(0, 1) = -_tmp0;
    J(0, 2) = -_tmp0 * x[0] - _tmp1 * x[1];
    // ...
}
```

### 7.4 리 그룹 연산의 자동 처리

SymForce는 리 그룹 연산을 자동으로 올바르게 처리합니다:

```python
# 포즈 정의
T_a = sf.Pose3()
T_b = sf.Pose3()

# 상대 변환 (자동으로 SE(3) 연산)
T_ab = T_a.inverse() * T_b

# 에러 (자동으로 tangent space로)
error = T_ab.local_coordinates(T_measured)

# 야코비안 (storage/tangent 변환 자동 처리)
J = error.jacobian([T_a, T_b])
```

---

## 8. 수동 vs 자동: 실전 비교

### 8.1 개발 시간

**수동 구현:**
- 수식 유도: 2-4시간
- 코드 구현: 1-2시간
- 디버깅: 2-8시간
- 총: 5-14시간

**자동 구현:**
- 에러 함수 정의: 30분
- 코드 생성: 즉시
- 검증: 30분
- 총: 1시간

### 8.2 정확도 비교

실습에서 확인하는 내용:

```python
# 수동 야코비안과 자동 야코비안 비교
error_norm = np.linalg.norm(J_manual - J_auto)

# 일반적인 결과:
# - 수동 (근사): 1e-6 ~ 1e-4 오차
# - 수동 (정확): 1e-12 ~ 1e-10 오차 (구현 오류 없을 때)
# - 자동: 1e-15 ~ 1e-14 오차 (기계 정밀도)
```

### 8.3 성능 비교

**실행 시간 (1000회 반복):**
- 수동 (Python): ~10ms
- 자동 (SymForce Python): ~8ms
- 자동 (SymForce C++): ~0.5ms

### 8.4 유지보수성

**수동 코드의 문제:**
```python
# 에러 함수가 바뀌면?
# 예: Huber norm 추가
def error_with_huber(T_i, T_j, z_ij, delta):
    e = compute_error(T_i, T_j, z_ij)
    
    # Huber norm
    if np.linalg.norm(e) < delta:
        return e
    else:
        return delta * e / np.linalg.norm(e)
    
    # 야코비안도 완전히 다시 유도해야 함!
```

**자동 코드의 장점:**
```python
# 에러 함수만 수정
def error_with_huber(T_i, T_j, z_ij, delta):
    e = compute_error(T_i, T_j, z_ij)
    return sf.huber_norm(e, delta)

# 야코비안은 자동으로 재생성!
```

---

## 9. 성능과 정확도 분석

### 9.1 수치 안정성

**수동 구현의 위험 요소:**

1. **Gimbal Lock**: 오일러 각 사용 시
2. **특이점**: $\sin(\theta) \approx 0$ 에서
3. **정규화 누락**: 쿼터니언 drift

**자동 구현의 안정성:**
- 검증된 수치 안정적 공식 사용
- 특이점 자동 처리
- 정규화 자동 수행

### 9.2 계산 복잡도

**Between Factor 야코비안:**
- 수동: $O(1)$ 하지만 상수가 큼
- 자동 (런타임): $O(1)$ 오버헤드 있음
- 자동 (코드 생성): $O(1)$ 최적화됨

### 9.3 메모리 사용

```python
# 메모리 프로파일링 결과
# 1000개 포즈, 5000개 엣지

# 수동 구현:
# - 야코비안 저장: 240 MB
# - 임시 변수: 50 MB

# SymForce:
# - 야코비안 저장: 240 MB
# - 심볼릭 그래프: 10 MB (한 번만)
# - 임시 변수: 최소화됨
```

### 9.4 병렬화 가능성

**자동 생성 코드의 장점:**
- 데이터 의존성 자동 분석
- SIMD 명령어 활용 가능
- GPU 코드 생성 가능

---

## 10. 요약 및 최선의 실천법

### 10.1 핵심 교훈

1. **야코비안은 최적화의 핵심**: 정확한 야코비안 없이는 효율적인 수렴 불가능

2. **수동 구현은 위험**: 복잡한 수식, 높은 오류 가능성, 긴 개발 시간

3. **자동 미분은 안전**: 정확도 보장, 빠른 개발, 쉬운 유지보수

4. **Storage vs Tangent 이해 필수**: 올바른 공간에서의 연산이 중요

5. **기호 연산의 성능**: 때로는 수치 연산보다 빠를 수 있음

### 10.2 실무 가이드라인

**언제 수동 구현을 고려할까?**
- 매우 단순한 함수
- 특수한 최적화가 필요한 경우
- 교육 목적

**언제 자동 미분을 사용할까?**
- 복잡한 함수 (대부분의 SLAM)
- 빠른 프로토타이핑
- 에러 함수가 자주 바뀌는 경우

### 10.3 SymForce 베스트 프랙티스

```python
# 1. 명확한 타입 사용
pose = sf.Pose3()  # 자동으로 SE(3) 연산

# 2. 심볼릭 표현 재사용
with sf.scope("between_factor"):
    error = compute_error(...)
    
# 3. 코드 생성 활용
generated_func = sf.Codegen(error, [T_i, T_j]).generate()

# 4. 수치 검증
assert sf.numerical_derivative(f, x).allclose(
    sf.symbolic_derivative(f, x).evalf()
)
```

### 10.4 다음 장으로

이제 야코비안의 중요성과 자동 계산의 장점을 이해했으니, Chapter 6에서는 실제 SLAM에서 자주 발생하는 **아웃라이어**를 어떻게 다루는지 배우게 됩니다. Cauchy robust kernel이 어떻게 잘못된 측정값에 대한 민감도를 줄이는지, 그리고 이것이 야코비안 계산에 어떤 영향을 미치는지 살펴볼 것입니다.

**핵심 질문 되돌아보기:**
- ✓ 야코비안이 최적화 업데이트 방향을 결정함
- ✓ 매니폴드 미분은 접선 공간을 통해 이루어짐
- ✓ Storage는 안정적 표현, Tangent는 최적화 공간
- ✓ 자동 미분은 인간 실수를 원천 차단
- ✓ 기호 최적화로 반복 계산 제거 가능

이 지식을 바탕으로 실습에서 두 방식의 차이를 직접 체험해보세요!