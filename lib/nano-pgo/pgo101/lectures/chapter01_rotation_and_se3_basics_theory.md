# PGO 101 - Chapter 1 이론 강의: 3D 공간의 언어, 회전과 변환

**강의 목표:** 이 강의를 마치면, 여러분은 3차원 공간에서 로봇의 움직임과 자세를 수학적으로 어떻게 표현하고 계산하는지 깊이 있게 이해하게 됩니다. 특히, Pose Graph Optimization의 가장 기본이 되는 **회전 (Rotation)** 과 **자세 (Pose)** 의 다양한 표현법을 배우고, 이들이 왜 중요한지, 그리고 어떤 상황에 어떤 방법을 사용해야 하는지 설명할 수 있게 됩니다. 나아가 **Lie 군론** 의 관점에서 이러한 표현들을 통합적으로 이해하고, 수치적 안정성을 고려한 실제 구현 방법까지 습득하게 됩니다. 이 강의는 `chapter01_rotation_and_se3_basics.ipynb` 실습을 위한 탄탄한 이론적 기반을 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 로봇의 회전을 표현하는 방법이 여러 가지일까요?
> - 짐벌락(Gimbal Lock)은 왜 발생하며, 어떻게 피할 수 있을까요?
> - SLAM 최적화에서 왜 매니폴드(Manifold) 개념이 필요할까요?
> - 수치적으로 안정적인 회전 계산은 어떻게 구현할까요?

---

## 목차

1. [들어가기 앞서: 3D 공간과 변환의 기초](#1-들어가기-앞서-3d-공간과-변환의-기초)
2. [회전 표현법: SO(3) 그룹](#2-회전-rotation-표현법-so3-그룹)
3. [자세 표현법: SE(3) 변환](#3-자세-pose-표현법-se3-변환)
4. [Lie 군과 Lie 대수: 통합적 이해](#4-lie-군과-lie-대수-통합적-이해)
5. [최적화의 시작: 매니폴드와 접선 공간](#5-최적화의-시작-매니폴드와-접선-공간)
6. [수치적 안정성과 구현](#6-수치적-안정성과-구현)
7. [SLAM에서의 실제 응용](#7-slam에서의-실제-응용)
8. [요약 및 다음 장 예고](#8-요약-및-다음-장-예고)

---

## 1. 들어가기 앞서: 3D 공간과 변환의 기초

본격적인 회전과 변환을 배우기 전에, 기본이 되는 몇 가지 선수 지식을 체계적으로 짚고 넘어가겠습니다.

### 1.1 기초 선형대수학

*   **좌표계 (Coordinate System)**: 우리가 이야기하는 모든 위치와 방향은 '기준'이 되는 좌표계 안에서 정의됩니다. 로보틱스에서는 주로 **오른손 좌표계**를 사용합니다. (오른손의 엄지, 검지, 중지를 각각 X, Y, Z 축으로 생각하면 됩니다.) 월드 (전역) 좌표계, 로봇 (바디) 좌표계, 센서 좌표계 등 다양한 좌표계가 존재하며, 이들 사이의 관계를 정의하는 것이 변환의 시작입니다.

*   **벡터 (Vector)**: 벡터는 크기와 방향을 가진 양입니다. 3D 공간에서 벡터 $\mathbf{v} \in \mathbb{R}^3$ 는 주로 특정 좌표계의 원점으로부터의 '위치'를 나타내거나, 특정 방향으로의 '이동' 또는 '방향' 자체를 나타내는 데 사용됩니다.

*   **선형 변환 (Linear Transformation)**: 하나의 벡터를 다른 벡터로 매핑하는 함수 $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 로, 다음 두 조건을 만족합니다:
    - **덧셈 보존**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
    - **스칼라 곱 보존**: $T(c\mathbf{v}) = cT(\mathbf{v})$
    
    3D 공간에서의 회전, 크기 조절 (scaling) 등은 모두 선형 변환에 속하며, 이는 **행렬 (Matrix)** $A \in \mathbb{R}^{m \times n}$ 로 표현될 수 있습니다: $T(\mathbf{v}) = A\mathbf{v}$

### 1.2 그룹 이론의 기초

회전과 변환을 이해하기 위해서는 **그룹 (Group)** 의 개념이 필수적입니다.

*   **그룹의 정의**: 집합 $G$ 와 이항연산 $\cdot$ 이 다음 네 가지 조건을 만족할 때, $(G, \cdot)$ 를 그룹이라고 합니다:
    1. **닫힘성 (Closure)**: $\forall a, b \in G: a \cdot b \in G$
    2. **결합법칙 (Associativity)**: $\forall a, b, c \in G: (a \cdot b) \cdot c = a \cdot (b \cdot c)$
    3. **항등원 (Identity)**: $\exists e \in G, \forall a \in G: e \cdot a = a \cdot e = a$
    4. **역원 (Inverse)**: $\forall a \in G, \exists a^{-1} \in G: a \cdot a^{-1} = a^{-1} \cdot a = e$

*   **회전과 변환에서의 의미**: 로봇의 회전과 위치 변환은 그룹 구조를 가집니다. 예를 들어:
    - 두 회전을 연속으로 적용하면 또 다른 회전이 됩니다 (닫힘성)
    - 회전하지 않는 것도 회전입니다 (항등원)
    - 모든 회전에는 반대 회전이 존재합니다 (역원)

## 2. 회전 (Rotation) 표현법: $SO(3)$ 그룹

로봇의 '방향' 또는 '자세'를 표현하는 방법은 여러 가지가 있으며, 각각 장단점이 뚜렷합니다. 수학적으로, 3D 공간에서의 회전은 **특수 직교 그룹 (Special Orthogonal Group)**, 즉 $SO(3)$ 에 속하는 원소로 표현됩니다.

> 📦 **$SO(3)$ 의 기하학적 의미**: $SO(3)$ 는 3차원 공간에서 원점을 고정한 회전들의 집합입니다. 이는 3개의 **자유도 (Degree of Freedom, DoF)** 를 가지는 3차원 매니폴드입니다.

### 2.1. 회전 행렬 (Rotation Matrix)

**정의**: 회전 행렬은 $SO(3)$ 그룹의 가장 대표적인 표현법입니다. 3x3 크기의 행렬 $R$ 로, 다음 두 가지 핵심 조건을 만족합니다.

1.  **직교성 (Orthogonality)**: $R^T R = R R^T = I$ (여기서 $I$ 는 단위 행렬입니다). 이는 행렬의 행과 열 벡터들이 서로 직교하며 크기가 1임을 의미합니다.
2.  **양의 행렬식 (Positive Determinant)**: $\det(R) = +1$ . 이 조건은 변환이 반사 (reflection) 를 포함하지 않는 순수한 회전임을 보장합니다.

$$
R \in SO(3) \equiv \{ R \in \mathbb{R}^{3 \times 3} \mid R^T R = I, \det(R) = 1 \}
$$

-   **직관적 의미**: 월드 좌표계의 기저 벡터 (basis vectors) $( \mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z )$ 가 회전 후 로봇의 좌표계에서 어떻게 보이는지를 나타냅니다. 즉, 행렬 $R$ 의 각 열은 변환된 기저 벡터가 됩니다:
    $$R = [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{r}_3]$$
    여기서 $\mathbf{r}_i$ 는 회전된 $i$ 번째 좌표축입니다.

-   **장점**:
    -   개념적으로 매우 직관적입니다.
    -   벡터에 회전을 적용하는 것이 간단한 행렬-벡터 곱으로 계산됩니다: $\mathbf{v}_{\text{rotated}} = R \cdot \mathbf{v}$ .
    -   여러 회전을 합성하는 것이 행렬 곱으로 간단하게 표현됩니다: $R_{\text{total}} = R_2 \cdot R_1$ .
    -   특이점(singularity)이 없습니다.

-   **단점**:
    -   **과다 표현 (Over-parameterization)**: 3개의 자유도를 표현하기 위해 9개의 숫자를 사용하며, 6개의 제약 조건이 따릅니다. 이는 메모리 낭비와 계산 비효율을 야기합니다.
    -   **수치적 불안정성**: 반복적인 계산 과정에서 부동소수점 오차가 누적되면, 행렬이 직교성 ($R^T R = I$) 을 미세하게 잃어버릴 수 있습니다. 이를 방지하기 위해 주기적인 **정규화 (Normalization)** 과정이 필요합니다.

**직교 정규화 (Orthonormalization)**: Gram-Schmidt 과정을 사용하여 수치 오차로 인해 직교성을 잃은 행렬을 복구할 수 있습니다.

### 2.2. 회전 벡터 (Rotation Vector / Axis-Angle)

**정의**: 3차원 벡터 $\mathbf{r} \in \mathbb{R}^3$ 하나로 회전을 표현하는 가장 간결한 방법입니다.
-   벡터의 **방향** ($\hat{\mathbf{r}} = \frac{\mathbf{r}}{||\mathbf{r}||}$) 이 회전축이 됩니다.
-   벡터의 **크기** ($\theta = ||\mathbf{r}||$) 가 해당 축을 중심으로 회전하는 각도 (라디안) 가 됩니다.

$$\mathbf{r} = \theta \cdot \hat{\mathbf{n}}$$

여기서 $\hat{\mathbf{n}}$ 은 단위 회전축, $\theta$ 는 회전 각도입니다.

-   **장점**:
    -   **최소 표현 (Minimal Representation)**: 3개의 자유도를 정확히 3개의 숫자로 표현하여 메모리 효율이 매우 높습니다.
    -   최적화 문제에서 변수로 사용하기에 가장 적합합니다. 그 이유는 이것이 바로 아래에서 배울 **접선 공간 (Tangent Space)** 의 벡터와 직접적으로 대응되기 때문입니다.
    -   작은 회전에 대해서는 거의 선형적으로 동작합니다.

-   **단점**:
    -   여러 회전을 합성하는 연산이 행렬 곱처럼 간단하지 않고, **로드리게스 회전 공식 (Rodrigues' Rotation Formula)** 과 같은 복잡한 계산이 필요합니다.
    -   $\theta = 0$ 또는 $\theta = \pi$ 근처에서 특이점 (singularity) 이 존재할 수 있습니다.
    -   $\theta > \pi$ 인 경우 표현이 유일하지 않습니다 (주기성).

### 2.3. 쿼터니언 (Quaternion)

**정의**: 쿼터니언은 4개의 실수로 구성된, 복소수를 확장한 개념입니다. 보통 하나의 스칼라 성분 ($q_w$) 과 3개의 벡터 성분 ($\mathbf{q}_v = [q_x, q_y, q_z]$) 으로 표현됩니다.

$$
q = q_w + q_x i + q_y j + q_z k = (q_w, \mathbf{q}_v)
$$

여기서 $i, j, k$ 는 쿼터니언 기저로, 다음 관계를 만족합니다:
$$i^2 = j^2 = k^2 = ijk = -1$$

회전을 표현할 때는 크기가 1인 **단위 쿼터니언 (Unit Quaternion)** 을 사용합니다: 
$$||q||^2 = q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$$

-   **회전 벡터와의 관계**: 회전축이 $\mathbf{u}$ , 회전 각도가 $\theta$ 일 때, 해당하는 쿼터니언은 다음과 같습니다.
    $$
    q = \left( \cos\left(\frac{\theta}{2}\right), \sin\left(\frac{\theta}{2}\right)\mathbf{u} \right) = \left[ \cos\left(\frac{\theta}{2}\right), \sin\left(\frac{\theta}{2}\right)u_x, \sin\left(\frac{\theta}{2}\right)u_y, \sin\left(\frac{\theta}{2}\right)u_z \right]
    $$

-   **장점**:
    -   **짐벌락이 없습니다.** 이는 3D 그래픽스, 항공 우주, 로보틱스에서 표준으로 사용되는 가장 큰 이유입니다.
    -   회전 행렬보다 메모리 효율적입니다 (4개 숫자).
    -   두 회전 사이를 부드럽게 보간 (Interpolation) 하는 **SLERP (Spherical Linear Interpolation)** 연산이 가능하여 애니메이션이나 경로 생성에 유리합니다.
    -   회전의 합성이 쿼터니언 곱으로 효율적으로 계산됩니다.

-   **단점**:
    -   4개의 숫자로 표현되므로 직관적인 이해가 가장 어렵습니다.
    -   단위 쿼터니언 제약 ($||q|| = 1$) 을 유지해야 합니다.
    -   이중 덮개 (double cover) 문제: $q$ 와 $-q$ 가 같은 회전을 나타냅니다.

> **⚠️ 짐벌락 (Gimbal Lock) 이란?**
> 오일러 각 (Euler Angles, 예: Roll, Pitch, Yaw) 과 같은 특정 표현법에서 발생하는 문제로, 회전축 중 두 개가 정렬되어 회전의 자유도 하나를 잃어버리는 현상입니다. 비행기나 우주선이 특정 자세에서 조종 불능에 빠지는 원인이 될 수 있습니다. 이 때문에 현대 로보틱스나 3D 그래픽스에서는 쿼터니언을 표준으로 사용합니다.

### 2.4. 오일러 각 (Euler Angles)

**정의**: 세 개의 각도 (예: Roll, Pitch, Yaw) 로 회전을 표현하는 직관적인 방법입니다.

$$\mathbf{e} = [\phi, \theta, \psi]^T$$

-   **장점**: 
    -   인간이 이해하기 가장 쉽습니다.
    -   최소 파라미터 (3개) 사용합니다.

-   **단점**:
    -   **짐벌락 문제**: 특정 각도에서 자유도를 잃습니다.
    -   회전 순서에 따라 결과가 달라집니다 (XYZ, ZYX 등 12가지 컨벤션).
    -   보간이 직관적이지 않습니다.

> 💡 **SLAM에서는 오일러 각을 사용하지 않습니다!** 짐벌락과 수치적 불안정성 때문에 최적화에 부적합합니다.

### [실습 연결]
`chapter01` 노트북의 **1. 회전 표현법 간 변환** 섹션에서는 `scipy` 라이브러리를 사용하여 방금 배운 세 가지 표현법 (회전 행렬, 쿼터니언, 회전 벡터) 을 서로 변환하는 코드를 실습합니다.

---

## 3. 자세 (Pose) 표현법: $SE(3)$ 변환

로봇의 상태를 완벽하게 표현하려면 방향 (회전) 뿐만 아니라 위치 (이동) 도 함께 알아야 합니다. 이 둘을 합친 것이 바로 **자세 (Pose)** 입니다. 수학적으로, 3D 공간에서의 자세는 **특수 유클리드 그룹 (Special Euclidean Group)**, 즉 $SE(3)$ 에 속하는 원소로 표현됩니다. 이는 6개의 자유도 (3-DoF 위치 + 3-DoF 방향) 를 가집니다.

### 3.1. 동차 변환 행렬 (Homogeneous Transformation Matrix)

**정의**: $SE(3)$ 변환은 보통 4x4 크기의 **동차 변환 행렬** $T$ 로 표현됩니다. 이 행렬은 회전 ($R$) 과 이동 ($\mathbf{t}$) 정보를 하나의 행렬 안에 깔끔하게 담고 있습니다.

$$
T = \begin{bmatrix} R & \mathbf{t} \\
\mathbf{0}^T & 1 \end{bmatrix} \in SE(3)
$$

-   $R$: 3x3 회전 행렬 ($R \in SO(3)$)
-   $\mathbf{t}$: 3x1 이동 벡터 ($\mathbf{t} \in \mathbb{R}^3$)
-   $\mathbf{0}^T$: [0, 0, 0] 행 벡터

**$SE(3)$ 의 정의**:
$$SE(3) = \left\{ T = \begin{bmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \bigg| R \in SO(3), \mathbf{t} \in \mathbb{R}^3 \right\}$$

### 3.2. $SE(3)$ 의 핵심 연산

-   **합성 (Composition)**: 두 변환을 순차적으로 적용하는 것은 행렬 곱으로 간단히 계산됩니다. A에서 B로, B에서 C로의 변환이 있을 때, A에서 C로의 변환은 다음과 같습니다.
    $$ T_{AC} = T_{AB} \cdot T_{BC} $$
    
    구체적으로:
    $$T_1 \cdot T_2 = \begin{bmatrix} R_1 & \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} R_2 & \mathbf{t}_2 \\ \mathbf{0}^T & 1 \end{bmatrix} = \begin{bmatrix} R_1 R_2 & R_1 \mathbf{t}_2 + \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{bmatrix}$$

-   **역변환 (Inverse)**: 변환을 되돌리는 연산입니다. B에서 A로의 변환은 A에서 B로의 변환의 역행렬입니다.
    $$ T_{BA} = T_{AB}^{-1} = \begin{bmatrix} R^T & -R^T \mathbf{t} \\
\mathbf{0}^T & 1 \end{bmatrix} $$

-   **점 변환 (Action on a point)**: B 좌표계에서 표현된 점 $\mathbf{p}_B$ 를 A 좌표계의 점 $\mathbf{p}_A$ 로 변환합니다. (점을 동차 좌표 [x, y, z, 1] 로 만들어 계산합니다.)
    $$ \begin{bmatrix} \mathbf{p}_A \\ 1 \end{bmatrix} = T_{AB} \cdot \begin{bmatrix} \mathbf{p}_B \\ 1 \end{bmatrix} $$

### 3.3. $SE(3)$ 의 그룹 성질

$SE(3)$ 는 행렬 곱셈에 대해 그룹을 이룹니다:

1. **닫힘성**: 두 SE(3) 변환의 곱은 또 다른 SE(3) 변환입니다.
2. **결합법칙**: $(T_1 T_2) T_3 = T_1 (T_2 T_3)$
3. **항등원**: $I_4 = \begin{bmatrix} I_3 & \mathbf{0} \\ \mathbf{0}^T & 1 \end{bmatrix}$
4. **역원**: 모든 $T \in SE(3)$ 에 대해 $T^{-1} \in SE(3)$ 존재

### [실습 연결]
`chapter01` 노트북의 **2. SE(3) 변환** 섹션에서는 `SE3Transform` 클래스를 직접 만들어, 자세의 합성과 역변환을 코드로 구현하고 그 결과를 시각화합니다.

---

## 4. Lie 군과 Lie 대수: 통합적 이해

앞서 배운 $SO(3)$ 와 $SE(3)$ 는 모두 **Lie 군** 의 예시입니다. 이제 이들을 통합적으로 이해할 수 있는 Lie 군론의 관점을 소개합니다.

### 4.1. Lie 군이란?

**Lie 군 (Lie Group)** 은 다음 두 구조를 동시에 가지는 수학적 대상입니다:
1. **군 구조**: 곱셈과 역원 연산이 정의됨
2. **매니폴드 구조**: 미분 가능한 곡면

즉, Lie 군의 원소들은 매끄럽게 변화할 수 있으며, 군 연산도 매끄러운 함수입니다.

**주요 예시**:
- $SO(3)$: 3D 회전의 Lie 군 (3차원 매니폴드)
- $SE(3)$: 3D rigid motion의 Lie 군 (6차원 매니폴드)
- $SU(2)$: 단위 쿼터니언의 Lie 군 (3차원 매니폴드, $SO(3)$ 의 double cover)

### 4.2. Lie 대수란?

**Lie 대수 (Lie Algebra)** 는 Lie 군의 항등원에서의 접선 공간(tangent space)입니다. 

**직관적 이해**: Lie 군이 "곡면"이라면, Lie 대수는 그 곡면의 한 점(항등원)에서의 "평면"입니다.

**주요 Lie 대수**:
- $\mathfrak{so}(3)$: $SO(3)$ 의 Lie 대수, 3x3 skew-symmetric 행렬의 집합
- $\mathfrak{se}(3)$: $SE(3)$ 의 Lie 대수, 4x4 twist 행렬의 집합

### 4.3. $\mathfrak{so}(3)$ - SO(3)의 Lie 대수

**정의**: $\mathfrak{so}(3)$ 는 skew-symmetric 행렬의 집합입니다.

$$\mathfrak{so}(3) = \{\hat{\omega} \in \mathbb{R}^{3 \times 3} | \hat{\omega}^T = -\hat{\omega}\}$$

**Hat operator**: 3D 벡터를 skew-symmetric 행렬로 변환
$$\hat{\omega} = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix} \text{ for } \omega = \begin{bmatrix} \omega_x \\ \omega_y \\ \omega_z \end{bmatrix}$$

**Lie bracket**: $[\hat{\omega}_1, \hat{\omega}_2] = \hat{\omega}_1 \hat{\omega}_2 - \hat{\omega}_2 \hat{\omega}_1 = \widehat{\omega_1 \times \omega_2}$

### 4.4. $\mathfrak{se}(3)$ - SE(3)의 Lie 대수

**정의**: $\mathfrak{se}(3)$ 는 다음 형태의 4x4 행렬의 집합입니다.

$$\mathfrak{se}(3) = \left\{\hat{\xi} = \begin{bmatrix} \hat{\omega} & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix} \bigg| \hat{\omega} \in \mathfrak{so}(3), \mathbf{v} \in \mathbb{R}^3\right\}$$

**6D twist 벡터**: $\xi = \begin{bmatrix} \mathbf{v} \\ \omega \end{bmatrix} \in \mathbb{R}^6$
- $\mathbf{v}$: linear velocity (translation 성분)
- $\omega$: angular velocity (rotation 성분)

### 4.5. 지수 사상과 로그 사상

**지수 사상 (Exponential Map)**: Lie 대수 → Lie 군
- $\exp: \mathfrak{so}(3) \rightarrow SO(3)$
- $\exp: \mathfrak{se}(3) \rightarrow SE(3)$

**로그 사상 (Logarithm Map)**: Lie 군 → Lie 대수
- $\log: SO(3) \rightarrow \mathfrak{so}(3)$
- $\log: SE(3) \rightarrow \mathfrak{se}(3)$

이 사상들은 서로 역함수 관계입니다 (항등원 근처에서).

---

## 5. 최적화의 시작: 매니폴드와 접선 공간

우리가 다루는 회전 ($SO(3)$) 과 자세 ($SE(3)$) 는 유클리드 공간처럼 평평하지 않고, 지구 표면처럼 '구부러진' 공간입니다. 이런 공간을 **매니폴드 (Manifold)** 라고 부릅니다.

### 5.1. 왜 매니폴드 개념이 필요한가?

**문제점**: 구부러진 공간에서는 일반적인 덧셈, 뺄셈이 의미가 없습니다.
- 예: 두 회전 행렬을 더하면? $(R_1 + R_2) / 2 \notin SO(3)$ (더 이상 회전 행렬이 아님!)

**해결책**: **접선 공간 (Tangent Space)** 을 도입합니다.

### 5.2. 매니폴드와 접선 공간

-   **매니폴드 (Manifold)**: 우리가 다루는 실제 공간 (예: 모든 가능한 회전들의 집합).
-   **접선 공간 (Tangent Space)**: 매니폴드의 한 점 (현재의 추정치) 에 접하는 '평평한' 벡터 공간입니다. 이 공간에서는 벡터의 덧셈, 뺄셈이 가능합니다. 최적화 과정에서 계산되는 작은 변화량 ($\Delta \mathbf{x}$) 이 바로 이 접선 공간의 벡터입니다.

> 💡 **핵심 비유**: 지구 (매니폴드) 위를 걷는다고 상상해보세요. 지구 전체는 둥글지만, 당신이 서 있는 주변의 작은 영역 (접선 공간) 은 평평해 보입니다. 우리는 이 '평평한' 공간에서 움직임을 계산한 뒤, 그 결과를 다시 '둥근' 지구 위로 옮깁니다.

### 5.3. SO(3)에서의 Exponential과 Logarithm

**Rodrigues' rotation formula** 를 통해 exponential map을 계산합니다:

$$\exp(\hat{\omega}) = I + \frac{\sin\theta}{\theta}\hat{\omega} + \frac{1-\cos\theta}{\theta^2}\hat{\omega}^2$$

여기서 $\theta = ||\omega||$ 입니다.

**작은 각도 근사**: $\theta \approx 0$ 일 때
$$\exp(\hat{\omega}) \approx I + \hat{\omega} + \frac{1}{2}\hat{\omega}^2$$

### 5.4. SE(3)에서의 Exponential과 Logarithm

SE(3)의 exponential map은 더 복잡합니다:

$$\exp\left(\begin{bmatrix} \hat{\omega} & \mathbf{v} \\ \mathbf{0}^T & 0 \end{bmatrix}\right) = \begin{bmatrix} R & J\mathbf{v} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

여기서:
- $R = \exp(\hat{\omega})$ (SO(3) exponential)
- $J = I + \frac{1-\cos\theta}{\theta^2}\hat{\omega} + \frac{\theta-\sin\theta}{\theta^3}\hat{\omega}^2$ (left Jacobian of SO(3))

### 5.5. Retraction과 Local Coordinates

**Retraction**: 접선 공간에서 매니폴드로의 매핑
$$\text{retract}: \mathcal{M} \times T_p\mathcal{M} \rightarrow \mathcal{M}$$
$$\text{retract}(T, \delta) = T \cdot \exp(\delta)$$

**Local coordinates**: 두 점 사이의 차이를 접선 공간에서 표현
$$\text{local\_coordinates}: \mathcal{M} \times \mathcal{M} \rightarrow T_p\mathcal{M}$$
$$\text{local\_coordinates}(T_1, T_2) = \log(T_1^{-1} \cdot T_2)$$

### [실습 연결]
`chapter01` 노트북의 **3. Tangent Space와 Manifold** 섹션에서는 `exp` 와 `log` 연산을 코드로 구현하고, 이를 통해 최적화에서 포즈를 어떻게 업데이트하는지 (`retract` 연산) 실습합니다.

---

## 6. 수치적 안정성과 구현

이론적으로 완벽한 수식도 컴퓨터에서 구현할 때는 수치 오차 문제에 직면합니다. 특히 회전과 관련된 계산에서는 작은 오차가 누적되어 큰 문제를 일으킬 수 있습니다.

### 6.1. 주요 수치적 문제들

1. **특이점 (Singularities)**
   - $\theta = 0$ 근처: $\sin\theta/\theta \rightarrow 0/0$
   - $\theta = \pi$ 근처: 회전축의 모호성

2. **정규화 문제**
   - 쿼터니언: $||q|| = 1$ 유지
   - 회전 행렬: $R^T R = I$ 유지

3. **부동소수점 오차 누적**
   - 반복적인 계산에서 오차 증가
   - 행렬 곱셈에서의 정밀도 손실

### 6.2. 안전한 구현 기법

**Taylor series 근사**: 특이점 근처에서 사용
```python
def sin_theta_over_theta(theta, epsilon=1e-10):
    if abs(theta) < epsilon:
        # Taylor series: sin(θ)/θ ≈ 1 - θ²/6 + θ⁴/120 - ...
        return 1.0 - theta**2 / 6.0 + theta**4 / 120.0
    else:
        return np.sin(theta) / theta
```

**Epsilon 처리**: 0으로 나누기 방지
```python
def safe_normalize(v, epsilon=1e-10):
    norm = np.linalg.norm(v)
    norm_safe = norm + epsilon
    return v / norm_safe, norm
```

### 6.3. 회전 행렬의 직교 정규화

**SVD를 이용한 방법**:
```python
def orthonormalize(R):
    U, S, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    # 반사가 아닌 회전 보장
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho
```

### 6.4. 쿼터니언 정규화

```python
def normalize_quaternion(q, epsilon=1e-10):
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if norm < epsilon:
        # 영 쿼터니언 방지
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm
```

### [실습 연결]
`chapter01` 노트북의 **수치적 안정성과 Epsilon 처리** 섹션에서는 실제로 이러한 수치적 문제들을 경험하고 해결하는 방법을 실습합니다.

---

## 7. SLAM에서의 실제 응용

지금까지 배운 이론이 실제 SLAM 시스템에서 어떻게 활용되는지 살펴봅시다.

### 7.1. Frontend: 센서 데이터 처리

**Visual SLAM**:
- 카메라 이미지에서 특징점 추출
- 특징점 매칭을 통한 상대 포즈 추정 (Essential/Fundamental matrix)
- PnP (Perspective-n-Point) 문제 해결

**LiDAR SLAM**:
- 포인트 클라우드 정합 (ICP, NDT)
- Scan-to-scan matching으로 상대 포즈 추정

### 7.2. Backend: Pose Graph Optimization

**그래프 구조**:
- 노드: 로봇의 포즈 ($T_i \in SE(3)$)
- 엣지: 포즈 간 상대 측정값 ($T_{ij}$)

**최적화 문제**:
$$\min_{\{T_i\}} \sum_{(i,j) \in \mathcal{E}} ||e_{ij}||^2_{\Omega_{ij}}$$

여기서 오차는:
$$e_{ij} = \log(T_{ij}^{-1} \cdot T_i^{-1} \cdot T_j)$$

### 7.3. Relative Pose Error

**정의**: 예측된 상대 변환과 측정된 상대 변환의 차이

$$T_{\text{error}} = T_{\text{measured}}^{-1} \cdot T_{\text{predicted}}$$
$$e = \log(T_{\text{error}}) \in \mathbb{R}^6$$

**왜 중요한가?**
- 오차가 작을수록 지도의 일관성이 높아집니다
- Loop closure 검출 시 누적 오차를 수정합니다
- 전체 최적화의 목적 함수가 됩니다

### 7.4. Jacobian 계산

최적화를 위해서는 오차 함수의 Jacobian이 필요합니다:

$$\frac{\partial e_{ij}}{\partial T_i} = -\text{Ad}_{T_j^{-1} T_i}$$

여기서 $\text{Ad}$ 는 Adjoint representation입니다.

### 7.5. 실제 구현 시 고려사항

1. **좌표계 컨벤션**
   - Right-handed vs Left-handed
   - Camera frame vs Robot frame

2. **시간 동기화**
   - 센서 간 타임스탬프 정렬
   - 보간(interpolation) 필요성

3. **Outlier rejection**
   - RANSAC
   - Robust kernels (Huber, Cauchy)

4. **계산 효율성**
   - Sparse matrix 활용
   - 병렬 처리

### [실습 연결]
`chapter01` 노트북의 **4. SLAM에서의 실제 응용: Relative Pose Error** 섹션에서는 실제로 두 포즈 간의 오차를 계산하고 이것이 최적화에 어떻게 사용되는지 실습합니다.

---

## 8. 요약 및 다음 장 예고

### 8.1. 핵심 내용 정리

이 장에서 우리는 다음을 배웠습니다:

1. **회전의 다양한 표현법**
   - 회전 행렬: 직관적이지만 과다 파라미터화
   - 회전 벡터: 최소 표현, 최적화에 적합
   - 쿼터니언: 짐벌락 없음, 보간에 유리

2. **$SO(3)$ 와 $SE(3)$ 그룹**
   - 회전과 자세의 수학적 구조
   - 그룹 연산과 성질

3. **Lie 군과 Lie 대수**
   - 통합적 이론 프레임워크
   - Exponential/Logarithm map

4. **매니폴드와 최적화**
   - 왜 접선 공간이 필요한지
   - Retraction과 local coordinates

5. **수치적 안정성**
   - 특이점 처리
   - Epsilon 기법

6. **SLAM 응용**
   - Relative pose error
   - Pose graph optimization

### 8.2. 다음 장 예고

**Chapter 2: g2o 파일 포맷 이해하기**

다음 장에서는:
- g2o (General Graph Optimization) 파일 포맷의 구조
- 실제 SLAM 데이터를 읽고 쓰는 방법
- 그래프 구조의 시각화
- 이번 장에서 배운 SE(3) 변환이 실제 데이터에서 어떻게 표현되는지

### 8.3. 추가 학습 자료

**입문서**:
- "3D Math Primer for Graphics and Game Development" - Fletcher Dunn
- "Quaternions and Rotation Sequences" - Jack Kuipers

**고급 교재**:
- "State Estimation for Robotics" - Timothy D. Barfoot
- "Probabilistic Robotics" - Sebastian Thrun et al.
- "A micro Lie theory for state estimation in robotics" - Joan Solà et al.

**온라인 자료**:
- SymForce documentation: https://symforce.org
- g2o documentation: https://github.com/RainerKuemmerle/g2o

### 8.4. 핵심 메시지

> "SLAM에서의 회전과 변환은 단순한 수학이 아닙니다. 로봇이 현실 세계를 이해하고 자신의 위치를 파악하는 핵심 도구입니다. 올바른 표현법과 안정적인 구현이 없다면, 로봇은 길을 잃고 말 것입니다."

이제 여러분은 SLAM의 수학적 기초를 탄탄히 다졌습니다. 다음 장에서는 이 지식을 실제 데이터에 적용해보겠습니다!