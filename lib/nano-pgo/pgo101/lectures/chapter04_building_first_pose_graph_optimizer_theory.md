# PGO 101 - Chapter 4 이론 강의: 나만의 첫 Pose Graph Optimizer 제작하기

**강의 목표:** 이 강의를 마치면, 여러분은 Pose Graph Optimization의 핵심 엔진이 어떻게 작동하는지 그 내부 원리를 이해하게 됩니다. 비선형 최적화 문제를 푸는 표준적인 접근법인 **Gauss-Newton 알고리즘**을 배우고, 이를 구현하기 위해 **Hessian 행렬 (H)** 과 **gradient 벡터 (b)** 를 어떻게 구축하는지, 그리고 왜 **희소 행렬 (Sparse Matrix)** 의 개념이 대규모 SLAM에서 필수적인지를 설명할 수 있게 됩니다. 이 강의는 `chapter04_building_first_pose_graph_optimizer.ipynb` 실습에서 직접 옵티마이저를 코딩하기 위한 모든 이론적 배경을 제공합니다.

---

## 1. 문제 정의: 비선형 최소 제곱 문제

Pose Graph Optimization의 목표는 모든 측정값 (간선) 과 추정된 로봇의 자세 (정점) 사이의 **모순을 최소화**하는 것입니다. 이는 수학적으로 다음과 같은 **비선형 최소 제곱 (Non-linear Least Squares)** 문제로 공식화됩니다.

$$ \underset{\mathbf{x}}{\text{minimize}} \quad F(\mathbf{x}) = \sum_{(i,j) \in \mathcal{C}} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j)^T \Omega_{ij} \mathbf{e}_{ij}(\mathbf{x}_i, \mathbf{x}_j) $$

-   $\mathbf{x}$: 우리가 최적화하려는 모든 변수 (모든 로봇의 자세 $\mathbf{x}_i, \mathbf{x}_j, ...$).
-   $\mathbf{e}_{ij}$: **에러 (Residual)**. 두 포즈 사이의 측정값과 예측값의 차이를 나타내는 벡터.
-   $\Omega_{ij}$: **정보 행렬 (Information Matrix)**. 해당 측정값의 신뢰도 (가중치) 를 나타내는 행렬.

이 비용 함수 $F(\mathbf{x})$ 는 에러 함수 $\mathbf{e}_{ij}$ 가 변수 $\mathbf{x}$ 에 대해 비선형적이기 때문에 (회전 변환 등이 포함되므로), 한 번에 정답을 찾을 수 없습니다. 따라서 우리는 **반복적인 방법**을 통해 점진적으로 해에 가까워져야 합니다.

## 2. Gauss-Newton 알고리즘: 비선형 문제를 선형 문제로

Gauss-Newton은 비선형 문제를 각 반복 단계마다 풀기 쉬운 **선형 문제로 근사**하여 푸는 영리한 방법입니다.

> 💡 **핵심 비유**: 구불구불한 언덕 (비선형 비용 함수) 을 내려간다고 상상해보세요. 현재 위치에서 가장 낮은 곳으로 가기 위해, 우리는 주변을 잠시 '평평한 경사면' (선형 근사) 이라고 가정하고, 그 경사면에서 가장 가파른 방향으로 한 걸음 내딛습니다. 이 과정을 반복하면 결국 골짜기 (최적해) 에 도달하게 됩니다.

### 2.1. 에러 함수의 선형화

현재 추정치 $\mathbf{x}$ 에서 작은 변화량 $\Delta\mathbf{x}$ 만큼 움직였을 때의 에러 함수 $\mathbf{e}(\mathbf{x} + \Delta\mathbf{x})$ 를 **1차 테일러 급수 (Taylor Series)** 로 근사하면 다음과 같습니다.

$$ \mathbf{e}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{e}(\mathbf{x}) + J(\mathbf{x}) \Delta\mathbf{x} $$

여기서 $J(\mathbf{x})$ 는 현재 추정치 $\mathbf{x}$ 에서의 에러 함수에 대한 **야코비안 (Jacobian)** 입니다. 이제 우리의 목표는 이 선형화된 에러의 제곱 합을 최소화하는 $\Delta\mathbf{x}$ 를 찾는 것입니다.

$$ \underset{\Delta\mathbf{x}}{\text{minimize}} \quad \| \mathbf{e}(\mathbf{x}) + J(\mathbf{x}) \Delta\mathbf{x} \|^2 $$

### 2.2. 정규 방정식 (Normal Equation) 유도

위의 선형 최소 제곱 문제의 해는, 목적 함수를 $\Delta\mathbf{x}$ 에 대해 미분하여 0으로 놓음으로써 구할 수 있습니다. 그 결과로 얻어지는 것이 바로 **정규 방정식 (Normal Equation)** 입니다.

$$ (J^T \Omega J) \Delta\mathbf{x} = -J^T \Omega \mathbf{e} $$

이 식을 우리가 풀고자 했던 $H \Delta\mathbf{x} = -b$ 와 비교하면, $H$ 와 $b$ 가 각각 다음과 같이 정의됨을 알 수 있습니다.

-   $H = J^T \Omega J$
-   $b = J^T \Omega \mathbf{e}$

이것이 Gauss-Newton 알고리즘의 핵심입니다. 각 반복마다 야코비안 $J$ 와 에러 $\mathbf{e}$ 를 계산하여 $H$ 와 $b$ 를 만들고, 선형 방정식을 풀어 최적의 업데이트 스텝 $\Delta\mathbf{x}$ 를 찾는 것입니다.

### [실습 연결]
`chapter04` 노트북의 **4. 최적화 알고리즘 구현** 섹션에서는, 이 Gauss-Newton 알고리즘의 반복 루프를 직접 코드로 구현합니다.

---

## 3. H 행렬과 b 벡터 구축: 정보의 집약

$H$ 와 $b$ 는 그래프의 모든 간선 (측정값) 이 제공하는 정보를 하나로 모아놓은 것입니다. 각 간선은 $H$ 와 $b$ 에 자신의 정보를 '더하는' 방식으로 기여합니다.

-   **하나의 간선 (측정값) 이 기여하는 양**:
    -   $H \mathrel{+}= J^T \Omega J$
    -   $b \mathrel{+}= J^T \Omega \mathbf{e}$

**중요한 점**: Gauss-Newton에서 사용하는 $H = J^T \Omega J$ 는 실제 Hessian의 근사치입니다. 이 근사의 장점은 $J^T \Omega J$ 가 항상 **양의 준정부호 행렬 (Positive Semi-Definite)** 이라는 것입니다. 이는 최적화 알고리즘이 항상 내리막 방향을 찾도록 보장하여 안정성에 기여합니다.

### [실습 연결]
`chapter04` 노트북의 **3. H 행렬과 b 벡터 구축** 섹션에서는, 모든 엣지를 순회하며 각 엣지의 야코비안과 에러를 계산하고, 이를 통해 전체 $H$ 와 $b$ 를 구축하는 과정을 구현합니다.

---

## 4. 희소성 (Sparsity): 대규모 SLAM의 열쇠

실제 SLAM 문제에서는 수천, 수만 개의 포즈가 존재합니다. 만약 1000개의 포즈가 있다면, H 행렬의 크기는 (1000 * 6) x (1000 * 6) = 6000x6000 이 됩니다. 이는 엄청난 양의 메모리와 계산을 요구합니다.

하지만 다행히도, 대부분의 포즈는 몇 개의 **이웃한 포즈**와만 직접적인 관계 (간선) 를 가집니다. 예를 들어, 포즈 $\mathbf{x}_5$ 는 보통 $\mathbf{x}_4$ 와 $\mathbf{x}_6$ 와만 연결됩니다.

-   **결과**: $H$ 행렬은 대부분의 원소가 0인 **희소 행렬 (Sparse Matrix)** 이 됩니다. `EDGE(i, j)` 는 $H$ 행렬에서 $(i,i), (i,j), (j,i), (j,j)$ 에 해당하는 블록에만 영향을 미칩니다.
-   **장점**: `scipy.sparse` 와 같은 희소 행렬 라이브러리를 사용하면, 0이 아닌 원소들만 저장하고 계산하여 메모리 사용량과 계산 시간을 **수백 배 이상** 절약할 수 있습니다. 이것이 대규모 SLAM이 가능한 이유입니다.

### [실습 연결]
`chapter04` 노트북의 **7. Sparse Matrix 분석** 섹션에서는, 구축된 H 행렬을 `matplotlib.pyplot.spy` 로 시각화하여 얼마나 많은 부분이 0으로 채워져 있는지, 즉 희소성 패턴을 직접 눈으로 확인합니다.
