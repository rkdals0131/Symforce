# PGO 101 - Chapter 10 이론 강의: 카메라와 함께하는 SLAM - Visual SLAM의 모든 것

**강의 목표:** 이 강의를 마치면, 여러분은 Visual SLAM의 핵심인 **카메라 모델**과 **Bundle Adjustment**의 수학적 기초를 완벽히 이해하게 됩니다. 핀홀 카메라 모델부터 어안 렌즈를 위한 ATAN 모델까지, 그리고 3D 점을 이미지 평면에 투영하는 과정의 모든 세부사항을 설명할 수 있게 됩니다. 특히 Bundle Adjustment가 어떻게 카메라 포즈와 3D 랜드마크를 동시에 최적화하여 전역적으로 일관된 지도를 만드는지, 그리고 이 과정에서 발생하는 수치적 문제들을 어떻게 처리하는지 깊이 있게 이해하게 됩니다. 이 강의는 `chapter10_visual_slam_with_cameras.ipynb` 실습에서 실제 Visual SLAM 시스템을 구현하기 위한 모든 이론적 토대를 제공합니다.

> 💡 **이 장의 핵심 질문들:**
> - 왜 카메라는 3차원 세계를 2차원으로 "압축"할 수 있는가?
> - 렌즈 왜곡이 SLAM 정확도에 미치는 영향은?
> - Bundle Adjustment는 어떻게 드리프트를 최소화하는가?
> - 시각적 특징의 야코비안이 왜 그렇게 복잡한가?
> - Visual SLAM이 LiDAR SLAM보다 어려운 이유는?

---

## 목차

1. [Visual SLAM의 개요와 도전 과제](#1-visual-slam의-개요와-도전-과제)
2. [카메라 모델의 수학적 기초](#2-카메라-모델의-수학적-기초)
3. [핀홀 카메라 모델 심화](#3-핀홀-카메라-모델-심화)
4. [렌즈 왜곡과 ATAN 모델](#4-렌즈-왜곡과-atan-모델)
5. [투영 기하학과 동차 좌표](#5-투영-기하학과-동차-좌표)
6. [Bundle Adjustment - 그래프 최적화의 관점](#6-bundle-adjustment---그래프-최적화의-관점)
7. [시각 측정값의 야코비안](#7-시각-측정값의-야코비안)
8. [수치적 안정성과 엡실론 처리](#8-수치적-안정성과-엡실론-처리)
9. [Visual SLAM vs Pose-Only SLAM](#9-visual-slam-vs-pose-only-slam)
10. [실제 구현 시 고려사항](#10-실제-구현-시-고려사항)

---

## 1. Visual SLAM의 개요와 도전 과제

### 1.1 Visual SLAM이란?

Visual SLAM은 카메라를 주 센서로 사용하여 동시에 자기 위치를 추정하고 환경의 3D 지도를 구축하는 기술입니다. LiDAR나 초음파 센서와 달리, 카메라는:

- **풍부한 정보**: 색상, 텍스처, 의미론적 정보
- **저비용**: 센서 가격이 매우 저렴
- **소형화**: 드론, 스마트폰에도 탑재 가능

하지만 동시에:

- **깊이 정보 부재**: 단일 이미지로는 거리를 알 수 없음
- **조명 민감성**: 빛의 변화에 취약
- **계산 복잡도**: 이미지 처리는 계산량이 많음

### 1.2 Visual SLAM의 파이프라인

```
이미지 입력 → 특징 추출 → 특징 매칭 → 모션 추정 → Bundle Adjustment → 지도 업데이트
    ↑                                                                          ↓
    └──────────────────────── 루프 클로저 검출 ←─────────────────────────────┘
```

각 단계는 수학적 도전을 포함합니다:

1. **특징 추출**: 회전/크기 불변 특징점 찾기
2. **특징 매칭**: 잘못된 대응점 제거
3. **모션 추정**: 에피폴라 기하학
4. **Bundle Adjustment**: 비선형 최적화
5. **루프 클로저**: 장소 인식

### 1.3 수학적 도전 과제

**1. 투영의 비선형성**

3D 점 $(X, Y, Z)$ 를 2D 점 $(u, v)$ 로 투영하는 과정:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_c} K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}$$

여기서 $1/Z_c$ 때문에 비선형성이 발생합니다.

**2. 스케일 모호성**

단일 카메라로는 절대적인 스케일을 알 수 없습니다:
- 작은 물체가 가까이 있는지
- 큰 물체가 멀리 있는지

**3. 동적 환경**

움직이는 물체들이 잘못된 제약을 생성할 수 있습니다.

---

## 2. 카메라 모델의 수학적 기초

### 2.1 좌표계 정의

Visual SLAM에서 사용하는 주요 좌표계:

1. **월드 좌표계 (World Frame)**: $\mathcal{F}_w$
   - 전역 참조 프레임
   - 고정된 원점과 축

2. **카메라 좌표계 (Camera Frame)**: $\mathcal{F}_c$
   - 카메라 중심이 원점
   - Z축이 광축 방향
   - X축은 오른쪽, Y축은 아래

3. **이미지 좌표계 (Image Frame)**: $\mathcal{F}_i$
   - 픽셀 단위
   - 원점은 이미지 왼쪽 위

### 2.2 좌표 변환

**월드에서 카메라로:**

$$\mathbf{P}_c = T_{cw} \mathbf{P}_w$$

여기서 $T_{cw} \in SE(3)$ 는 월드 좌표계에서 카메라 좌표계로의 변환:

$$T_{cw} = \begin{bmatrix} R_{cw} & \mathbf{t}_{cw} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

### 2.3 투영의 기하학적 의미

카메라는 3D 세계를 2D 이미지로 "압축"합니다. 이 과정에서:

- **깊이 정보 손실**: $Z$ 좌표가 사라짐
- **원근 효과**: 멀리 있는 물체가 작게 보임
- **시야각 제한**: 카메라 뒤의 점은 보이지 않음

---

## 3. 핀홀 카메라 모델 심화

### 3.1 핀홀 카메라의 원리

이상적인 핀홀 카메라는 모든 빛이 하나의 점(핀홀)을 통과한다고 가정합니다:

```
실제 세계                핀홀              이미지 평면
    *                     |                    *
     \                    |                   /
      \                   |                  /
       \                  O                 /
        \                 |                /
         \                |               /
          *               |              *
```

### 3.2 수학적 모델링

**단계 1: 3D에서 정규화된 이미지 평면으로**

카메라 좌표계의 점 $\mathbf{P}_c = [X_c, Y_c, Z_c]^T$ 를 정규화된 이미지 평면 (focal length = 1)에 투영:

$$\begin{bmatrix} x_n \\ y_n \end{bmatrix} = \begin{bmatrix} X_c / Z_c \\ Y_c / Z_c \end{bmatrix}$$

**단계 2: 내부 파라미터 적용**

내부 파라미터 행렬 $K$:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

여기서:
- $f_x, f_y$: 초점 거리 (픽셀 단위)
- $c_x, c_y$: 주점 (principal point)
- $s$: skew 파라미터 (보통 0)

최종 픽셀 좌표:

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = K \begin{bmatrix} X_c/Z_c \\ Y_c/Z_c \\ 1 \end{bmatrix}$$

### 3.3 완전한 투영 파이프라인

월드 좌표의 점을 이미지로 투영하는 전체 과정:

$$\mathbf{p} = \pi(K, T_{cw}, \mathbf{P}_w)$$

단계별로:

1. **월드에서 카메라로**: $\mathbf{P}_c = R_{cw} \mathbf{P}_w + \mathbf{t}_{cw}$
2. **투영**: $\mathbf{p}_n = [X_c/Z_c, Y_c/Z_c]^T$
3. **픽셀 변환**: $\mathbf{p} = K[\mathbf{p}_n; 1]$

### 3.4 역투영 (Backprojection)

픽셀에서 3D 광선으로:

$$\mathbf{r} = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

이는 깊이 정보 없이는 유일한 3D 점을 결정할 수 없습니다.

---

## 4. 렌즈 왜곡과 ATAN 모델

### 4.1 왜곡의 원인

실제 렌즈는 완벽하지 않아 왜곡이 발생합니다:

1. **방사 왜곡 (Radial Distortion)**
   - 배럴 왜곡: 이미지가 바깥쪽으로 부풂
   - 핀쿠션 왜곡: 이미지가 안쪽으로 움츠러듦

2. **접선 왜곡 (Tangential Distortion)**
   - 렌즈와 이미지 센서가 평행하지 않을 때 발생

### 4.2 표준 왜곡 모델

**방사 왜곡:**

$$\begin{aligned}
x_d &= x_u (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\
y_d &= y_u (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
\end{aligned}$$

여기서 $r^2 = x_u^2 + y_u^2$ 는 정규화된 좌표에서의 거리입니다.

**접선 왜곡:**

$$\begin{aligned}
x_d &= x_u + [2p_1 x_u y_u + p_2(r^2 + 2x_u^2)] \\
y_d &= y_u + [p_1(r^2 + 2y_u^2) + 2p_2 x_u y_u]
\end{aligned}$$

### 4.3 ATAN (Fisheye) 모델

광각 렌즈를 위한 ATAN 모델:

**투영 함수:**

$$r = \frac{2}{\pi} \arctan(r_u \cdot \omega)$$

여기서:
- $r_u = \sqrt{x_u^2 + y_u^2}$: 정규화된 거리
- $\omega$: 왜곡 파라미터
- $r$: 왜곡된 거리

**왜곡된 좌표:**

$$\begin{aligned}
x_d &= \frac{r}{r_u} x_u \\
y_d &= \frac{r}{r_u} y_u
\end{aligned}$$

### 4.4 왜곡 보정

**반복적 방법:**

왜곡 제거는 일반적으로 반복적 방법을 사용:

```python
def undistort_iterative(x_d, y_d, k1, k2, p1, p2, max_iter=10):
    x_u, y_u = x_d, y_d  # 초기 추정
    
    for _ in range(max_iter):
        r2 = x_u**2 + y_u**2
        radial = 1 + k1*r2 + k2*r2**2
        
        dx = 2*p1*x_u*y_u + p2*(r2 + 2*x_u**2)
        dy = p1*(r2 + 2*y_u**2) + 2*p2*x_u*y_u
        
        x_u = (x_d - dx) / radial
        y_u = (y_d - dy) / radial
    
    return x_u, y_u
```

---

## 5. 투영 기하학과 동차 좌표

### 5.1 동차 좌표의 힘

동차 좌표는 투영 기하학을 선형 대수로 표현할 수 있게 합니다:

**2D 점**: $(x, y) \rightarrow [x, y, 1]^T$ 또는 $[wx, wy, w]^T$

**3D 점**: $(X, Y, Z) \rightarrow [X, Y, Z, 1]^T$

### 5.2 투영 행렬

완전한 투영은 $3 \times 4$ 행렬로 표현:

$$P = K[R | \mathbf{t}]$$

투영:

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = P \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$

여기서 $\lambda$ 는 스케일 팩터입니다.

### 5.3 에피폴라 기하학

두 뷰 사이의 기하학적 제약:

**기본 행렬 (Fundamental Matrix):**

$$\mathbf{p}_2^T F \mathbf{p}_1 = 0$$

**본질 행렬 (Essential Matrix):**

$$\mathbf{p}_2'^T E \mathbf{p}_1' = 0$$

여기서 $\mathbf{p}'$ 는 정규화된 이미지 좌표입니다.

관계: $E = K_2^T F K_1$

### 5.4 삼각측량 (Triangulation)

두 뷰에서 관찰된 점의 3D 위치 복원:

**선형 방법:**

$$A \mathbf{X} = 0$$

여기서:

$$A = \begin{bmatrix}
u_1 P_1^{3T} - P_1^{1T} \\
v_1 P_1^{3T} - P_1^{2T} \\
u_2 P_2^{3T} - P_2^{1T} \\
v_2 P_2^{3T} - P_2^{2T}
\end{bmatrix}$$

SVD를 사용하여 해를 구합니다.

---

## 6. Bundle Adjustment - 그래프 최적화의 관점

### 6.1 문제 정의

Bundle Adjustment는 다음을 동시에 최적화합니다:

- **카메라 포즈**: $\{T_i\}_{i=1}^{N_c}$
- **3D 랜드마크**: $\{\mathbf{P}_j\}_{j=1}^{N_l}$

목적 함수:

$$\min_{\{T_i\}, \{\mathbf{P}_j\}} \sum_{i,j \in \mathcal{V}} \rho\left(\|\mathbf{z}_{ij} - \pi(K_i, T_i, \mathbf{P}_j)\|^2_{\Sigma_{ij}}\right)$$

여기서:
- $\mathbf{z}_{ij}$: 카메라 $i$ 에서 관찰된 랜드마크 $j$ 의 픽셀 좌표
- $\pi$: 투영 함수
- $\rho$: robust cost function
- $\mathcal{V}$: 가시성 집합

### 6.2 그래프 표현

Bundle Adjustment를 이분 그래프로 표현:

```
카메라 노드     엣지 (측정값)      랜드마크 노드
    C1 ━━━━━━━━━━━━━━━━━━━━━━━━━ L1
    C2 ━━━━━━━┓    ┏━━━━━━━━━━━━ L2
    C3 ━━━━━━━╋━━━━╋━━━━━━━━━━━━ L3
    C4 ━━━━━━━┛    ┗━━━━━━━━━━━━ L4
```

### 6.3 희소성 패턴

야코비안 행렬의 구조:

$$J = \begin{bmatrix} 
\frac{\partial \mathbf{e}_{11}}{\partial T_1} & 0 & \cdots & \frac{\partial \mathbf{e}_{11}}{\partial \mathbf{P}_1} & 0 & \cdots \\
0 & \frac{\partial \mathbf{e}_{22}}{\partial T_2} & \cdots & 0 & \frac{\partial \mathbf{e}_{22}}{\partial \mathbf{P}_2} & \cdots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots
\end{bmatrix}$$

정보 행렬 $H = J^T J$ 의 구조:

$$H = \begin{bmatrix} U & W \\ W^T & V \end{bmatrix}$$

여기서:
- $U$: 카메라-카메라 블록
- $V$: 랜드마크-랜드마크 블록 (대각)
- $W$: 카메라-랜드마크 블록

### 6.4 Schur Complement

랜드마크를 주변화하여 계산 효율성 향상:

$$S = U - W V^{-1} W^T$$

$V$ 가 대각 행렬이므로 역행렬 계산이 효율적입니다.

---

## 7. 시각 측정값의 야코비안

### 7.1 재투영 오차

재투영 오차 정의:

$$\mathbf{e}_{ij} = \mathbf{z}_{ij} - \pi(K_i, T_i, \mathbf{P}_j)$$

### 7.2 카메라 포즈에 대한 야코비안

연쇄 법칙을 사용:

$$\frac{\partial \mathbf{e}_{ij}}{\partial T_i} = -\frac{\partial \pi}{\partial T_i} = -\frac{\partial \pi}{\partial \mathbf{P}_c} \cdot \frac{\partial \mathbf{P}_c}{\partial T_i}$$

**단계 1: 투영에 대한 미분**

$$\frac{\partial \pi}{\partial \mathbf{P}_c} = \frac{1}{Z_c} \begin{bmatrix} f_x & 0 & -f_x X_c/Z_c \\ 0 & f_y & -f_y Y_c/Z_c \end{bmatrix}$$

**단계 2: 변환에 대한 미분**

SE(3)의 리 대수를 사용:

$$\frac{\partial (R\mathbf{P}_w + \mathbf{t})}{\partial \boldsymbol{\xi}} = \begin{bmatrix} I & -[\mathbf{P}_c]_\times \end{bmatrix}$$

여기서 $\boldsymbol{\xi} = [\boldsymbol{\rho}^T, \boldsymbol{\phi}^T]^T$ 는 se(3) 원소입니다.

**최종 야코비안:**

$$\frac{\partial \mathbf{e}_{ij}}{\partial \boldsymbol{\xi}_i} = -\frac{1}{Z_c} \begin{bmatrix} f_x & 0 & -f_x X_c/Z_c \\ 0 & f_y & -f_y Y_c/Z_c \end{bmatrix} \begin{bmatrix} I & -[\mathbf{P}_c]_\times \end{bmatrix}$$

### 7.3 랜드마크에 대한 야코비안

$$\frac{\partial \mathbf{e}_{ij}}{\partial \mathbf{P}_j} = -\frac{\partial \pi}{\partial \mathbf{P}_c} \cdot \frac{\partial \mathbf{P}_c}{\partial \mathbf{P}_j} = -\frac{\partial \pi}{\partial \mathbf{P}_c} \cdot R_i$$

### 7.4 왜곡이 있는 경우

왜곡 모델을 포함한 야코비안:

$$\frac{\partial \mathbf{e}_{ij}}{\partial \boldsymbol{\xi}_i} = -\frac{\partial \mathbf{d}}{\partial \mathbf{p}_u} \cdot \frac{\partial \mathbf{p}_u}{\partial \mathbf{P}_c} \cdot \frac{\partial \mathbf{P}_c}{\partial \boldsymbol{\xi}_i}$$

여기서 $\mathbf{d}$ 는 왜곡 함수입니다.

---

## 8. 수치적 안정성과 엡실론 처리

### 8.1 영으로 나누기 방지

카메라 뒤의 점 ($Z_c \leq 0$) 처리:

```python
def safe_projection(P_c, epsilon=1e-9):
    X_c, Y_c, Z_c = P_c
    
    # Z가 너무 작으면 엡실론으로 대체
    if Z_c < epsilon:
        return None  # 또는 특별한 처리
    
    x = X_c / Z_c
    y = Y_c / Z_c
    
    return x, y
```

### 8.2 특이점 근처에서의 안정성

**ATAN 모델의 특이점:**

$$r_d = \frac{2}{\pi} \arctan(r_u \cdot \omega)$$

$r_u = 0$ 근처에서:

```python
def atan_distortion(r_u, omega, epsilon=1e-9):
    if abs(r_u) < epsilon:
        # Taylor 전개 사용
        return r_u * omega * (2/np.pi)
    else:
        return (2/np.pi) * np.arctan(r_u * omega)
```

### 8.3 야코비안의 수치적 안정성

**조건수 개선:**

$$J_{scaled} = D_r^{-1/2} J D_c^{-1/2}$$

여기서 $D_r$, $D_c$ 는 스케일링 행렬입니다.

### 8.4 Robust Cost Functions

아웃라이어 처리를 위한 Huber norm:

$$\rho_{Huber}(e) = \begin{cases}
\frac{1}{2}e^2 & |e| \leq \delta \\
\delta(|e| - \frac{1}{2}\delta) & |e| > \delta
\end{cases}$$

---

## 9. Visual SLAM vs Pose-Only SLAM

### 9.1 Pose-Only SLAM의 한계

Pose-only SLAM (예: ICP)은:
- 랜드마크를 최적화하지 않음
- 드리프트가 누적됨
- 루프 클로저 후 불일치 발생

### 9.2 Bundle Adjustment의 장점

**1. 전역 일관성**

모든 관측값이 동시에 고려되어 전역적으로 일관된 지도를 생성합니다.

**2. 최적의 추정**

Maximum Likelihood 관점에서 최적:

$$\mathbf{x}^* = \arg\max_{\mathbf{x}} P(\mathbf{z} | \mathbf{x})$$

**3. 불확실성 정량화**

공분산 행렬에서 불확실성 추정:

$$\Sigma = (J^T \Omega J)^{-1}$$

### 9.3 계산 복잡도 비교

| 방법 | 시간 복잡도 | 메모리 복잡도 |
|------|-------------|---------------|
| Pose-only | $O(N_c^3)$ | $O(N_c^2)$ |
| Bundle Adjustment | $O((N_c + N_l)^3)$ | $O((N_c + N_l)^2)$ |
| Schur BA | $O(N_c^3 + N_l)$ | $O(N_c^2 + N_l)$ |

### 9.4 하이브리드 접근법

실시간 성능을 위한 전략:

1. **키프레임 기반**: 모든 프레임이 아닌 선택된 키프레임만 최적화
2. **슬라이딩 윈도우**: 최근 N개 프레임만 최적화
3. **로컬 BA**: 현재 위치 주변만 최적화

---

## 10. 실제 구현 시 고려사항

### 10.1 특징점 선택

좋은 특징점의 조건:

1. **반복성**: 다른 시점에서도 검출 가능
2. **독특성**: 매칭이 용이
3. **분포**: 이미지 전체에 고르게 분포

```python
def select_features(features, image_width, image_height, min_distance=30):
    # 그리드 기반 특징 선택
    grid_size = min_distance
    grid = {}
    
    for feature in features:
        x, y = feature.pt
        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)
        
        key = (grid_x, grid_y)
        if key not in grid or feature.response > grid[key].response:
            grid[key] = feature
    
    return list(grid.values())
```

### 10.2 초기화 전략

**1. 5점 알고리즘**

Essential Matrix 추정 후 분해:

$$E = U \Sigma V^T$$

가능한 해:
- $R_1 = UWV^T$, $\mathbf{t}_1 = U[:, 2]$
- $R_2 = UWV^T$, $\mathbf{t}_2 = -U[:, 2]$
- $R_3 = UW^TV^T$, $\mathbf{t}_3 = U[:, 2]$
- $R_4 = UW^TV^T$, $\mathbf{t}_4 = -U[:, 2]$

**2. Chirality 검사**

올바른 해는 모든 점이 두 카메라 앞에 있어야 합니다.

### 10.3 루프 클로저 통합

루프 클로저 검출 시:

1. **기하학적 검증**: Essential Matrix + RANSAC
2. **Bundle Adjustment 트리거**: 전체 또는 로컬 최적화
3. **포즈 그래프 업데이트**: 새로운 제약 추가

### 10.4 실시간 최적화 팁

**1. 증분적 업데이트**

```python
# 이전 솔루션을 초기값으로 사용
optimizer.set_initial_values(previous_solution)
optimizer.optimize(max_iterations=10)  # 적은 반복
```

**2. 병렬 처리**

- 특징 추출/매칭: GPU 활용
- Bundle Adjustment: 멀티스레드 선형 대수

**3. 적응적 품질**

```python
def adaptive_optimization(optimizer, time_budget):
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < time_budget:
        optimizer.iterate()
        iteration += 1
        
        if optimizer.converged():
            break
    
    return optimizer.get_solution()
```

### 10.5 메모리 관리

**1. 키프레임 선택**

```python
def should_add_keyframe(current_pose, last_keyframe_pose, 
                       min_translation=0.1, min_rotation=0.1):
    translation = np.linalg.norm(current_pose.t - last_keyframe_pose.t)
    rotation = np.arccos((np.trace(current_pose.R.T @ last_keyframe_pose.R) - 1) / 2)
    
    return translation > min_translation or rotation > min_rotation
```

**2. 랜드마크 제거**

관측 횟수가 적거나 오래된 랜드마크 제거:

```python
def prune_landmarks(landmarks, min_observations=3, max_age=100):
    pruned = []
    for landmark in landmarks:
        if (landmark.observation_count >= min_observations and 
            landmark.age < max_age):
            pruned.append(landmark)
    return pruned
```

---

## 요약 및 핵심 포인트

### 핵심 개념 정리

1. **카메라 모델**: 3D 세계를 2D 이미지로 투영하는 수학적 모델
   - 핀홀 모델: 이상적이지만 단순
   - ATAN 모델: 광각 렌즈를 위한 현실적 모델

2. **Bundle Adjustment**: 카메라 포즈와 3D 점을 동시 최적화
   - 재투영 오차 최소화
   - 희소 구조 활용으로 효율적 계산

3. **야코비안 계산**: 최적화의 핵심
   - 카메라 포즈에 대한 미분: SE(3) 리 대수 활용
   - 랜드마크에 대한 미분: 선형 변환

4. **수치적 안정성**: 실제 구현의 핵심
   - 엡실론 처리로 특이점 회피
   - Robust cost function으로 아웃라이어 처리

5. **실시간 고려사항**: 
   - 키프레임 기반 처리
   - 증분적 최적화
   - 병렬 처리

### 실무 체크리스트

✅ 카메라 캘리브레이션 정확도 확인  
✅ 특징점 분포 균일성 검사  
✅ 수치적 안정성을 위한 엡실론 값 설정  
✅ 키프레임 선택 기준 조정  
✅ Bundle Adjustment 수렴 조건 설정  
✅ 메모리 사용량 모니터링  

### 다음 단계

이제 Visual SLAM의 이론적 기초를 완전히 이해했으니, 실습에서:

1. PosedCamera 클래스로 투영 구현
2. Bundle Adjustment 그래프 구축
3. 야코비안 계산 및 검증
4. 전체 Visual SLAM 파이프라인 구현

을 진행하여 이론을 실제 코드로 구현해보세요!

**핵심 질문 되돌아보기:**
- ✓ 카메라는 원근 투영으로 3D를 2D로 압축
- ✓ 렌즈 왜곡은 정확한 3D 복원을 방해
- ✓ Bundle Adjustment는 전역 일관성으로 드리프트 최소화
- ✓ 시각 야코비안은 투영의 비선형성 때문에 복잡
- ✓ Visual SLAM은 깊이 모호성과 조명 변화로 어려움

이 지식을 바탕으로 강력한 Visual SLAM 시스템을 구축할 수 있습니다!