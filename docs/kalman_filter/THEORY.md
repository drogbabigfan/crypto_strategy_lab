# Adaptive Kalman Filter 이론

금융 시계열을 위한 적응형 칼만 필터의 이론적 배경.

---

## 1. 상태 공간 모델

### 1.1 기본 구조

칼만 필터는 **선형 동적 시스템(Linear Dynamic System)**의 상태를 추정하는 재귀적 알고리즘이다.

**상태 방정식 (Process Equation)**:
$$x_t = F_t x_{t-1} + w_t$$

- $x_t$: 시점 $t$에서의 숨겨진 상태 벡터 (추세, 속도)
- $F_t$: 상태 전이 행렬
- $w_t \sim N(0, Q_t)$: 프로세스 잡음

**관측 방정식 (Measurement Equation)**:
$$z_t = H_t x_t + v_t$$

- $z_t$: 시점 $t$에서의 관측된 가격
- $H_t$: 관측 행렬
- $v_t \sim N(0, R_t)$: 측정 잡음

### 1.2 금융적 해석

| 요소 | 수학적 의미 | 금융적 의미 |
|------|------------|------------|
| $x_t$ | 상태 벡터 | 실제 추세 + 모멘텀 |
| $z_t$ | 관측값 | 시장 가격 (노이즈 포함) |
| $Q_t$ | 프로세스 잡음 | 추세 자체의 변동성 |
| $R_t$ | 측정 잡음 | 미시적 시장 노이즈 |

---

## 2. 칼만 이득 (Kalman Gain)

### 2.1 수식

$$K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R_t)^{-1}$$

### 2.2 해석

칼만 이득 $K_t$는 **"새로운 관측치를 얼마나 신뢰할 것인가"**를 결정하는 가중치.

| 조건 | 효과 |
|------|------|
| $R_t$ 크다 (시장 시끄러움) | $K_t$ 작아짐 → 기존 추세 유지 (스무딩) |
| $Q_t$ 크다 (추세 급변) | $P$ 증가 → $K_t$ 커짐 → 빠른 반응 (추적) |
| $P$ 크다 (불확실성 높음) | $K_t$ 커짐 → 새 정보에 민감 |

### 2.3 그래픽 해석

```
R 높음 (노이즈 큼)    →  K 낮음  →  Smoothing 강화
     ↑                        ↓
     └─────── 트레이드오프 ─────┘
     ↓                        ↑
Q 높음 (추세 변동)    →  K 높음  →  Tracking 강화
```

---

## 3. 적응형 메커니즘

### 3.1 필요성

전통적 칼만 필터는 $Q$와 $R$을 **상수**로 가정하지만, 금융 시장은:

1. **이분산성 (Heteroscedasticity)**: 변동성이 시간에 따라 변함
2. **변동성 군집화 (Clustering)**: 고변동 기간이 연속됨
3. **레짐 전환**: 추세/횡보/급변 등 상태 변화

고정 파라미터 사용 시:
- 급격한 시장 충격 → **과소 반응**
- 평온한 시장 → **과잉 반응**

### 3.2 적응형 R_t (측정 잡음)

**목적**: 시장 변동성에 따라 필터링 강도 조절

**추정 방법**:
$$R_t = \text{Var}(\Delta P_{t-N:t}) \times r_{scale}$$

또는 Parkinson 변동성 사용:
$$R_t = \frac{1}{4 \ln 2} \sum_{i=t-N}^{t} \left(\ln \frac{H_i}{L_i}\right)^2$$

**효과**:
- 변동성 ↑ → $R_t$ ↑ → 스무딩 강화
- 변동성 ↓ → $R_t$ ↓ → 추적 민감도 증가

### 3.3 적응형 Q_t (프로세스 잡음)

**목적**: 추세 변화 강도에 따라 반응 속도 조절

**추정 방법**:
$$Q_t = \text{Var}(v_{t-N:t}) \times q_{scale} \times \text{momentum\_factor}$$

여기서:
$$\text{momentum\_factor} = 1 + w \times \frac{|\Delta v|}{|\bar{v}|}$$

**효과**:
- 추세 가속 → $Q_t$ ↑ → 빠른 추적
- 추세 안정 → $Q_t$ ↓ → 부드러운 필터링

---

## 4. 2-상태 모델

### 4.1 Constant Velocity Model

상태 벡터:
$$x_t = \begin{bmatrix} \text{trend} \\ \text{velocity} \end{bmatrix}$$

상태 전이 행렬:
$$F = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

의미:
- $\text{trend}_{t+1} = \text{trend}_t + \text{velocity}_t$
- $\text{velocity}_{t+1} = \text{velocity}_t$

관측 행렬:
$$H = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

의미: 관측값 = 추세만 (속도는 관측 불가)

### 4.2 왜 2-상태인가?

| 상태 수 | 모델 | 장점 | 단점 |
|---------|------|------|------|
| 1 | Random Walk | 단순 | 추세 추적 불가 |
| 2 | Constant Velocity | 추세+모멘텀 | 가속도 무시 |
| 3 | Constant Acceleration | 가속도 포착 | 과적합 위험 |

**2-상태 선택 이유**:
- 금융 시장에서 가속도는 노이즈에 묻힘
- 복잡도 vs 성능 트레이드오프 최적점

---

## 5. 알고리즘

### 5.1 Prediction Step

```
x_pred = F @ x
P_pred = F @ P @ F^T + Q
```

### 5.2 Update Step

```
y = z - H @ x_pred          # Innovation (예측 오차)
S = H @ P_pred @ H^T + R    # Innovation covariance
K = P_pred @ H^T @ S^(-1)   # Kalman gain

x = x_pred + K @ y          # 상태 업데이트
P = (I - K @ H) @ P_pred    # 공분산 업데이트
```

### 5.3 의사 코드

```python
def kalman_filter(prices, config):
    x = [prices[0], 0]  # [trend, velocity]
    P = [[initial_p, 0], [0, initial_p]]

    for t in range(len(prices)):
        # Adaptive noise estimation
        R_t = estimate_R(prices, t, config)
        Q_t = estimate_Q(prices, velocities, t, config)

        # Prediction
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q_matrix(Q_t)

        # Update
        z = prices[t]
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R_t
        K = P_pred @ H.T / S

        x = x_pred + K * y
        P = (I - K @ H) @ P_pred

        yield x, P, K
```

---

## 6. 포지션 사이징

### 6.1 이론적 근거

오차 공분산 $P_{t|t}$는 **현재 추세 추정의 불확실성**을 나타냄.

**켈리 베팅 유사 접근**:
$$\text{Size}_t \propto \frac{1}{\sqrt{P_{t|t}}}$$

### 6.2 해석

| 조건 | $P$ 값 | 포지션 |
|------|--------|--------|
| 추세 명확 | 낮음 | 크게 |
| 추세 불확실 | 높음 | 작게 |
| 급변 직후 | 급증 | 축소 |
| 안정 기간 | 낮음 | 확대 |

### 6.3 구현

```python
def position_size(uncertainty, min_unc, max_unc):
    # Log-scale 정규화 (분포 개선)
    log_unc = log(clamp(uncertainty, min_unc, max_unc))
    normalized = 1 - (log_unc - log_min) / (log_max - log_min)
    return clamp(normalized, 0, 1)
```

---

## 7. Typical Price

### 7.1 정의

$$\text{Typical Price} = \frac{H + L + C}{3}$$

### 7.2 효과

| 가격 유형 | Deviation Std | 노이즈 수준 |
|-----------|---------------|------------|
| Close | 55.04 | 기준 |
| Typical | 42.47 | -23% |

### 7.3 이유

- Close는 마지막 거래 가격으로 미시적 노이즈에 민감
- Typical Price는 바의 전체적인 가격 레벨을 대표
- High/Low 포함으로 극단값 영향 완화

---

## 8. Innovation 분석

### 8.1 최적 필터의 조건

이론적으로 최적의 칼만 필터는:
$$\text{Innovation} = z_t - \hat{z}_{t|t-1} \sim \text{White Noise}$$

### 8.2 적응형 필터의 특성

**실제 결과**:
- ACF(1) ≈ 0.65 (자기상관 존재)
- Ljung-Box: 백색 잡음 기각

**해석**:
- 적응형 필터는 의도적으로 스무딩
- Innovation에 구조가 남는 것은 트레이드오프
- ACF < 0.8이면 허용 가능한 수준

---

## 9. 한계점

### 9.1 모델 가정

1. **선형성**: 비선형 동학 무시
2. **가우시안 잡음**: Fat-tail 분포 무시
3. **정상성**: 구조 변화 즉시 반영 어려움

### 9.2 금융 시장 특성

- **점프 (Jump)**: 급격한 가격 변동 시 지연
- **레짐 전환**: 근본적 변화 감지 한계
- **유동성 변화**: 호가 스프레드 미반영

### 9.3 완화 방법

| 한계점 | 완화 방법 |
|--------|----------|
| 비선형 | Extended/Unscented Kalman Filter |
| Fat-tail | Student-t 분포 적용 |
| 레짐 전환 | HMM-Kalman 결합 |
| 점프 | Jump-Diffusion 모델 |

---

## 10. 참고 문헌

1. **Kalman, R.E.** (1960). "A New Approach to Linear Filtering and Prediction Problems"
2. **Alpha Architect**. "Noise-Adaptive Kalman Filter for Trading"
3. **Benhamou, E.** "Kalman Filter in Finance"
4. **Marcos Lopez de Prado**. "Advances in Financial Machine Learning"
5. **Nystrup et al.** (2020). "Greedy Online Classification of Persistent Market States"
