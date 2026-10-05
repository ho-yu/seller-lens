# Day02 · 딥러닝 실전 1장 1강 — logits / softmax / log-softmax

> 강의: 노션 `딥러닝 실전 & 프롬프트 엔지니어링` → 1장 1강
> 코드: `labs/day02/dl_ch01_softmax.py`
>
> ```bash
> uv run python labs/day02/dl_ch01_softmax.py
> ```

## 진행 방식

| 누가 | 무엇을 |
|---|---|
| **나** | `✍️` 표시된 계산 함수의 **본문만** 쓴다. 강의 개념이 담긴 1~4줄이다 |
| Claude | 나머지 전부: 예시 값, 출력(`print`, f-string, 구분선), assert, demo 함수, `main` |

1. 함수의 docstring("무엇을 위한 함수인가", Args, Returns)과 "작성 순서" 주석을 읽는다.
2. 본문을 쓰고 `TODO` 주석과 `raise` 줄을 지운다.
3. 실행해서 아래 **정답 출력**과 비교한다. assert가 통과하면 맞게 쓴 것이다. 작성 전인 키워드는 `⏸ 건너뜀`으로 표시되고 나머지는 그대로 실행된다.
4. **설명해 보기**에 내 말로 답한다.

막히면 함수 이름을 알려 준다. 바로 정답과 해설을 준다.

## 키워드 한눈에

| # | 키워드 | ✍️ 직접 작성 | 강의안 | 상태 |
|---|---|---|---|---|
| 1 | logit과 확률 | `stable_softmax` ✅, `predict_class` ✅, `is_valid_probs` ✅ | 1~3절 | ✅ 작성 완료 |
| 2 | max trick과 overflow | `naive_softmax` ✅ | 4~5절 | ✅ 작성 완료 |
| 3 | axis와 keepdims | `softmax_no_keepdims` ✅ | 5~6절, 9절 | ✅ 작성 완료 |
| 4 | log-softmax와 underflow | `stable_log_softmax` ✅ | 7절 | ✅ 작성 완료 |
| 5 | temperature (선택) | `softmax_with_temperature` ✅ | 강의 밖 | ✅ 작성 완료 |

---

## 1. logit과 확률 ✅

**핵심**
- **logit**은 모델이 클래스마다 낸 비교 점수다. 음수도 되고 합에 제한이 없다.
- **softmax**는 `exp`로 모두 양수로 만든 뒤 합으로 나눠 **합이 1인 확률**로 바꾼다.
- 크기 순서는 그대로다. 가장 큰 logit이 가장 큰 확률이 된다.
- 중요한 건 logit의 **차이**다. 모두에 같은 수를 더해도 결과는 같다.

**✍️ 직접 작성**

| 함수 | 무엇을 위한 함수인가 | 반환 |
|---|---|---|
| `predict_class(logits)` | 분류 모델의 마지막 단계: 점수 → 확률 → 가장 큰 확률의 클래스 | 클래스 이름 (`"부정"` 등) |
| `is_valid_probs(probs, axis)` | 확률 분포가 되려면 지켜야 하는 조건(0 이상, 1 이하, 합 1)을 코드로 | `True` / `False` |

**정답 출력**

```
=== 키워드 1. logit과 확률 ===
logits:    [ 2.   0.5 -1. ]
확률:      [0.7856 0.1753 0.0391]
확률의 합: 1.0000
+100:      [0.7856 0.1753 0.0391]  ← 차이가 그대로라 결과도 그대로
×2:        [0.9503 0.0473 0.0024]  ← 차이가 벌어져 1등 확률이 커짐
예측:      부정
is_valid_probs(확률)            → True
is_valid_probs([ 0.7  0.5 -0.2]) → False
```

**설명해 보기**
1. logit과 확률의 차이를 한 문장으로.
2. 100을 더하면 그대로인데 2를 곱하면 왜 달라질까?
3. `predict_class`에서 softmax를 빼고 logits에 바로 `np.argmax`를 해도 결과가 같을까? 그럼 softmax는 왜 필요할까?

<details>
<summary>모범 답</summary>

1. logit은 클래스별 원시 점수이고, 확률은 그 점수를 0~1 사이, 합이 1이 되도록 바꾼 값이다.
2. 더하기는 모든 logit 사이의 **차이를 그대로** 둔다. 곱하기는 차이를 2배로 **벌린다**. softmax는 차이만 보므로, 차이가 벌어지면 1등 확률이 더 커진다 (0.79 → 0.95). 이건 키워드 5 temperature와 같은 이야기다.
3. 같다. softmax는 크기 순서를 바꾸지 않기 때문이다. 하지만 "얼마나 확신하는지"는 확률이어야 알 수 있고(예: 0.79), 2강의 손실 계산과 학습에도 확률(log 확률)이 필요하다.

</details>

## 2. max trick과 overflow ✅

**핵심**
- `exp`는 아주 빨리 커진다. float64에서는 logit이 약 **709.8**을 넘으면 `exp`가 `inf`가 되고, `inf / inf = nan`이 된다. 이것이 **overflow**다.
- **max trick:** 계산 전에 최댓값을 빼면 가장 큰 값이 0이 되고 나머지는 음수가 된다. `exp`의 결과는 1 이하라서 터지지 않는다.
- 같은 수를 뺀 것이니 결과는 바뀌지 않는다 (키워드 1).
- float32에서는 경계가 약 **88.7**로 훨씬 가깝다. GPU 학습에서 쓰는 float16은 더 가깝다 (강의 밖).

**✍️ 직접 작성**

| 함수 | 무엇을 위한 함수인가 | 반환 |
|---|---|---|
| `naive_softmax(x, axis)` | max trick이 **없으면** 깨지는 걸 보여 줄 비교 대상. 서비스에는 쓰지 않는다 | `x`와 같은 shape의 확률 (큰 값이면 `nan`) |

**정답 출력**

```
=== 키워드 2. max trick과 overflow ===
보통 값 [ 2.   0.5 -1. ]: naive와 stable이 같다
큰 값 [1000. 1001. 1002.]
  naive:  [nan nan nan]
  stable: [0.09   0.2447 0.6652]  ← [0, 1, 2]의 softmax와 같다
exp가 inf가 되는 경계 (float64): 709.78
exp가 inf가 되는 경계 (float32): 88.72
```

**설명해 보기**
1. max trick이 softmax 결과를 바꾸지 않는 이유는?
2. 왜 최솟값이 아니라 **최댓값**을 뺄까?

<details>
<summary>모범 답</summary>

1. 모든 logit에서 같은 수를 빼면 분자와 분모가 똑같이 줄어 약분된다. 차이는 그대로라서 결과도 같다.
2. 최댓값을 빼야 **모든 값이 0 이하**가 되어 `exp`가 1을 넘지 않는다. 최솟값을 빼면 나머지는 오히려 양수로 남아서 큰 값은 여전히 overflow가 날 수 있다.

</details>

## 3. axis와 keepdims ✅

**핵심**
- 리뷰 N개 × 클래스 3개인 `(N, 3)` 배열에서는 **리뷰마다(행마다)** 합이 1이어야 한다. 클래스가 마지막 축이므로 `axis=-1`이다.
- `axis=0`으로 계산해도 **에러는 안 난다.** 결과가 틀렸을 뿐이다. "실행된다"와 "맞다"는 다르다.
- `sum`이나 `max`로 축을 줄이면 shape이 `(N, 3)` → `(N,)`이 된다. `keepdims=True`를 쓰면 `(N, 1)`이 남아, 빼기·나누기를 할 때 각 행에 그 행의 값이 맞게 적용된다(broadcasting).
- `keepdims`를 빼면 리뷰 수와 클래스 수가 다를 때는 에러가 나고, **같을 때는 에러 없이 조용히 틀린다.**

**✍️ 직접 작성**

| 함수 | 무엇을 위한 함수인가 | 반환 |
|---|---|---|
| `softmax_no_keepdims(x, axis)` | `keepdims=True`를 일부러 뺀 비교 대상. keepdims가 무엇을 막아 주는지 직접 본다 | 계산 결과 (2차원이면 틀릴 수 있음) |

**정답 출력**

```
=== 키워드 3. axis와 keepdims ===
axis=-1 확률:
[[0.7856 0.1753 0.0391]
 [0.3333 0.3333 0.3333]
 [0.0171 0.0466 0.9362]]
axis=-1 행마다 합: [1. 1. 1.]
axis=0  확률:
[[0.8438 0.4519 0.0171]
 [0.1142 0.2741 0.0466]
 [0.042  0.2741 0.9362]]
axis=0  행마다 합: [1.3128 0.4349 1.2523]  ← 에러 없이 틀림

--- keepdims ---
np.max(axis=-1) shape:                (3,)
np.max(axis=-1, keepdims=True) shape: (3, 1)
1차원 (3,): keepdims 없어도 결과가 같다
(3, 3) keepdims 없음:
[[0.3749 1.3912 0.0089]
 [0.0507 0.8438 0.0243]
 [0.0187 0.8438 0.4879]]
행마다 합: [1.7751 0.9188 1.3503]  ← 리뷰 수 = 클래스 수라 에러 없이 틀림
(2, 3) keepdims 없음: ValueError ← shape이 안 맞아 에러
  operands could not be broadcast together with shapes (2,3) (2,)
```

**설명해 보기**
1. 리뷰 486개, shape `(486, 3)`인 logits에 softmax를 적용할 축은? 축을 틀리면 에러가 날까?
2. `keepdims=True`는 무엇을 막아 줄까? `(3, 3)`에서 keepdims가 없으면 왜 에러 없이 틀릴까?

<details>
<summary>모범 답</summary>

1. `axis=-1`(= `axis=1`). 축을 틀려도 에러가 나지 않을 수 있어서, 리뷰마다 합이 1인지 확인하는 검사(`is_valid_probs`)가 필요하다.
2. 줄인 축을 길이 1로 남겨 `(N, 1)`로 만들어서, 각 행에 그 행의 최댓값·합이 적용되게 한다. keepdims가 없으면 `(3,)`이 되고, numpy는 이것을 **열 방향**으로 맞춰 붙인다. 그래서 1행에 1행의 값이 아니라 1·2·3행의 값이 하나씩 섞여 적용된다. 행 수와 열 수가 같으면 shape이 우연히 맞아 에러가 나지 않는다.

</details>

## 4. log-softmax와 underflow ✅

**핵심**
- 2강 Cross Entropy는 확률이 아니라 **log 확률**이 필요하다.
- softmax를 먼저 구하면 아주 작은 확률은 0이 된다(`exp(-1000)` → 0). 너무 작아서 0으로 뭉개지는 것이 **underflow**다. 그다음 `log(0)`은 `-inf`가 된다.
- **log-softmax**는 공식을 log로 풀어 쓴 `x_i - log(Σ exp(x_j))`를 바로 계산한다. 작은 확률을 0으로 만들지 않으므로 `-1000`이 정확하게 나온다.
- 그래서 2강의 Cross Entropy(그리고 PyTorch의 `CrossEntropyLoss`)는 확률이 아니라 **logits를 직접** 받는다.

**✍️ 직접 작성**

| 함수 | 무엇을 위한 함수인가 | 반환 |
|---|---|---|
| `stable_log_softmax(x, axis)` | 2강 Cross Entropy에 넣을 log 확률을 underflow 없이 구한다. **2강에서 다시 쓴다** | `x`와 같은 shape의 log 확률 (0 이하) |

**정답 출력**

```
=== 키워드 4. log-softmax와 underflow ===
log 확률:      [-0.3715 -1.3715 -2.8715]
exp(log 확률): [0.6897 0.2537 0.0566]  ← softmax와 같다
극단 값 [    0. -1000.]
  log(softmax): [  0. -inf]  ← 작은 확률이 0이 되어 -inf
  log-softmax:  [    0. -1000.]  ← 정확
```

**설명해 보기**
1. overflow와 underflow는 각각 어떤 상황이고, softmax / log-softmax의 어느 단계에서 생길까?
2. 2강 Cross Entropy는 왜 확률이 아니라 logits를 받을까?

<details>
<summary>모범 답</summary>

1. overflow는 값이 **너무 커서** `inf`가 되는 것으로, softmax의 `exp` 단계에서 큰 logit 때문에 생긴다 (max trick으로 해결). underflow는 값이 **너무 작아서** 0이 되는 것으로, softmax 결과에 `log`를 씌울 때 문제가 된다 (log-softmax로 해결).
2. 확률로 바꾼 뒤 log를 씌우면 작은 확률이 0 → `-inf`가 되어 loss가 `inf`가 될 수 있다. logits에서 log 확률을 바로 계산하면 이 문제가 없다.

</details>

## 5. temperature (선택, 강의 밖) ✅

**핵심**
- `softmax(logits / T)`. T로 나누는 건 logit 사이의 차이를 줄이거나 벌리는 것이다 (키워드 1의 "× 2"와 같은 원리. T = 0.5와 × 2의 결과가 같다).
- **T < 1:** 분포가 뾰족해진다. T → 0이면 1등만 1이 된다.
- **T > 1:** 분포가 평평해진다. T → ∞이면 모두 같은 확률이 된다.
- LLM은 다음 토큰의 logits에 이 계산을 한 뒤 확률대로 토큰을 뽑는다. API의 `temperature` 파라미터가 바로 이 T다.

**✍️ 직접 작성**

| 함수 | 무엇을 위한 함수인가 | 반환 |
|---|---|---|
| `softmax_with_temperature(logits, temperature)` | T로 분포를 뾰족하게/평평하게. LLM API `temperature`가 하는 일 | 확률 배열 |

**정답 출력**

```
=== 키워드 5. temperature ===
T= 0.1: [1. 0. 0.]
T= 0.5: [0.9503 0.0473 0.0024]
T= 1.0: [0.7856 0.1753 0.0391]
T= 2.0: [0.5898 0.2786 0.1316]
T=10.0: [0.3844 0.3308 0.2848]
```

**설명해 보기**
1. LLM으로 리뷰를 태깅할 때 같은 리뷰에 매번 같은 태그가 나오게 하려면 temperature를 어떻게 둘까? 답글 초안을 여러 버전 받고 싶을 때는?

<details>
<summary>모범 답</summary>

1. 태깅은 일관성이 중요하므로 temperature를 **낮게**(0에 가깝게) 둔다. 답글 초안처럼 다양한 표현이 필요하면 **조금 높인다.**

</details>

---

## 도전 과제 (선택)

강의안보다 한 걸음 더 나간 응용이다. 여유가 있을 때만 한다. 하지 않아도 Day02는 끝난다.

<details>
<summary>펼치기</summary>

- **overflow 경계 직접 찾기:** logit을 키워 가며 `naive_softmax`가 처음 `nan`이 되는 값을 반복문으로 찾는다. float32로도 해 본다.
- **리뷰 데이터 적용:** `data/interim/reviews_merged.csv`에서 키워드 개수로 `[부정, 중립, 긍정]` 점수를 만들고 softmax → 예측 → 정확도를 구한다. 전부 긍정으로 찍었을 때의 정확도와 비교한다.
- **라이브러리 대조:** `scipy.special.softmax`, `log_softmax`, `logsumexp`와 결과를 `np.allclose`로 비교한다.

</details>

## 다 하면

- 모든 키워드가 `⏸ 건너뜀` 없이 실행되고, 정답 출력과 같고, "설명해 보기"에 내 말로 답할 수 있으면 끝이다.
- "오늘 정리"라고 하면 답을 일지에 남기고 노션에 반영한다.
- 다음은 1장 2강(손실 함수, Cross Entropy). `stable_log_softmax`를 바로 이어서 쓴다.
