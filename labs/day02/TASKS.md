# Day02 · 딥러닝 실전 1장 1강 — logits / softmax / log-softmax

> 강의: 노션 `딥러닝 실전 & 프롬프트 엔지니어링` → 1장 1강
> 방식: 키워드마다 **강의안 코드를 참고해 직접 작성**하고, 실행 결과를 보며 **내 말로 설명**할 수 있으면 끝 (`CLAUDE.md`).
> 강의안 코드를 그대로 복사하지 말고, 보면서 직접 타이핑하되 값과 이름은 sellerlens에 맞게 바꾼다 (클래스 = 부정 / 중립 / 긍정).
> 코드: `labs/day02/dl_ch01_softmax.py`의 `TODO` 자리. 작성한 뒤 실행한다.
>
> ```bash
> uv run python labs/day02/dl_ch01_softmax.py
> ```
>
> 값을 바꿔 보는 실험은 Colab에서 해도 된다. 필요한 함수를 셀에 복사해 쓰면 된다.

## 키워드 한눈에

| # | 키워드 | 강의안 | 코드 | 상태 |
|---|---|---|---|---|
| 1 | logit과 확률 | 1~3절 | `stable_softmax` ✅, `demo_logit_vs_prob` | ◀ 지금 |
| 2 | max trick과 overflow | 4~5절 | `naive_softmax`, `demo_overflow` | |
| 3 | axis와 keepdims | 5~6절, 9절 | `demo_axis` | |
| 4 | log-softmax와 underflow | 7절 + 강의 밖 | `stable_log_softmax`, `demo_underflow` | |
| 5 | temperature (부가) | 강의 밖 | `demo_temperature` | |

막히면: ① 표의 강의안 절을 다시 본다 → ② 그래도 안 되면 바로 물어본다. 정답과 해설을 준다.

---

## 1. logit과 확률 ◀ 지금

**핵심**
- **logit**은 모델이 클래스마다 낸 비교 점수다. 음수도 되고 합에 제한이 없다.
- **softmax**는 `exp`로 모두 양수로 만든 뒤 합으로 나눠 **합이 1인 확률**로 바꾼다.
- 크기 순서는 그대로다. 가장 큰 logit이 가장 큰 확률이 된다.
- 중요한 건 logit의 **차이**다. 모두에 같은 수를 더해도 결과는 같다.

**직접 작성** (`demo_logit_vs_prob`, 강의안 3절)
- `stable_softmax` ✅ 이미 작성함
- 리뷰 하나의 `[부정, 중립, 긍정]` logits를 만든다 (예: `[2.0, 0.5, -1.0]`).
- `stable_softmax`를 적용하고 logits, 확률, 확률의 합을 출력한다 (3절의 `print` 부분).
- 예측 클래스 이름을 출력한다: `CLASS_NAMES[np.argmax(probs)]`
- 3절 "가장 먼저 할 검증"의 assert 4개를 내 데이터로 쓴다 (shape, 0 이상, 1 이하, 합이 1).
- 응용 한 줄: `logits + 100`과 `logits * 2`의 softmax도 출력해 비교한다.

**출력에서 볼 것:** `logits + 100`은 결과가 그대로인데 `logits * 2`는 달라진다.

**설명해 보기**
1. logit과 확률의 차이를 한 문장으로.
2. 100을 더하면 그대로인데 2를 곱하면 왜 달라질까?

<details>
<summary>모범 답</summary>

1. logit은 클래스별 원시 점수이고, 확률은 그 점수를 0~1 사이, 합이 1이 되도록 바꾼 값이다.
2. 더하기는 모든 logit 사이의 **차이를 그대로** 둔다. 곱하기는 차이를 2배로 **벌린다**. softmax는 차이만 보므로, 차이가 벌어지면 1등 확률이 더 커진다 (0.79 → 0.95). 이건 키워드 5 temperature와 같은 이야기다.

</details>

## 2. max trick과 overflow

**핵심**
- `exp`는 아주 빨리 커진다. float64에서는 logit이 약 **709.8**을 넘으면 `exp`가 `inf`가 되고, `inf / inf = nan`이 된다. 이것이 **overflow**다.
- **max trick:** 계산 전에 최댓값을 빼면 가장 큰 값이 0이 되고 나머지는 음수가 된다. `exp`의 결과는 1 이하라서 터지지 않는다.
- 같은 수를 뺀 것이니 결과는 바뀌지 않는다 (키워드 1).
- float32에서는 경계가 약 **88.7**로 훨씬 가깝다. GPU 학습에서 쓰는 float16은 더 가깝다 (강의 밖).

**직접 작성** (`naive_softmax`, `demo_overflow`, 강의안 4절)
- `naive_softmax`: 3절의 세 단계(exp → 합 → 나누기)를 함수로 만든다. `stable_softmax`에서 max trick 줄만 뺀 모양이다.
- `demo_overflow`: `[1000.0, 1001.0, 1002.0]`에 naive와 stable을 각각 적용해 출력한다. naive는 4절처럼 `with np.errstate(over="ignore", invalid="ignore"):` 안에서 호출하면 경고 없이 결과만 볼 수 있다.
- 4절 끝의 assert: stable 결과에 `inf` / `nan`이 없는지 (`np.isfinite`).
- 선택 (강의 밖, 한 줄): `np.log(np.finfo(np.float64).max)`와 `np.float32`의 값을 출력해 본다. 이 숫자가 무슨 의미일까?

**설명해 보기**
1. max trick이 softmax 결과를 바꾸지 않는 이유는?
2. 왜 최솟값이 아니라 **최댓값**을 뺄까?

<details>
<summary>모범 답</summary>

1. 모든 logit에서 같은 수를 빼면 분자와 분모가 똑같이 줄어 약분된다. 차이는 그대로라서 결과도 같다.
2. 최댓값을 빼야 **모든 값이 0 이하**가 되어 `exp`가 1을 넘지 않는다. 최솟값을 빼면 나머지는 오히려 양수로 남아서 큰 값은 여전히 overflow가 날 수 있다.

</details>

## 3. axis와 keepdims

**핵심**
- 리뷰 N개 × 클래스 3개인 `(N, 3)` 배열에서는 **리뷰마다(행마다)** 합이 1이어야 한다. 클래스가 마지막 축이므로 `axis=-1`이다.
- `axis=0`으로 계산해도 **에러는 안 난다.** 결과가 틀렸을 뿐이다. "실행된다"와 "맞다"는 다르다.
- `sum`이나 `max`로 축을 줄이면 shape이 `(N, 3)` → `(N,)`이 된다. `keepdims=True`를 쓰면 `(N, 1)`이 남아, 나눌 때 각 행에 그 행의 값이 맞게 적용된다(broadcasting).
- 리뷰 수와 클래스 수가 같으면(3 × 3) `keepdims` 없이도 에러 없이 **조용히 틀린다.** 출력에서 확인할 수 있다.

**직접 작성** (`demo_axis`, 강의안 6절, 9절)
- 리뷰 3개 × `[부정, 중립, 긍정]`인 2차원 logits를 만든다 (6절의 `batch_logits` 모양).
- `axis=-1`과 `axis=0`으로 각각 softmax를 구하고, 각 행(리뷰)의 합을 출력한다 (9절).
- assert: `axis=-1` 결과의 행마다 합이 1인지 (`np.allclose`).
- 선택: `np.sum(logits, axis=-1)`과 `np.sum(logits, axis=-1, keepdims=True)`의 `.shape`를 출력해 비교한다.

**설명해 보기**
1. 리뷰 486개, shape `(486, 3)`인 logits에 softmax를 적용할 축은? 축을 틀리면 에러가 날까?
2. `keepdims=True`는 무엇을 막아 줄까?

<details>
<summary>모범 답</summary>

1. `axis=-1`(= `axis=1`). 축을 틀려도 에러가 나지 않을 수 있어서, 리뷰마다 합이 1인지 확인하는 검사가 필요하다.
2. 합계나 최댓값의 차원이 사라지는 것을 막아, 원래 배열과 연산할 때 **행마다** 맞게 broadcasting되게 한다.

</details>

## 4. log-softmax와 underflow

**핵심**
- 2강 Cross Entropy는 확률이 아니라 **log 확률**이 필요하다.
- softmax를 먼저 구하면 아주 작은 확률은 0이 된다(`exp(-1000)` → 0). 너무 작아서 0으로 뭉개지는 것이 **underflow**다. 그다음 `log(0)`은 `-inf`가 된다.
- **log-softmax**는 공식을 log로 풀어 쓴 `x_i - log(Σ exp(x_j))`를 바로 계산한다. 작은 확률을 0으로 만들지 않으므로 `-1000`이 정확하게 나온다.
- 그래서 2강의 Cross Entropy(그리고 PyTorch의 `CrossEntropyLoss`)는 확률이 아니라 **logits를 직접** 받는다.

**직접 작성** (`stable_log_softmax`, `demo_underflow`, 강의안 7절)
- `stable_log_softmax`: 7절의 함수를 참고해 작성한다.
- 7절의 assert: 보통 logits(예: `[2.0, 1.0, -0.5]`)에서 `np.exp(log_softmax)`와 `softmax`가 같은지 (`np.allclose`).
- 강의 밖 (짧음): logits `[0.0, -1000.0]`에 `np.log(stable_softmax(x))`와 `stable_log_softmax(x)`를 각각 출력한다. 앞쪽은 `with np.errstate(divide="ignore"):` 안에서 계산한다.

**출력에서 볼 것:** `np.log(softmax)`는 `-inf`, `stable_log_softmax`는 `-1000`이 나온다.

**설명해 보기**
1. overflow와 underflow는 각각 어떤 상황이고, softmax / log-softmax의 어느 단계에서 생길까?
2. 2강 Cross Entropy는 왜 확률이 아니라 logits를 받을까?

<details>
<summary>모범 답</summary>

1. overflow는 값이 **너무 커서** `inf`가 되는 것으로, softmax의 `exp` 단계에서 큰 logit 때문에 생긴다 (max trick으로 해결). underflow는 값이 **너무 작아서** 0이 되는 것으로, softmax 결과에 `log`를 씌울 때 문제가 된다 (log-softmax로 해결).
2. 확률로 바꾼 뒤 log를 씌우면 작은 확률이 0 → `-inf`가 되어 loss가 `inf`가 될 수 있다. logits에서 log 확률을 바로 계산하면 이 문제가 없다.

</details>

## 5. temperature (부가)

**핵심**
- `softmax(logits / T)`. T로 나누는 건 logit 사이의 차이를 줄이거나 벌리는 것이다 (키워드 1의 "× 2"와 같은 원리. 실제로 출력에서 T = 0.5와 × 2의 결과가 같다).
- **T < 1:** 분포가 뾰족해진다. T → 0이면 1등만 1이 된다.
- **T > 1:** 분포가 평평해진다. T → ∞이면 모두 같은 확률이 된다.
- LLM은 다음 토큰의 logits에 이 계산을 한 뒤 확률대로 토큰을 뽑는다. API의 `temperature` 파라미터가 바로 이 T다.

**직접 작성** (`demo_temperature`, 강의 밖, 선택)
- 키워드 1의 logits에 대해 `TEMPERATURES`를 반복하며 `stable_softmax(logits / T)`를 출력한다. 3~4줄이면 된다.

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

- **assert 더 쓰기:** argmax 위치가 logits와 확률에서 같은지, 상수를 더해도 결과가 같은지를 assert로 쓴다.
- **overflow 경계 직접 찾기:** logit을 키워 가며 `naive_softmax`가 처음 `nan`이 되는 값을 반복문으로 찾는다. float32로도 해 본다.
- **리뷰 데이터 적용:** `data/interim/reviews_merged.csv`에서 키워드 개수로 `[부정, 중립, 긍정]` 점수를 만들고 softmax → 예측 → 정확도를 구한다. 전부 긍정으로 찍었을 때의 정확도와 비교한다.
- **라이브러리 대조:** `scipy.special.softmax`, `log_softmax`, `logsumexp`와 결과를 `np.allclose`로 비교한다.

</details>

## 다 하면

- 키워드마다 "직접 작성"이 실행되고, "설명해 보기"에 내 말로 답할 수 있으면 끝이다.
- "오늘 정리"라고 하면 답을 일지에 남기고 노션에 반영한다.
- 다음은 1장 2강(손실 함수, Cross Entropy). 키워드 4가 바로 이어진다.
