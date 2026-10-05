# Day02 - softmax가 logit의 차이만 본다는 것과 max trick·keepdims·log-softmax·temperature를 개념 함수로 직접 구현했다

- 날짜: 2026-10-02 ~ 2026-10-05
- 과목 / 장: 딥러닝 실전 & 프롬프트 엔지니어링 / 1장 1강
- 기능: 공통
- 상태: 완료
- git: `day02` 태그

## 🎯 오늘의 목표

- 이해할 개념: logit과 확률, softmax, max trick(overflow), axis와 keepdims, log-softmax(underflow), temperature
- 직접 해 볼 것: 키워드마다 개념이 담긴 `✍️` 계산 함수의 본문을 직접 작성하고, demo 출력을 보며 "설명해 보기"에 답하기

## 🔑 복습 키워드

> 키워드 하나에 한 문장. 헷갈리면 VS Code에서 `Ctrl/Cmd + P` → 파일 이름으로 찾는다.

| 키워드 | 한 문장 | 파일 |
|---|---|---|
| logit과 확률 | logit은 모델이 클래스마다 낸 점수(음수 가능, 합 제한 없음)이고, 확률은 softmax로 바꾼 0~1 사이, 합이 1인 값이다 | `labs/day02/dl_ch01_softmax.py` |
| softmax는 차이만 본다 | 모든 logit에 같은 수를 더하거나 빼면 차이가 그대로라 결과도 같고, 곱하면 차이가 벌어져 1등 확률이 커진다 | `labs/day02/TASKS.md` |
| argmax와 확률 | 예측 클래스는 logits에 바로 argmax해도 같지만, 얼마나 확신하는지(0.79 vs 0.51)와 학습(loss)에는 확률이 필요하다 | `labs/day02/dl_ch01_softmax.py` |
| max trick | 최댓값을 빼면 모든 값이 0 이하가 되어 `exp`가 1을 넘지 않으므로 overflow(`inf` → `nan`)가 없다. 최솟값을 빼면 큰 값이 그대로 남는다 | `labs/day02/dl_ch01_softmax.py` |
| keepdims와 broadcasting | `keepdims` 없이 `(3, 3)`에서 행마다 최댓값 `(3,)`을 빼면 numpy가 그것을 가로 한 줄로 모든 행에 복사해, 길이가 우연히 맞아 에러 없이 조용히 틀린다 | `labs/day02/TASKS.md` |
| underflow와 log-softmax | 아주 작은 확률은 0으로 뭉개지고 `log(0)` = `-inf`가 된다. log-softmax는 나누기를 빼기로 바꿔 확률을 거치지 않고 정확한 log 확률을 낸다 | `labs/day02/dl_ch01_softmax.py` |
| temperature | logits를 T로 나누면 T가 작을수록 뾰족해지고(1등 독식) 클수록 평평해진다. LLM은 이 확률대로 단어를 뽑으므로 낮으면 일관성, 높으면 다양성이 생긴다 | `labs/day02/dl_ch01_softmax.py` |

## 💬 Claude와 나눈 질문

1. **Q. `def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:`에서 `stable`, `axis`, 반환값은 각각 무슨 뜻인가?**
   - A. `stable`은 큰 logit에서도 overflow 없이 수치적으로 안전하게 계산한다는 뜻이다(max trick). `axis`는 확률의 합이 1이 되어야 하는 축으로, 클래스가 마지막 축이면 `-1`이다. 반환값은 입력과 같은 shape의 확률 배열이다. 타입 힌트(`: np.ndarray`, `-> np.ndarray`)는 실행에는 영향이 없고 읽는 사람과 에디터를 위한 표시다.
2. **Q. logit과 확률은 무엇이 다른가?**
   - A. logit은 모델이 클래스마다 낸 원시 점수라서 음수도 되고 합에 제한이 없다. 확률은 이 점수를 softmax로 바꾼 값으로, 모두 0~1 사이이고 합이 1이다.
3. **Q. logits에 100을 더하면 softmax 결과가 그대로인데, 2를 곱하면 왜 달라지는가?**
   - A. softmax는 logit의 값 자체가 아니라 logit 사이의 **차이**만 본다. 모든 값에 같은 수를 더하면 차이가 그대로라 결과도 같다. 2를 곱하면 차이가 2배로 벌어져 1등 확률이 커진다 (`[2.0, 0.5, -1.0]`에서 1등 확률 0.79 → 0.95). temperature도 같은 원리다.
4. **Q. 예측 클래스는 logits에 바로 argmax를 해도 같은데, softmax는 왜 필요한가?**
   - A. softmax는 크기 순서를 바꾸지 않으므로 argmax 결과는 같다. 하지만 확률이 있어야 모델이 얼마나 확신하는지 알 수 있다. 예를 들어 둘 다 "부정"으로 예측해도 확률이 0.79인 리뷰와 0.51인 리뷰는 다르다. 또 학습에 쓰는 Cross Entropy loss가 정답 클래스의 (log) 확률로 계산되므로 확률 개념이 필요하다.
5. **Q. max trick이 softmax 결과를 바꾸지 않는 이유와, 최솟값이 아니라 최댓값을 빼는 이유는?**
   - A. 모든 logit에서 같은 수를 빼면 차이가 그대로라 결과가 같다 (분자·분모에 같은 `exp(-m)`이 곱해져 약분된다). 최댓값을 빼야 모든 값이 0 이하가 되어 `exp`가 1을 넘지 않으므로 어떤 입력에서도 overflow가 없다. 최솟값을 빼면 큰 값이 여전히 양수로 남는다. 예: `[-1000, 0, 1000]`에서 최솟값을 빼면 `[0, 1000, 2000]` → `exp(2000)` = `inf`.
6. **Q. 리뷰 486개, shape `(486, 3)`인 logits에 softmax를 적용할 축은? 축을 틀리면 어떻게 되는가?**
   - A. 클래스 3개가 있는 마지막 축 `axis=-1`(= `axis=1`)이다. `axis=0`으로 틀리면 에러가 나지 않고 리뷰 486개를 세로로 묶어 계산해 버린다. 열마다 합이 1이 되고 행(리뷰)마다 합은 1이 아니다. 실행된다고 맞는 것이 아니므로 행마다 합이 1인지 검사해야 한다.
7. **Q. `keepdims=True`는 무엇을 막아 주는가? 빼면 왜 에러 없이 틀릴 수 있는가?**
   - A. 축을 줄인 결과를 `(N,)`이 아니라 `(N, 1)`로 남겨, 빼기·나누기를 할 때 각 행에 그 행의 값이 적용되게 한다. 빼면 `(N,)`이 되는데, numpy는 shape을 오른쪽 끝부터 맞춰 보므로 `(N, C)`와 `(N,)`에서 `C`와 `N`을 비교한다. 둘이 같거나 한쪽이 1이면 에러 없이 계산한다. 그래서 `(3, 3)`에서는 `(3,)`이 가로 한 줄로 모든 행에 복사되어 조용히 틀리고, `(486, 3)`에서는 3 ≠ 486이라 에러가 난다. 리뷰가 하나인 `(1, 3)`은 에러도 없고 결과도 맞다.
8. **Q. overflow와 underflow는 각각 무엇이고, softmax 계산의 어디서 생기는가?**
   - A. overflow는 값이 너무 커서 `inf`가 되는 것으로, 큰 logit에 `exp`를 할 때 생긴다 (`exp(1000)` → `inf` → `inf / inf` = `nan`). max trick으로 막는다. underflow는 값이 0에 너무 가까워 0이 되는 것으로, 아주 작은 logit에 `exp`를 할 때 생기고 그 확률에 `log`를 씌우면 `log(0)` = `-inf`로 드러난다. log-softmax로 막는다.
9. **Q. Cross Entropy는 왜 확률이 아니라 logits를 받는가?**
   - A. Cross Entropy는 `-log(정답 클래스 확률)`이다. 확률을 먼저 구하면 아주 작은 확률이 underflow로 0이 되어 loss가 `inf`가 될 수 있다. logits에서 log-softmax(`shifted - log Σ exp(shifted)`)로 log 확률을 바로 계산하면 이 문제가 없다. 예: logits `[0, -1000]`, 정답이 두 번째일 때 확률을 거치면 loss `inf`, log-softmax로는 1000. PyTorch `CrossEntropyLoss`도 logits를 받는다.
10. **Q. LLM으로 리뷰를 태깅할 때와 답글 초안을 만들 때 temperature를 어떻게 두는가? 이유는?**
    - A. LLM은 다음 단어 후보의 logits를 T로 나눈 뒤 softmax 확률대로 단어를 뽑는다. T가 낮으면 분포가 뾰족해져 1등 후보만 나오므로 같은 리뷰에 매번 같은 태그가 나온다 → 태깅은 낮게(0에 가깝게). T를 조금 높이면 2·3등 표현도 뽑혀 초안이 다양해진다 → 답글 초안은 적당히 높게(예: 0.7~1.0). 너무 높으면 분포가 평평해져 엉뚱한 단어가 나온다. 단, 실제 API는 T = 0이어도 결과가 아주 가끔 달라질 수 있어서 정확도를 잴 때는 여러 번 돌려 확인한다.

## 🧪 실습 (labs/day02/)

| 파일 | 해 본 것 | 결과 |
|---|---|---|
| `labs/day02/dl_ch01_softmax.py` | `stable_softmax` 직접 구현 (max trick + `keepdims`) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | ✍️ `predict_class`, `is_valid_probs` (키워드 1) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | ✍️ `naive_softmax` (키워드 2) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | ✍️ `softmax_no_keepdims` (키워드 3) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | ✍️ `stable_log_softmax` (키워드 4, 2강에서 다시 씀) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | ✍️ `softmax_with_temperature` (키워드 5, 선택) | ✅ 완료 |
| `labs/day02/TASKS.md` | 키워드 5개, ✍️ 함수 표, 정답 출력, 설명해 보기 | ✅ 5개 모두 답함 |

## 🔧 서비스에 반영한 것 (src/)

- `src/` 변경 없음
- `CLAUDE.md`: 작업 규칙. 학습 방식(사용자는 개념이 담긴 `✍️` 함수 본문만 쓰고, 이름·출력·demo·assert는 Claude가 씀), 과제 형식(함수마다 목적·반환, 줄 모양과 `___` 빈칸이 있는 작성 순서, 실제로 돌려 본 정답 출력), 서비스 코드 역할 분담
- `docs/CONVENTIONS.md`: 코딩 컨벤션(파일 구성, 이름, 함수, 경로, assert vs raise, 출력, 실습 파일의 `✍️` 함수와 demo 구성)
- `docs/CURRICULUM.md`, `README.md`, `labs/README.md`: 위 두 문서 연결, 진행 방식 문장 수정

## ✅ 검증

| 명령 | 결과 |
|---|---|
| `uv run python labs/day02/dl_ch01_softmax.py` | ✅ 키워드 1~5 모두 `⏸ 건너뜀` 없이 실행, assert 전부 통과, `TASKS.md` 정답 출력과 같음 |
| `uv run ruff check labs/day02` | ✅ All checks passed |
| `uv run ruff format labs/day02` | ✅ 형식 정리 (`stable_log_softmax`의 쉼표 뒤 공백 1곳) |

## 🛠 문제 → 원인 → 해결

1. **문제**: `predict_class`에서 `np.exp(logits)`를 확률로 썼다
   - 원인: `exp`는 softmax 세 단계(exp → 합 → 나누기) 중 첫 단계일 뿐이다. `[2.0, 0.5, -1.0]`의 `exp`는 `[7.389, 1.649, 0.368]`로 합이 9.4라 확률이 아니다
   - 해결: 세 단계가 모두 들어 있는 `stable_softmax(logits)`를 호출했다
2. **문제**: `is_valid_probs`가 올바른 확률에도 `False`를 냈다
   - 원인: `np.allclose(np.sum(probs), probs)`로 "합"을 "각 확률"과 비교했다. 확률 분포의 조건은 "합이 1"이다
   - 해결: 비교 대상을 `1.0`으로 바꿨다. demo의 assert가 버그를 잡아 줬다
3. **문제**: temperature 결과가 거꾸로 나오고 예측도 중립으로 바뀌었다
   - 원인: `temperature / logits`로 나누는 순서가 반대였다. logits가 분모로 가면 역수가 되어 크기 순서가 뒤바뀐다 (`1 / [2.0, 0.5, -1.0]` = `[0.5, 2.0, -1.0]`)
   - 해결: `logits / temperature`로 고쳤다. T로 나누는 것은 logit 사이의 차이를 줄이거나 벌리는 것이다

## ⏭ 다음에 할 것

- [ ] 1장 2강: 손실 함수와 Cross Entropy. `stable_log_softmax`를 이어서 쓴다
- [ ] 2강 과제도 `✍️` 개념 함수 + 완성된 demo 구조로 만든다
