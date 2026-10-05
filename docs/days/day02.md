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

1. Q: `def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:`는 무슨 뜻인가?
   - 이해한 것: `stable_`은 수치적으로 안전하게 계산한다는 뜻이고, 타입 힌트는 실행에 영향이 없다. `axis`는 합이 1이 되어야 하는 축이고, 결과는 입력과 같은 shape다.
2. Q: 과제 1은 강의안을 전부 읽어야 하나? 1장인가?
   - 이해한 것: 1장 1강만 보면 된다. 과제 1은 강의안 3~6절(공식, max trick, keepdims, axis)과 8절 연습 문제에 거의 다 있다.
3. Q: 0단계(numpy 실험)는 어디서 어떻게 실행하나?
   - 이해한 것: 프로젝트 루트에서 `uv run python`으로 REPL을 열거나 Colab을 쓴다. Colab은 `Shift+Enter`로 실행하고, 셀의 마지막 줄만 자동으로 출력된다.
4. Q: 강의안만 보고는 코드 작성이 불가능한 것 아닌가?
   - 이해한 것: 과제 1은 강의안으로 가능했다. 막혔던 건 과제 시트가 강의안의 어느 절을 보라고 연결해 주지 않았고, 강의 밖 응용(경계값 찾기, 리뷰 데이터 적용 등)이 필수처럼 섞여 있었기 때문이다.
5. Q: 학습 방식을 어떻게 바꿀까?
   - 이해한 것: 강마다 키워드 3~5개를 정하고, 개념이 담긴 함수만 직접 작성한 뒤 내 말로 설명할 수 있으면 끝이다. 강의 밖 응용은 선택이다.
6. Q: 이 방식으로 해도 서비스 개발에 지장이 없나? 지금 코드도 나중에 쓰이나?
   - 이해한 것: 서비스 코드는 주로 LLM API, pandas, RAG를 쓰므로 지장이 없다. softmax 함수는 1~2장 분류기에 다시 쓰이고, 그 분류기의 기준선 정확도가 6장에서 LLM 분류와 비교된다. `src/`에서는 핵심 로직은 직접 쓰고, 연결 코드는 Claude가 쓰고 설명한다.
7. Q: 예시 logits를 하드코딩하라는 건가, 공통 함수를 쓰라는 건가?
   - 이해한 것: 계산 함수는 인자 `x`로 받아 `return`하는 공통 부품이고, demo 함수는 예시 값을 직접 적어 계산 함수를 호출해 보는 확인용이다. 아직 logits를 내는 모델이 없어서 예시 점수를 손으로 넣는다.
8. Q: demo는 왜 있나?
   - 이해한 것: 내가 쓴 개념 함수를 예시 값으로 실행해 보고, 개념이 드러나는 비교(naive vs stable, axis=-1 vs 0 등)를 출력하고, assert로 맞게 썼는지 채점하는 실험 장치다.
9. Q: (설명해 보기, 키워드 1) logit과 확률, `+100`과 `×2`, argmax와 softmax
   - 내 답: logit은 모델이 처음 낸 값이다. +100과 ×2가 왜 다른지는 몰랐다. argmax만 해도 결과는 같다.
   - 이해한 것: softmax는 logit의 값이 아니라 차이를 본다. 더하기는 차이를 그대로 두고 곱하기는 벌린다. 예측 클래스는 같아도 확률이 있어야 확신 정도(0.79 vs 0.51)를 알 수 있고 학습할 수 있다. axis·keepdims는 softmax를 계산하는 방법이지 softmax가 필요한 이유가 아니다.
10. Q: (설명해 보기, 키워드 2) max trick이 결과를 바꾸지 않는 이유, 최댓값을 빼는 이유
    - 내 답: 최댓값을 빼기 때문에. 숫자가 크거나 차이가 크면 최솟값을 빼는 건 의미가 없다.
    - 이해한 것: 모든 값에서 같은 수를 빼면 차이가 그대로라 결과가 같다. 최댓값을 빼야 모든 값이 0 이하가 되어 `exp`가 1을 넘지 않는다 (`[-1000, 0, 1000]`에서 최솟값을 빼면 `exp(2000)` = `inf`).
11. Q: (설명해 보기, 키워드 3) `(486, 3)`의 축, keepdims가 막아 주는 것
    - 내 답: y축, overflow 에러 가능성. keepdims는 연산 결과의 차원을 유지하는 옵션인데 왜 에러 없이 틀리는지 몰랐다.
    - 이해한 것: `axis=-1`은 클래스 3개가 있는 마지막 축이고, 축을 틀리면 overflow가 아니라 에러 없이 틀린 결과(행마다 합 ≠ 1)가 나온다. keepdims 없이 `(3,)`을 빼면 모든 행에 같은 줄이 복사되고, 리뷰 수와 클래스 수가 우연히 같으면 numpy가 shape이 맞다고 보고 에러를 내지 않는다.
12. Q: (설명해 보기, 키워드 4) overflow와 underflow, Cross Entropy가 logits를 받는 이유
    - 내 답: overflow는 숫자가 너무 커진 것이고 underflow는 그 반대일 것이다. 지수·로그에서 생긴다. Cross Entropy는 아직 모른다.
    - 이해한 것: underflow는 값이 0에 너무 가까워 0이 되는 것이다. 둘 다 `exp`에서 생기고, underflow는 `log`를 씌울 때 `-inf`로 드러난다. Cross Entropy는 `-log(정답 확률)`이라 확률이 0이 되면 loss가 `inf`가 되므로, logits에서 log-softmax로 바로 계산한다 (2강).
13. Q: (설명해 보기, 키워드 5) 태깅과 답글 초안의 temperature
    - 내 답: 태깅은 낮게, 답글 초안은 높게. 이유는 몰랐다.
    - 이해한 것: LLM은 확률대로 단어를 뽑는다. T가 낮으면 1등만 나와 같은 리뷰에 매번 같은 태그가 나오고, 조금 높이면 2·3등 표현도 뽑혀 초안이 다양해진다. 너무 높으면 엉뚱한 단어가 나온다.

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

1. **문제**: 과제 1을 어떻게 시작할지 감이 오지 않음
   - 원인: 과제 시트가 강의안의 어느 절을 보면 되는지 알려 주지 않았음
   - 해결: 키워드마다 강의안 절을 적고, 강의 밖 내용은 "강의 밖"으로 표시
2. **문제**: 너무 어려워서 진행이 막힘
   - 원인: 강의 밖 응용(경계값 찾기, float32 비교, 리뷰 데이터 적용, scipy 대조)까지 전부 필수였음
   - 해결: 강의안 범위만 필수로 하고, 강의 밖 응용은 "선택" 또는 "도전 과제"로 접어 둠
3. **문제**: 방식을 바꾸면서 Claude가 코드를 대신 쓰는 쪽으로 너무 많이 바꿈
   - 원인: "개념 이해 중심"을 "코드는 읽기만"으로 잘못 해석
   - 해결: Claude가 채운 함수를 다시 `TODO` 뼈대로 돌림
4. **문제**: "강의안 보고 작성"만으로는 무엇을 어떻게 쓸지 알 수 없었음
   - 원인: 함수마다 목적, 반환값, 쓰는 순서가 없었고, 하드코딩할지 공통 함수를 쓸지도 불분명했음
   - 해결: 함수마다 docstring에 "무엇을 위한 함수인가"·Args·Returns를 넣고, 작성 순서를 줄 모양 그대로 적고 개념 부분만 `___` 빈칸으로 남김
5. **문제**: 이름·구분선·f-string까지 쓰느라 강의 개념에 집중하지 못함
   - 원인: demo 함수(출력, assert)까지 사용자가 쓰는 구조였음
   - 해결: 사용자는 개념이 담긴 `✍️` 계산 함수 본문만 쓰고, demo와 `main`은 Claude가 완성해 둠. 작성 전인 키워드는 `⏸ 건너뜀`으로 넘김
6. **문제**: `predict_class`에서 `np.exp(logits)`를 직접 계산함
   - 원인: "stable_softmax로 확률로 바꾼다"는 주석이 "이미 만든 함수를 호출하라"는 뜻으로 읽히지 않았음
   - 해결: `probs = stable_softmax(logits)`로 호출. 이후 주석을 `변수 = 함수(___)` 모양으로 바꿈
7. **문제**: `is_valid_probs`가 올바른 확률에도 `False`를 냄
   - 원인: `np.allclose(np.sum(probs), probs)`로 합을 각 확률과 비교함
   - 해결: 비교 대상을 `1.0`으로 바꿈. demo의 assert가 버그를 잡아 줌
8. **문제**: temperature 결과가 거꾸로 나오고 예측도 중립으로 바뀜
   - 원인: `temperature / logits`로 나누는 순서가 반대였음 (logits가 역수가 되어 순서가 뒤바뀜)
   - 해결: `logits / temperature`로 고침

## ⏭ 다음에 할 것

- [ ] 1장 2강: 손실 함수와 Cross Entropy. `stable_log_softmax`를 이어서 쓴다
- [ ] 2강 과제도 `✍️` 개념 함수 + 완성된 demo 구조로 만든다
