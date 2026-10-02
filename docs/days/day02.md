# Day02 - softmax를 max trick과 keepdims로 안정적으로 구현하고, 학습 방식을 강의안 수준으로 정했다

- 날짜: 2026-10-02
- 과목 / 장: 딥러닝 실전 & 프롬프트 엔지니어링 / 1장 1강
- 기능: 공통
- 상태: 진행 중
- git: `day02` 태그 (마무리 전, 중간 커밋 `bc051d6`)

## 🎯 오늘의 목표

- 이해할 개념: logit과 확률, softmax, max trick(overflow), axis와 keepdims, log-softmax(underflow), temperature
- 직접 해 볼 것: 강의안 코드를 참고해 `stable_softmax`와 키워드별 `demo_` 함수 작성

## 🔑 복습 키워드

> 키워드 하나에 한 문장. 헷갈리면 VS Code에서 `Ctrl/Cmd + P` → 파일 이름으로 찾는다.

| 키워드 | 한 문장 | 파일 |
|---|---|---|
| logit과 확률 | logit은 클래스마다 낸 비교 점수(음수 가능, 합 제한 없음)이고, softmax는 그것을 0~1 사이, 합이 1인 확률로 바꾼다 | `labs/day02/TASKS.md` |
| softmax 세 단계 | `exp`로 모두 양수로 만들고 → 합을 구하고 → 합으로 나눈다. 크기 순서는 그대로 유지된다 | `labs/day02/dl_ch01_softmax.py` |
| max trick | 모든 logit에서 최댓값을 빼면 `exp`의 입력이 0 이하가 되어 overflow가 없고, 같은 수를 뺀 것이라 결과도 같다 | `labs/day02/dl_ch01_softmax.py` |
| axis=-1 | `(N, C)`에서는 리뷰마다(행마다) 합이 1이어야 하므로 클래스 축인 마지막 축을 쓴다. 축을 틀려도 에러는 안 난다 | `labs/day02/TASKS.md` |
| keepdims=True | 축을 줄인 결과를 `(N,)`이 아니라 `(N, 1)`로 남겨 행마다 맞게 broadcasting되게 한다. N과 C가 같으면 빠뜨려도 에러 없이 조용히 틀린다 | `labs/day02/dl_ch01_softmax.py` |
| 실수 비교는 allclose | 소수 계산 결과는 `== 1.0` 대신 `np.isclose`(값 하나), `np.allclose`(배열)로 "충분히 가까운지"를 본다 | `docs/CONVENTIONS.md` |

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
   - 이해한 것: 강마다 키워드 3~5개를 정하고, 강의안 코드를 참고해 직접 작성한 뒤 내 말로 설명할 수 있으면 끝이다. 강의 밖 응용은 선택이다.
6. Q: 이 방식으로 해도 서비스 개발에 지장이 없나? 지금 코드도 나중에 쓰이나?
   - 이해한 것: 서비스 코드는 주로 LLM API, pandas, RAG를 쓰므로 지장이 없다. softmax 함수는 1~2장 분류기에 다시 쓰이고, 그 분류기의 기준선 정확도가 6장에서 LLM 분류와 비교된다. `src/`에서는 핵심 로직은 직접 쓰고, 연결 코드는 Claude가 쓰고 설명한다.

## 🧪 실습 (labs/day02/)

| 파일 | 해 본 것 | 결과 |
|---|---|---|
| `labs/day02/dl_ch01_softmax.py` | `stable_softmax` 직접 구현 (max trick + `keepdims`) | ✅ 완료 |
| `labs/day02/dl_ch01_softmax.py` | 키워드별 `demo_` 함수 5개, `naive_softmax`, `stable_log_softmax` | 뼈대만 있음 (다음에 작성) |
| `labs/day02/TASKS.md` | 1장 1강 키워드 5개, 강의안 절 연결, 직접 작성할 것, 설명해 보기 | 키워드 1 진행 중 |

## 🔧 서비스에 반영한 것 (src/)

- `src/` 변경 없음
- `CLAUDE.md`: 작업 규칙. 학습 방식(강의안 수준으로 직접 작성하고 설명), 서비스 코드 역할 분담(핵심 로직은 사용자, 연결 코드는 Claude)
- `docs/CONVENTIONS.md`: 코딩 컨벤션(파일 구성, 이름, 함수, 경로, assert vs raise, 출력, 실습 파일 맨 위 가이드)
- `docs/CURRICULUM.md`, `README.md`, `labs/README.md`: 위 두 문서 연결, 진행 방식 문장 수정

## ✅ 검증

| 명령 | 결과 |
|---|---|
| `uv run python labs/day02/dl_ch01_softmax.py` | ✅ 에러 없이 실행 (`demo_` 함수가 뼈대라 출력 없음) |
| 임시 확인 코드로 `stable_softmax` 실행 | ✅ 확률의 합 1.00, `[1000, 1001, 1002]`에서 `nan` 없음 |
| `uv run ruff check labs/day02` | ✅ All checks passed |
| `uv run ruff format --check labs/day02` | ✅ 이미 정리됨 |

## 🛠 문제 → 원인 → 해결

1. **문제**: 과제 1을 어떻게 시작할지 감이 오지 않음
   - 원인: 과제 시트가 강의안의 어느 절을 보면 되는지 알려 주지 않았음
   - 해결: 키워드마다 강의안 절을 적고, 강의 밖 내용은 "강의 밖"으로 표시
2. **문제**: 너무 어려워서 진행이 막힘
   - 원인: 강의 밖 응용(경계값 찾기, float32 비교, 리뷰 데이터 적용, scipy 대조)까지 전부 필수였음
   - 해결: 강의안에 코드가 있는 부분만 직접 작성하고, 강의 밖 응용은 "선택" 또는 "도전 과제"로 접어 둠
3. **문제**: 방식을 바꾸면서 Claude가 코드를 대신 쓰는 쪽으로 너무 많이 바꿈
   - 원인: "개념 이해 중심"을 "코드는 읽기만"으로 잘못 해석
   - 해결: Claude가 채운 함수를 다시 `TODO` 뼈대로 돌리고, "강의안 코드를 참고해 직접 작성할 수 있는 수준"으로 규칙을 고침

## ⏭ 다음에 할 것

- [ ] 키워드 1: `demo_logit_vs_prob` 작성 (강의안 3절)
- [ ] 키워드 2: `naive_softmax`, `demo_overflow` (강의안 4절)
- [ ] 키워드 3: `demo_axis` (강의안 6, 9절)
- [ ] 키워드 4: `stable_log_softmax`, `demo_underflow` (강의안 7절)
- [ ] 키워드 5: `demo_temperature` (선택)
- [ ] 키워드마다 "설명해 보기"에 내 말로 답하고 이 일지에 남기기
