# 코딩 컨벤션

> 기준은 PEP 8. 줄바꿈·따옴표·import 순서는 Ruff가 맞춘다 (`uv run ruff format <경로>`).

## 파일 구성 순서

```
① 모듈 설명 (""" ... """)   무엇을 하는지 2~3줄 + 실행 명령
② import                    표준 라이브러리 → 외부 패키지 → 내 코드
③ 상수                       경로, 설정값
④ 계산 함수                  계산만 한다 (print, assert 없음)
⑤ demo / 실행 함수           ④를 호출하고 출력·확인
⑥ main()
⑦ if __name__ == "__main__": main()
```

- ⑦이 있어야 다른 파일에서 `import`할 때 `main()`이 돌지 않는다.
- import는 파일 맨 위에만 둔다.

## 이름

| 대상 | 규칙 | 예 |
|---|---|---|
| 함수 · 변수 | `snake_case` | `stable_softmax`, `reviews` |
| 상수 | `UPPER_SNAKE_CASE` | `REVIEWS_PATH` |
| 클래스 | `CapWords` | `ReviewTag` |
| 파일 | `snake_case.py` | `dl_ch01_softmax.py` |

- 무엇인지 드러나게 짓는다 (`data2` 대신 `reviews`). 수식 기호(`x`, `y`)와 반복 인덱스(`i`)는 괜찮다.

## 함수

- 인자와 반환값에 타입 힌트를 붙인다: `def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:`
- docstring은 한두 줄. 배열이면 shape를 적는다.
- 한 함수는 한 가지 일만.

## 경로

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/ 파일 기준. labs/dayNN/ 파일은 parents[2]
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.csv"
```

- 문자열 경로 대신 `pathlib.Path`를 쓴다. Mac·Windows 모두에서 동작한다.

## 검증

| 용도 | 쓰는 것 |
|---|---|
| 내 코드가 맞나 확인 (개발 중) | `assert 조건, "메시지"` |
| 잘못된 입력 막기 | `if ...: raise ValueError("...")` |

- 실수는 `==` 대신 `np.isclose` / `np.allclose`로 비교한다.

## 실습 파일 (`labs/dayNN/`)

- 이름: `<과목>_chNN_<주제>.py` (과목: `dl`, `api`, `lc`, `rag`). 과제 설명은 같은 폴더의 `TASKS.md`.
- 사용자가 쓰는 함수에는 `✍️`를 붙이고, 본문 자리에 "작성 순서" 주석(쪼갠 문장, 손 계산 숫자)과 `raise NotImplementedError("✍️ <함수> 작성 전")`를 둔다.
- 100줄 안팎으로 유지한다. 형식은 [`CLAUDE.md`](../CLAUDE.md)의 "파일은 짧게".
