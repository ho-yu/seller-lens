# 코딩 컨벤션

> sellerlens의 Python 코드 작성 규칙. 기준은 [PEP 8](https://peps.python.org/pep-0008/) (파이썬 공식 스타일 가이드)이고, 모양(줄바꿈, 따옴표, import 순서)은 Ruff가 자동으로 맞춘다.
> 예시 파일: `labs/day02/dl_ch01_softmax.py`

## 1. 파일 구성 순서

```
① 모듈 설명 (""" ... """)   이 파일이 뭘 하는지, 어떻게 실행하는지
② import                    표준 라이브러리 → (빈 줄) → 외부 패키지 → (빈 줄) → 내 코드
③ 상수                       경로, 설정값, 키워드 목록
④ 계산 함수                  다시 쓰는 부품
⑤ 실행·검증 함수             부품을 호출하고 결과를 확인 (labs에서는 키워드 하나에 demo 함수 하나)
⑥ main()                     ⑤를 순서대로 실행
⑦ if __name__ == "__main__": main()
```

- **⑦이 필요한 이유:** 파일을 직접 실행할 때만 `main()`이 돈다. 다른 파일에서 `import`할 때는 돌지 않으므로, 함수만 가져다 쓸 수 있다.
- **④와 ⑤를 나누는 이유:** 계산 함수에는 `print`나 `assert`를 넣지 않고 계산만 한다. 검증과 출력은 ⑤가 맡는다. 그래야 계산 함수를 다른 곳에서 그대로 재사용할 수 있다.
- **import는 파일 맨 위에만 둔다.** 함수 안에서 import하지 않는다.

## 2. 이름

| 대상 | 규칙 | 예 |
|---|---|---|
| 함수 · 변수 | `snake_case` | `stable_softmax`, `exp_values` |
| 상수 | `UPPER_SNAKE_CASE`, 파일 맨 위 | `REVIEWS_PATH`, `TEMPERATURES` |
| 클래스 | `CapWords` | `ReviewTag` |
| 내부용 (밖에서 쓰지 말 것) | 앞에 `_` | `_clean_text` |
| 파일 | `snake_case.py` | `dl_ch01_softmax.py` |

- 무엇인지 드러나게 짓는다. `a`, `tmp`, `data2` 대신 `logits`, `probs`, `reviews`.
- 반복문 인덱스(`i`), 수식 기호(`x`, `y`)처럼 관례가 있는 짧은 이름은 괜찮다.
- 숫자를 코드 중간에 바로 쓰지 말고 상수로 뺀다 (예: `0.1` 대신 `TEMPERATURES`).

## 3. 함수

```python
def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """logits를 합이 1인 확률로 바꾼다 (max trick 적용).

    Args:
        x: logits. 1차원 (C,) 또는 2차원 (N, C).
        axis: 확률의 합이 1이 되어야 하는 축.

    Returns:
        x와 같은 shape의 확률 배열.
    """
```

- **타입 힌트**: 인자와 반환값에 붙인다 (`x: np.ndarray`, `-> np.ndarray`). 실행에는 영향이 없지만 VS Code 자동완성과 오류 표시가 좋아진다. 반환값이 없으면 `-> None`.
- **docstring**: 첫 줄에 한 문장 요약, 필요하면 `Args:` / `Returns:` (Google 스타일). 배열을 다루면 **shape**를 꼭 적는다.
- **기본값 있는 인자**: `axis: int = -1`. 기본값으로 리스트·딕셔너리처럼 바뀔 수 있는 객체는 쓰지 않는다 (함수를 부를 때마다 같은 객체가 공유되는 버그가 생긴다).
- **한 함수는 한 가지 일만.** 길어지면 나눈다.
- **아직 안 만든 함수**는 `raise NotImplementedError`와 `# TODO(...)`로 표시하고, 구현하면 지운다.

## 4. 경로

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 파일 위치에서 루트까지 올라간 횟수
REVIEWS_PATH = PROJECT_ROOT / "data" / "interim" / "reviews_merged.csv"
```

- 문자열 경로(`"data/..."`) 대신 `pathlib.Path`를 쓴다. `/`로 이어 붙이면 Mac·Windows 모두에서 동작한다.
- **파일 자신의 위치를 기준으로** 루트를 찾는다. 그러면 어느 폴더에서 실행해도 같은 파일을 읽는다.
  - `labs/dayNN/파일.py` → `parents[2]`
  - `src/파일.py` → `parents[1]`
- Day01까지의 `src/` 스크립트는 실행한 위치 기준 경로(`Path("data/...")`)라 프로젝트 루트에서만 동작한다. 새로 쓰는 코드는 위 방식을 따른다.

## 5. 검증: `assert` vs `raise`

| 용도 | 쓰는 것 | 예 |
|---|---|---|
| 개발 중 "내 코드가 맞나" 확인 | `assert 조건, "실패 시 메시지"` | softmax 합이 1인지 |
| 잘못된 입력 막기 (실행 중 언제든) | `if ...: raise ValueError("...")` | shape이 맞지 않으면 에러 |

- `python -O`로 실행하면 `assert`는 통째로 무시된다. 그래서 입력 검사를 `assert`로 하면 안 된다.
- `assert`에는 실패 메시지를 붙이면 디버깅이 쉽다: `assert np.isclose(s, 1.0), f"합이 1이 아님: {s}"`
- 실수(float)는 `==`로 비교하지 않는다. `np.isclose`(값 하나), `np.allclose`(배열)를 쓴다.
- labs의 assert는 나중에 `src/`로 옮길 때 pytest 테스트 파일(`tests/`)로 옮긴다.

## 6. 출력

- 결과를 확인할 때는 무엇을 출력하는지 같이 적는다: `print(f"정확도: {accuracy:.3f}")`
- 과제·단계가 여러 개면 구분선을 둔다: `print("\n=== 과제 2 ===")`
- 소수점 자릿수는 f-string 형식으로 맞춘다 (`:.3f`, 퍼센트는 `:.1%`).

## 7. 도구

| 언제 | 무엇 | 명령 |
|---|---|---|
| 저장할 때 | import 정렬 (자동) | VS Code 설정에 들어 있음 |
| 수시로 | 스타일·실수 검사 | `uv run ruff check <경로>` |
| 정리하고 싶을 때 | 줄바꿈·따옴표 자동 정리 | `uv run ruff format <경로>` |
| 실행 | 프로젝트 환경으로 실행 | `uv run python <파일>` |

- 모양은 도구에 맡기고, **이름과 구조**는 사람이 신경 쓴다.
- 한 줄 길이 100자, 공백 4칸 들여쓰기, 큰따옴표 `"` (Ruff 설정: `pyproject.toml`).
- 노란 밑줄이 뜨면 마우스를 올려 메시지를 읽는다. Ruff 경고면 규칙 코드(예: `I001`)가 함께 나온다.

## 8. 실습 파일 (`labs/dayNN/`)

- 파일 이름: `<과목>_chNN_<주제>.py` (과목: `dl`, `api`, `lc`, `rag`)
- 과제 설명은 같은 폴더의 `TASKS.md`, 실습 파일은 이 문서의 구성 순서를 따른다.
- 키워드 하나에 함수 하나 (`demo_...`), `main()`에서 순서대로 호출한다. Claude가 뼈대를 만들고, 강의안에 코드가 있는 부분은 사용자가 채운다 (`CLAUDE.md`).
- 파일 맨 위 모듈 설명에는 아래를 적는다. 파일을 열었을 때 무엇을 왜 만드는지, 지금 어디까지 했는지 바로 보이게 하기 위해서다.
  - **이 파일이 하는 일**: 무엇을 만들고 무엇을 실험하는지, 다음 강에서 어떻게 다시 쓰는지
  - **전체 흐름**: 데이터가 어떤 함수를 거쳐 어떤 shape으로 바뀌는지 한 줄 그림
  - **키워드 ↔ 코드**: 키워드마다 계산 함수와 `demo` 함수의 대응
  - **진행 상황**: 키워드별 체크박스, 지금 단계에 `◀ 지금`과 그 단계가 무엇을 하는지 한두 줄
  - 과제 설명 위치, 실행 명령
