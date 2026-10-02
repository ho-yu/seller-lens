# Day01 - 리뷰 엑셀 3개를 개인정보 없이 합쳐 학습용 리뷰 486건을 만들었다

- 날짜: 2026-10-02
- 과목 / 장: 준비 / Step 0 사전 준비
- 기능: F1, 공통
- 상태: 완료
- git: `day01` 태그

## 🎯 오늘의 목표

- 이해할 개념: 병합·정제 단계에서 무엇을 남기고 무엇을 지우는지, Mac·Windows에서 같은 환경을 유지하는 방법
- 직접 해 볼 것: Step 0 (엑셀 3개 병합, 개인정보 제거, 제목+본문 `text`, 학습용 CSV 재생성)

## 🔑 복습 키워드

> 키워드 하나에 한 문장. 헷갈리면 VS Code에서 `Ctrl/Cmd + P` → 파일 이름으로 찾는다.

| 키워드 | 한 문장 | 파일 |
|---|---|---|
| 개인정보는 병합할 때 지운다 | `이름`, `리뷰어번호`는 병합 단계에서 바로 지워서 이후 어떤 파일에도 남지 않게 한다 | `src/00_merge_reviews.py` |
| 제목 + 본문 = `text` | 본문 없이 제목만 있는 리뷰가 22건이라 둘을 합쳐 쓰면 그만큼 데이터가 늘어난다 | `src/02_prepare_data.py` |
| interim vs processed | `interim`은 병합만 한 전체 리뷰(모든 평점), `processed`는 학습용(3점 제외, `label` 포함)이다 | `data/interim/reviews_merged.csv` |
| 클래스 불균형 | 부정 리뷰가 486건 중 47건(약 10%)이라 정확도만 보면 안 되고 부정 리뷰의 재현율·F1을 같이 본다 | `data/processed/reviews.csv` |
| 스마트 앱 컨트롤 | Windows가 평판이 쌓이지 않은 새 DLL(scipy 1.18.1)을 막아서, 직전 버전으로 제한해 해결했다 | `pyproject.toml` |
| 인터프리터는 실행 파일 | VS Code 인터프리터를 `.venv` 폴더로 지정하면 Pylance가 패키지를 못 찾아 "could not be resolved from source"가 뜬다 | `.vscode/settings.json` |
| import 묶음 정리 (I001) | 표준 라이브러리와 외부 패키지 import 사이에 빈 줄을 두는 규칙이고, 저장할 때 Ruff가 자동으로 맞춘다 | `.vscode/settings.json` |

## 💬 Claude와 나눈 질문

1. Q: `uv sync`를 했는데 환경이 제대로 됐나?
   - 이해한 것: 설치는 잠금 파일과 일치했지만 `import sklearn`이 실패했다. Windows 스마트 앱 컨트롤이 scipy 1.18.1의 DLL을 막은 것이고, scipy를 1.18 미만으로 제한해 해결했다.
2. Q: import 블록에 노란색 밑줄이 있는 이유는?
   - 이해한 것: Ruff의 I001 경고로, 표준 라이브러리와 외부 패키지 import 사이에 빈 줄이 없어서다. 저장할 때 자동 정리를 켰다.
3. Q: pandas만 노란 줄이 뜬다 ("could not be resolved from source")
   - 이해한 것: 인터프리터 설정이 `.venv` 폴더를 가리켜서 Pylance가 패키지를 못 찾았다. 설정을 지우고 인터프리터를 다시 고른 뒤 창을 다시 불러와 해결했다.
4. Q: Step 0이 학습 내용인가?
   - 이해한 것: 아니다. 1과목 실습에 쓸 데이터를 만드는 준비 작업이고, 같은 주제는 데이터 수집 & API 4장에서 정식으로 배운다.
5. Q: 맥에서도 동일하게 작업하려면 전부 커밋해야 하나?
   - 이해한 것: 그렇다. 코드, 문서, 데이터, `pyproject.toml`·`uv.lock`, `.vscode/` 설정까지 커밋한다. `.venv`는 각 기기에서 `uv sync`로 만들고, `.env` 같은 비밀값은 커밋하지 않는다.

## 🧪 실습 (labs/day01/)

오늘은 준비 작업이라 실습 파일 없이 `src/`에서 바로 진행했다.

## 🔧 서비스에 반영한 것

- `src/00_merge_reviews.py`: `03_merge_reviews.py`를 가장 먼저 실행되도록 이름을 바꾸고 완성. 엑셀 3개 병합, `product_id` 추가, 필수 컬럼 확인, `이름`·`리뷰어번호` 삭제 → `data/interim/reviews_merged.csv`
- `src/01_inspect_data.py`, `src/02_1_check_filtering.py`, `src/02_2_check_text_columns.py`: 입력을 병합 결과로 변경
- `src/02_prepare_data.py`: 제목+본문 `text` 사용, 상품·작성일·옵션 컬럼 유지 → `data/processed/reviews.csv` (486건)
- `pyproject.toml`, `uv.lock`: `scipy<1.18` 제약
- `.vscode/settings.json`: 저장할 때 import 정리, 인터프리터 경로 설정 삭제
- `.claude/skills/day-wrap/SKILL.md`: 파일 이름은 백틱으로 감싸는 규칙

## ✅ 검증

| 명령 | 결과 |
|---|---|
| `uv lock` → `uv sync` | ✅ scipy 1.18.1 → 1.17.1 |
| pandas·numpy·scipy·scikit-learn·openpyxl·matplotlib import | ✅ 모두 성공 |
| `uv run python src/00_merge_reviews.py` | ✅ 1,359행 병합, 개인정보 컬럼 없음 |
| `01`, `02_1`, `02_2` 실행 | ✅ 모두 정상 종료 |
| `uv run python src/02_prepare_data.py` | ✅ 486건 (긍정 439, 부정 47) |
| 결과 검산 | ✅ 빈 `text` 0, 중복 0, 3점 0, `label`과 평점 일치 |
| `uv run ruff check src` | ✅ All checks passed |

## 🛠 문제 → 원인 → 해결

1. **문제**: `import sklearn`이 "애플리케이션 제어 정책에서 이 파일을 차단했습니다"로 실패
   - 원인: Windows 스마트 앱 컨트롤이 평판이 없는 scipy 1.18.1의 DLL을 차단
   - 해결: `pyproject.toml`에 `[tool.uv] constraint-dependencies = ["scipy<1.18"]`
2. **문제**: `Import "pandas" could not be resolved from source`
   - 원인: `python.defaultInterpreterPath`가 `.venv` 폴더를 가리켜 Pylance가 패키지 경로를 못 찾음
   - 해결: 설정 삭제 → Python: Select Interpreter로 `.venv` 선택 → Reload Window
3. **문제**: import 블록 노란 밑줄
   - 원인: Ruff I001 (표준 라이브러리 / 외부 패키지 사이 빈 줄 없음)
   - 해결: 저장할 때 `source.organizeImports.ruff` 실행
4. **문제**: 병합 스크립트가 `03`이라 실행 순서와 번호가 맞지 않음
   - 원인: 병합이 다른 스크립트보다 먼저 실행돼야 함
   - 해결: `00_merge_reviews.py`로 이름 변경

## ⏭ 다음에 할 것

- [ ] Mac에서 `git clone` → `uv sync` → 스크립트 실행 확인
- [ ] 딥러닝 실전 1장: logits/softmax, Cross Entropy를 NumPy로 구현 (`labs/day02/`)
- [ ] 평가할 때 부정 리뷰의 재현율·F1을 같이 보기
