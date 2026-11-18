# 기능 진행 현황

> 서비스 기능이 어느 Day에 걸쳐 만들어졌는지 기록한다. 하루를 마무리할 때 그날 일지의 `기능`과 함께 갱신한다.
> 기능 정의는 [ROADMAP.md](ROADMAP.md), 계획 순서는 [CURRICULUM.md](CURRICULUM.md)를 따른다.
> 그날 시점의 코드는 git 태그 `dayNN`, 그날 바뀐 코드는 `git diff day<N-1> day<N> -- src/`로 본다.

## 요약

| ID | 기능 | 상태 | 계획된 단계 | 관련 Day |
|---|---|---|---|---|
| 공통 | 환경·구조·API 서버·운영 품질 | 진행 중 | Day00 환경 → 2과목 5~8장 API 서버 → LangChain 8~9장 관측·오류 대응 | Day00, Day01, Day02, Day03, Day04, Day05, Day06 |
| F1 | 리뷰 통합·정제 | 진행 중 | Step 0 기본 → 2과목 4장 완성 → RAG 2장 metadata 확정 | Day01, Day06 |
| F2 | 리뷰 자동 태깅 | 예정 | 1과목 5~7장 → LangChain 3·5장 | |
| F3 | 대응 우선순위 큐 | 예정 | 1과목 7장 | |
| F4 | 리뷰 답글 초안 | 진행 중 | 1과목 5장 → LangChain 2~3장 | Day06 |
| F5 | 상품 리뷰 리포트 | 예정 | 1과목 8장 → LangChain 4장 | |
| F6 | 리뷰 근거 검색 | 예정 | LangChain 6장 v0 → RAG 4~7장 | |
| F7 | CS 문의 답변 초안 | 예정 | RAG 3·6장 | |
| F8 | 판매 소구점 추출 | 예정 | 1과목 10장 | |
| F9 | 셀러 어시스턴트 | 예정 | RAG 9장 | |

상태: `예정` → `진행 중` → `완료`. 완료 뒤에 다시 고치면 `진행 중`으로 돌리고 기록을 이어 쓴다.

## 기능별 기록

> 기능마다 `DayNN: 그날 무엇을 만들거나 바꿨는지` 한 줄씩, 순서대로 쌓는다.

### 공통
- Day00: uv 프로젝트(`pyproject.toml`, `uv.lock`), Mac·Windows 공통 설정(`.gitattributes`, `.vscode/`), 하루 기록 절차(`.claude/skills/day-wrap/`), 기능 추적(`docs/FEATURES.md`), README
- Day01: `scipy<1.18` 제약(스마트 앱 컨트롤 차단 회피), VS Code 저장 시 import 정리·인터프리터 경로 설정 삭제(`.vscode/settings.json`)
- Day02: 작업 규칙(`CLAUDE.md`: 사용자는 개념이 담긴 `✍️` 함수만 쓰고 나머지는 Claude가 쓰는 학습 방식, 과제 형식, `src/` 역할 분담), 코딩 컨벤션(`docs/CONVENTIONS.md`)
- Day03: 학습 방식을 필수 핵심 개념 12개만 직접 코딩하는 A·B·C 등급으로 변경(`CLAUDE.md`, `docs/CURRICULUM.md`), 1회성 데이터 확인 스크립트 3개 삭제(`src/`), CSV·엑셀 보기 설정과 추천 확장(`.vscode/`)
- Day04: Codex용 작업 규칙·마무리 절차 복사본(`AGENTS.md`, `.agents/skills/day-wrap/`)과 원본 동기화 규칙(`CLAUDE.md`), 기본 LLM으로 `qwen3:4b-instruct-2507-q4_K_M` 선택
- Day05: `openai` 패키지 추가(`pyproject.toml`, `uv.lock`), Ollama를 OpenAI SDK로 부르는 방법 실습(`labs/day05/`)
- Day06: 엑셀(`*.xlsx`, `*.xls`)과 `.idea/`를 `.gitignore`에 추가하고 원본 엑셀 3개의 git 추적 해제(`.gitignore`)

### F1 리뷰 통합·정제
- Day01: Step 0 기본 버전. 엑셀 3개 병합·`product_id`·개인정보 삭제(`src/00_merge_reviews.py` → `data/interim/reviews_merged.csv`), 제목+본문 `text`·학습용 486건(`src/02_prepare_data.py` → `data/processed/reviews.csv`)
- Day06: 컬럼 이름이 다른 새 엑셀 2개를 `rename_columns`로 맞춰 병합에 추가(`src/00_merge_reviews.py`), `reviews.csv`를 2,076건(부정 658건)으로 재생성

### F2 리뷰 자동 태깅

### F3 대응 우선순위 큐

### F4 리뷰 답글 초안
- Day06: 답글 프롬프트 V0·V1(고객 탓 금지)·V2(예시 1개)를 10번씩 재서 비교하고 V1 유지(`labs/day05/dl_ch05_llm_api.py`)

### F5 상품 리뷰 리포트

### F6 리뷰 근거 검색

### F7 CS 문의 답변 초안

### F8 판매 소구점 추출

### F9 셀러 어시스턴트
