# 기능 진행 현황

> 서비스 기능이 어느 Day에 걸쳐 만들어졌는지 기록한다. 하루를 마무리할 때 그날 일지의 `기능`과 함께 갱신한다.
> 기능 정의는 [ROADMAP.md](ROADMAP.md), 계획 순서는 [CURRICULUM.md](CURRICULUM.md)를 따른다.
> 그날 시점의 코드는 git 태그 `dayNN`, 그날 바뀐 코드는 `git diff day<N-1> day<N> -- src/`로 본다.

## 요약

| ID | 기능 | 상태 | 계획된 단계 | 관련 Day |
|---|---|---|---|---|
| 공통 | 환경·구조·API 서버·운영 품질 | 진행 중 | Day00 환경 → 2과목 5~8장 API 서버 → LangChain 8~9장 관측·오류 대응 | Day00, Day01, Day02 |
| F1 | 리뷰 통합·정제 | 진행 중 | Step 0 기본 → 2과목 4장 완성 → RAG 2장 metadata 확정 | Day01 |
| F2 | 리뷰 자동 태깅 | 예정 | 1과목 5~7장 → LangChain 3·5장 | |
| F3 | 대응 우선순위 큐 | 예정 | 1과목 7장 | |
| F4 | 리뷰 답글 초안 | 예정 | 1과목 5장 → LangChain 2~3장 | |
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

### F1 리뷰 통합·정제
- Day01: Step 0 기본 버전. 엑셀 3개 병합·`product_id`·개인정보 삭제(`src/00_merge_reviews.py` → `data/interim/reviews_merged.csv`), 제목+본문 `text`·학습용 486건(`src/02_prepare_data.py` → `data/processed/reviews.csv`)

### F2 리뷰 자동 태깅

### F3 대응 우선순위 큐

### F4 리뷰 답글 초안

### F5 상품 리뷰 리포트

### F6 리뷰 근거 검색

### F7 CS 문의 답변 초안

### F8 판매 소구점 추출

### F9 셀러 어시스턴트
