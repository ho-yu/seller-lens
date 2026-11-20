# sellerlens 커리큘럼

> KANT 강의 4과목을 순서대로 듣고, 배운 것 중 **기능에 꼭 필요하고 실무에서 자주 쓰는 것**만 sellerlens에 직접 구현한다.
> 등급: **A** 직접 코딩 + `src/` 반영 · **B** Claude 코드를 실행해 보고 질문에 답하기 · **C** 강의만. 작업 규칙은 [`CLAUDE.md`](../CLAUDE.md). (개정: 2026-10-27, 2026-11-20 소싱 판단용으로 기능 변경 — [`PLAN.md`](PLAN.md))

## 필수 핵심 개념 12개 (A 등급)

| # | 개념 | 기능 | 강의 |
|---|---|---|---|
| 1 | pandas로 읽기·거르기·집계 | F1, F5 | 2과목 4장 |
| 2 | LLM API 호출, System Prompt, temperature | F2 | 1과목 5장 |
| 3 | Few-shot과 정확도 평가 | F2 | 1과목 6장 |
| 4 | JSON 출력과 Pydantic 검증 | F2 | 1과목 7장 |
| 5 | Tool Calling | F5, F9, F11 | 1과목 8장 |
| 6 | timeout·retry | 전체 | 1과목 10장 |
| 7 | FastAPI 엔드포인트와 요청·응답 스키마 | 전체 | 2과목 5~7장 |
| 8 | PromptTemplate과 LCEL 체인 | 전체 | 3과목 2~3·5장 |
| 9 | 임베딩과 코사인 유사도 | F6 | 4과목 4장 |
| 10 | 벡터DB 검색과 메타데이터 필터 | F6 | 4과목 5장 |
| 11 | 근거를 붙인 요약 (근거 없으면 쓰지 않기) | F5 | 4과목 6장 |
| 12 | 평가셋으로 개선 전후 비교 | 전체, F12 | 4과목 8장 |

## 사전 준비 ✅

엑셀 5개 병합, 개인정보 제거, 학습용 리뷰 2,076건 (`data/processed/reviews.csv`, Day01·Day05).

## 1과목. 딥러닝 실전 & 프롬프트

**끝나면:** 상품 리뷰 엑셀을 넣으면 태깅과 상품 리뷰 리포트 v0이 나오는 CLI. 1~4장은 이해만 하고 직접 코딩은 5장부터.

| 장 | 주제 | 등급 | 남길 것 / 할 것 |
|---|---|---|---|
| 1 | softmax, 손실 함수 | B | LLM은 다음 단어를 확률로 고른다, temperature (Day02~03) |
| 2 | 경사하강법, 옵티마이저 | C | "loss가 줄어드는 방향으로 조금씩 고친다" |
| 3 | Attention | C | 강의만 |
| 4 | Base/SFT/Instruct | C | instruct 모델을 쓴다. Ollama 모델 1개 고르기 |
| 5 | LLM API, System Prompt | **A** | 리뷰 1개 → 감성 분류 (Day05 답글 프롬프트 실습은 F4 제외로 기록만 남김) |
| 6 | Few-shot, 평가 | **A** | 평점 라벨로 정확도 측정 |
| 7 | JSON Schema, Pydantic | **A** | F2 태깅 스키마와 검증. 다음 태깅 작업 때 긴급도를 빼고 좋은 점·나쁜 점 유형으로 바꾼다 |
| 8 | Tool Calling | **A** | 집계 함수를 도구로. F5 리포트 v0 |
| 9 | 멀티모달 | C | 건너뛴다 |
| 10 | Retry, 캐싱, CLI 통합 | **A** | timeout·retry. CLI 뼈대는 Claude (엑셀 → 리포트) |

## 2과목. 데이터 수집 & API 서버

**끝나면:** 1과목 기능을 FastAPI로 쓰고, 엑셀 업로드와 일괄 처리를 한다.

| 장 | 주제 | 등급 | 남길 것 / 할 것 |
|---|---|---|---|
| 1 | 수집 방식, 리스크 | C | 강의만 |
| 2 | HTTP, 인증, 환경변수 | **A** | httpx로 Ollama 호출, `.env` |
| 3 | pagination, rate limit | B | 공휴일 API 예제 실행 |
| 4 | 정제, 본문/메타데이터 분리 | **A** | 리뷰 정제 파이프라인 (F1 완성) |
| 5 | FastAPI 구조 | **A** | `/health` |
| 6 | Pydantic 스키마, 등록/조회 API | **A** | `POST /reviews/upload`, `GET /reviews` |
| 7 | Query API, LLM 연동 | **A** | `/reviews/{id}/tag`, `/products/{id}/report` 등 1과목 기능 옮기기. F11 금액 계산 |
| 8 | 비동기, middleware | B | 일괄 태깅 (코드는 Claude) |

## 3과목. LangChain

**끝나면:** 같은 API가 체인으로 동작하고, 실패해도 멈추지 않는다.

| 장 | 주제 | 등급 | 남길 것 / 할 것 |
|---|---|---|---|
| 1 | 구조 비교 | C | 강의만 |
| 2 | PromptTemplate, Output Parser | **A** | 태깅·요약 프롬프트를 템플릿으로 |
| 3 | LCEL, invoke/batch/stream | **A** | 태깅 체인 |
| 4 | RunnableParallel | B | 좋은 점·나쁜 점 요약 병렬 |
| 5 | 구조화 출력, retry | **A** | `with_structured_output` |
| 6 | Document, Retriever | B | 근거 리뷰 찾기 v0 |
| 7 | 대화 기록 | B | 여러 턴 질의 |
| 8 | Streaming, Tracing | C | 강의만 |
| 9 | with_retry, Fallback | B | 모델 실패 시 대체 |
| 10 | 통합 | B | API 내부를 체인으로 교체 (Claude) |

## 4과목. RAG

**끝나면:** 리포트 항목마다 근거 리뷰가 붙고, 조건으로 근거를 찾고, 상품을 비교한다. 내 분류기를 LLM 태깅과 비교한다.

| 장 | 주제 | 등급 | 남길 것 / 할 것 |
|---|---|---|---|
| 1 | RAG 필요성 | B | 질문 유형별로 RAG / 전체 입력 / 집계 도구 고르기 |
| 2 | 문서 로딩, Metadata | B | metadata 확정 (상품, 옵션, 별점, 작성일) |
| 3 | 청킹 | C | 리뷰는 짧아 거의 필요 없다. 긴 리뷰만 나누면 되는지 확인 |
| 4 | 임베딩, 코사인 유사도 | **A** | 리뷰 검색 기준선. 이어서 F12 내 분류기 (임베딩 + 로지스틱 회귀) |
| 5 | Qdrant, 필터 | **A** | F6 리뷰 근거 검색 |
| 6 | Citation, 근거 없으면 쓰지 않기 | **A** | F5 리포트 항목마다 근거 리뷰 인용 |
| 7 | Hybrid, Re-ranking | B | 개선 전후 비교 |
| 8 | 평가 (Hit@k) | **A** | 검색 질문 20개 평가셋. F12 분류기와 LLM 태깅 비교 |
| 9 | Parent-Child, Agentic, GraphRAG | C | Agentic 분기(F9)만 B로 |
