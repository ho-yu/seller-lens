# sellerlens 학습·개발 커리큘럼

> KANT 강의 4과목을 순서대로 학습하면서, 각 장에서 배운 내용을 sellerlens에 바로 적용한다.
> **딥러닝 실전 & 프롬프트 엔지니어링 → 데이터 수집 & API 서버 구축 → LangChain → RAG 기초/심화**
> 기능 ID(F1~F9)는 [ROADMAP.md](ROADMAP.md)를 참고한다. 장별 과제는 노션 강의 목차를 기준으로 짰다. (작성: 2026-10-02)

## 진행 방식

1. **강의 학습:** 강의 예제를 그대로 실행해 본다.
2. **장별 실습:** 같은 개념을 sellerlens 데이터로 다시 구현한다. 실습 코드는 `labs/dayNN/<과목>_chNN_*.py`에 둔다.
3. **기능 반영:** 실습에서 쓸 만한 코드는 `src/sellerlens/`로 옮겨 쌓는다.
4. **기록:** 하루를 `docs/days/dayNN.md`에 남기고, "마무리"하면 커밋 → `dayNN` 태그 → 푸시 → 노션 개발 일지 반영까지 한 번에 한다 (`.claude/skills/day-wrap`).

```
sellerlens/
├─ data/          raw(엑셀) · processed(JSONL) · docs(정책·FAQ 문서)
├─ labs/dayNN/    Day별 실습 (파일명에 과목·장 표시)
├─ src/sellerlens/ 실제 기능 코드 (과목이 진행될수록 커짐)
├─ eval/          평가셋과 결과 기록
└─ docs/          ROADMAP.md · CURRICULUM.md · days/dayNN.md
```

---

## 사전 준비 (Step 0, 반나절)

데이터 수집 과목은 두 번째지만, 첫 과목부터 리뷰 데이터가 필요하다. 그래서 최소한의 정제만 먼저 해 둔다. 제대로 된 정제는 2과목 4장에서 다시 한다.

- [ ] `03_merge_reviews.py` 완성: 엑셀 3개 병합, 파일명으로 `product_id` 컬럼 추가
- [ ] `이름`, `리뷰어번호` 컬럼 제거
- [ ] 제목과 본문을 합친 `text` 컬럼 생성, 빈 리뷰 제거
- [ ] `data/processed/reviews.csv` 재생성 (평점 라벨 포함)

**완료 기준:** 상품 3개 리뷰가 개인정보 없이 CSV 하나에 들어 있다.

---

## 1과목. 딥러닝 실전 & 프롬프트 엔지니어링

**과목 목표:** 리뷰 하나를 넣으면 분류, 태깅, 답글 초안이 나오는 CLI를 만든다. (F2, F3, F4, F5, F8의 기본 버전)

| 장 | 강의 주제 | sellerlens 실습 | 산출물 |
|---|---|---|---|
| 1장 | logits/softmax, 손실 함수 | 리뷰 TF-IDF 특성에 NumPy로 softmax와 Cross Entropy를 직접 구현한다. 긍정/부정 2클래스와 평점 5클래스 둘 다 해 본다 | `labs/dayNN/dl_ch01_loss.py` |
| 2장 | Gradient Descent, 옵티마이저 | 1장 분류기를 GD, SGD, Momentum, Adam으로 학습시키고 손실 곡선을 비교한다. 결과를 scikit-learn 로지스틱 회귀와도 비교한다 | 옵티마이저 비교 그래프, **분류 기준선 정확도** |
| 3장 | Attention, MHA | 리뷰 한 문장을 토큰 단위로 나눠 NumPy로 Q/K/V 어텐션을 계산하고 shape를 추적한다 (개념 실습, 선택) | `labs/dayNN/dl_ch03_attention.py` |
| 4장 | Base/SFT/Instruct, RLHF | sellerlens에 쓸 로컬 모델 후보를 정리한다 (instruct 여부, 한국어 성능, 크기) | `docs/model_choice.md` |
| 5장 | LLM API 구조, System Prompt | Ollama API로 리뷰 감성 분류를 호출한다. 판매자 답글 **응답 정책**을 System Prompt로 쓴다 (톤, 보상 확약 금지, 사실 확인) | F4 답글 초안 v0 |
| 6장 | Few-shot, 평가 루브릭, Self-Consistency | zero-shot과 few-shot 분류를 평점 라벨로 채점해 2장 기준선과 비교한다. 애매한 리뷰에는 Self-Consistency를 적용한다. 답글 품질 루브릭도 만든다 | `eval/classification.md` (정확도 비교표) |
| 7장 | JSON Schema, Pydantic, 출력 복구 | 리뷰 태깅 스키마를 만든다: `감성 / 불만 유형 / 긴급도 / 근거 문장`. 검증에 실패하면 복구한다. 평점과 내용이 다른 리뷰를 탐지한다 | **F2 리뷰 태깅**, **F3 우선순위** |
| 8장 | Tool / Function Calling | 집계 함수(평점 분포, 불만 유형 순위, 옵션별 통계)를 도구로 등록하고 LLM이 호출해 리포트를 쓰게 한다 | **F5 리포트 v0** |
| 9장 | 멀티모달 | (선택) 상품 상세페이지 이미지에서 스펙과 주의사항을 추출해 `data/docs/`에 저장한다. 이 문서는 나중에 RAG 근거로 쓴다 | 상품 스펙 문서 |
| 10장 | Retry/Fallback, 캐싱, 비용 로그, 통합 | 5~8장을 CLI 하나로 묶는다. timeout, retry, 토큰·지연시간 로그를 추가한다. 긍정 리뷰에서 소구점을 추출한다 | `sellerlens analyze` CLI, **F8** |

**과목 완료 기준:** 리뷰 파일을 넣으면 태깅 결과, 우선순위 목록, 답글 초안, 상품 리포트가 나오고 분류 정확도가 기록되어 있다.

---

## 2과목. 데이터 수집 & API 서버 구축

**과목 목표:** 1과목 CLI 기능을 FastAPI 서버로 옮기고, RAG에 쓸 문서 데이터셋을 만든다. (F1 완성)

| 장 | 강의 주제 | sellerlens 실습 | 산출물 |
|---|---|---|---|
| 1장 | 수집 방식 비교, 리스크 체크 | sellerlens 데이터 출처를 분류한다. 리뷰 엑셀은 스냅샷, 정책·FAQ는 RAG 문서, 공휴일 같은 값은 질문 시점에 API로 조회한다. 개인정보와 약관 체크리스트도 만든다 | `docs/data_sources.md` |
| 2장 | HTTP, API 인증, 환경변수 | Ollama HTTP API를 `httpx`로 직접 호출한다. 공공데이터포털 인증키를 `.env`로 관리한다 | `.env.example`, `config.py` |
| 3장 | pagination, rate limit, 원천 저장 | 공공 API(예: 특일정보 API로 공휴일 조회)를 수집해 원본 JSON을 저장한다. CS 답변에서 "연휴 배송 안내"에 쓴다 | `data/raw/holidays/*.json` |
| 4장 | 데이터 정제, 본문/메타데이터 분리 | **Step 0을 정식 파이프라인으로 다시 만든다.** 리뷰는 본문과 메타데이터(상품, 옵션, 평점, 날짜, 사진수)로 나눠 JSONL로 저장한다. 정책·FAQ 문서도 같은 형식으로 저장한다 | **F1 완성**, `reviews.jsonl`, `docs.jsonl` |
| 5장 | FastAPI 구조, REST, `/health` | 서버 뼈대를 만든다 (`src/sellerlens/api/`) | `/health` |
| 6장 | Pydantic 스키마, 문서 등록/조회 API | 리뷰 엑셀 업로드 API와 조회 API를 만든다. 상품, 평점, 기간으로 걸러 볼 수 있게 한다 | `POST /reviews/upload`, `GET /reviews` |
| 7장 | Query API, LLM 연동 | 1과목 기능을 API로 옮긴다. context는 아직 직접 골라서 넣는다 | `/reviews/{id}/tag`, `/reviews/{id}/reply-draft`, `/reports/{product_id}` |
| 8장 | 비동기, timeout/retry, middleware | 리뷰 수백 건을 `httpx.AsyncClient`로 동시에 태깅한다. 요청 로그 middleware와 rate limit을 추가한다 | 일괄 태깅 API, 처리 시간 비교 |

**과목 완료 기준:** Swagger UI(`/docs`)에서 엑셀 업로드부터 일괄 태깅, 답글 초안, 리포트까지 한 흐름으로 실행된다.

---

## 3과목. LangChain

**과목 목표:** 직접 짠 호출 코드를 LangChain 체인으로 다시 구성하고, 관측과 오류 대응, 대화 기능을 붙인다.

| 장 | 강의 주제 | sellerlens 실습 | 산출물 |
|---|---|---|---|
| 1장 | 단순 호출과 LangChain 구조 비교, 패키지 구조 | 1·2과목 코드를 LangChain 구조와 비교해 옮길 부분을 정한다. LLM 생성 코드는 한곳(`llm.py`)으로 모은다 | `src/sellerlens/llm.py` |
| 2장 | PromptTemplate, ChatModel, Output Parser | 태깅, 답글, 소구점 프롬프트를 템플릿으로 분리한다 | `prompts/` 모듈 |
| 3장 | LCEL, invoke/batch/stream | 태깅 체인을 LCEL로 만든다. `batch`로 일괄 태깅하고, 답글 초안은 `stream`으로 출력한다 | 태깅·답글 체인 |
| 4장 | RunnableParallel, Passthrough, assign | 리뷰 하나에서 태깅, 답글, 소구점을 병렬로 만든다. 리포트 섹션도 병렬로 생성한다 | 통합 분석 체인 |
| 5장 | 구조화 출력, 검증, retry | 1과목 7장의 Pydantic 스키마를 `with_structured_output`으로 옮기고, 실패하면 재시도한다 | 스키마 검증 통과율 기록 |
| 6장 | Document, Retriever 기초 | 리뷰를 Document와 metadata로 바꾸고 간단한 retriever로 "근거 리뷰 찾기"를 해 본다 | **F6 검색 v0** |
| 7장 | 대화 기록, 세션 관리 | 셀러와 여러 턴으로 대화하는 세션을 만든다 ("그중 사이즈 불만만 다시 보여줘") | 대화형 질의 |
| 8장 | Streaming, Callback, Tracing | FastAPI 응답을 스트리밍으로 바꾸고, LangSmith나 대체 도구로 체인 단계를 추적한다 | tracing 화면 캡처 |
| 9장 | with_retry, Fallback | 로컬 모델이 실패하면 다른 모델로 넘어가게 하고, 그래도 안 되면 기본 응답을 돌려준다 | fallback 체인 |
| 10장 | 통합 앱, 한계 정리 | 2과목 API 내부를 체인으로 교체한다. LangGraph가 필요한 지점은 정리만 해 둔다 | API v2 |

**과목 완료 기준:** 2과목과 같은 API가 체인 기반으로 동작하고, tracing으로 각 단계를 확인할 수 있으며, 대화형 질의가 된다.

---

## 4과목. RAG 기초 / 심화

**과목 목표:** 리뷰와 정책 문서를 근거로 답하는 검색 기능(F6), CS 답변 초안(F7), 셀러 어시스턴트(F9)를 만들고 평가한다.

| 장 | 강의 주제 | sellerlens 실습 | 산출물 |
|---|---|---|---|
| 1장 | RAG 필요성, 파이프라인, 2-Step 기준선, RAG·Long Context·CAG 비교 | 상품 하나의 리뷰는 전체를 한 번에 넣는 방식도 가능하다. 그래서 질문 유형별로 RAG, 전체 입력, 집계 도구 중 무엇을 쓸지 정한다 | `docs/rag_design.md`, 기준선 |
| 2장 | 문서 로딩, Metadata, 정제, JSONL | 2과목 4장 JSONL을 점검한다. 출처 추적용 metadata(`source`, `product_id`, `option`, `rating`, `date`)를 확정한다 | 검수 리포트 |
| 3장 | Chunk Size/Overlap, 한국어 청킹 | 정책·FAQ·상세 문서 청킹 전략을 비교한다. 리뷰는 1건을 청크 1개로 두는 이유를 정리한다 | 청킹 비교표 |
| 4장 | 임베딩, 코사인 유사도, Top-k, 모델 비교 | 한국어 임베딩 모델 2~3개를 리뷰 검색 질문으로 비교한다 | Dense 기준선 |
| 5장 | Qdrant, Upsert, Metadata Filter, 갱신 | `reviews`, `docs` 두 컬렉션을 만든다. 상품, 평점, 옵션, 기간 필터를 걸고, 새 엑셀이 들어오면 갱신한다 | **F6 리뷰 근거 검색** |
| 6장 | Context 구성, Citation, 답변 보류 | 정책·FAQ 근거로 CS 답변 초안을 쓴다. 출처를 표시하고, 근거가 없으면 "담당자 확인 필요"로 보류한다 | **F7 CS 답변 초안** |
| 7장 | BM25, Hybrid(RRF), Re-ranking, Query Rewrite | 한국어 BM25(Kiwi 형태소 분석)를 붙여 Dense와 RRF로 합치고 리랭킹한다. 셀러의 구어체 질문을 검색어로 바꾼다 | 개선 전후 비교 |
| 8장 | Golden Dataset, Hit@k, RAGAS, 오류 분석 | 실제 셀러 질문 20~30개로 평가셋을 만들고, 검색과 답변을 평가해 오류 유형을 분류한다 | `eval/rag_golden.jsonl`, 평가 리포트 |
| 9장 | Parent-Child, Corrective·Agentic RAG, (GraphRAG, Multimodal) | 긴 정책 문서에 Parent-Child를 적용한다. Agentic RAG로 질문을 통계(집계 도구)와 근거(RAG)로 분기한다. GraphRAG와 Multimodal은 선택 | **F9 셀러 어시스턴트** |

**과목 완료 기준:** `/query` 하나로 통계, 리뷰 근거, CS 답변 질문에 모두 답하고, 평가 지표가 개선 전후로 기록되어 있다.

---

## 과목별 마일스톤 요약

| 과목 | 끝나면 셀러가 할 수 있는 일 |
|---|---|
| 1. 딥러닝 실전 & 프롬프트 | 리뷰 파일로 태깅, 우선순위, 답글 초안, 리포트를 CLI에서 받는다 |
| 2. 데이터 수집 & API | 같은 기능을 API로 쓰고, 엑셀 업로드와 일괄 처리를 한다 |
| 3. LangChain | 대화하면서 리뷰를 분석하고, 오류가 나도 서비스가 멈추지 않는다 |
| 4. RAG | 리뷰와 정책 근거를 출처와 함께 찾고, CS 답변 초안을 받는다 |
