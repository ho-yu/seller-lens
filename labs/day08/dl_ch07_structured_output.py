"""Day08 · 딥러닝 실전 7장 — Structured Output과 Pydantic 검증 (F2 리뷰 태깅).

LLM의 답 문자열을 ReviewTag 모양으로 검증한다. 먼저 고정된 답 5개로 어떤 답이 왜 실패하는지 보고,
Ollama가 켜져 있으면 리뷰 3개를 "프롬프트만" / "스키마도 전달" 두 방식으로 물어 비교한다.

    ReviewTag      감성·불만 유형·긴급도·근거 문장의 출력 계약          ✍️
    validate_tag   답 문자열 → (ReviewTag 또는 None, 오류 목록)          ✍️

실행 (프로젝트 루트에서): uv run python labs/day08/dl_ch07_structured_output.py
"""

from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
ComplaintType = Literal["배송", "파손", "품질·내구성", "사이즈·호환", "사용성", "소재·냄새", "가격", "CS", "없음"]
TAG_PROMPT = (
    "너는 온라인 쇼핑몰 리뷰 분석가다. 리뷰의 감성, 불만 유형, 긴급도(1 낮음 ~ 3 바로 대응), 근거 문장을 "
    "JSON으로만 답한다. 근거 문장은 리뷰 속 문장을 그대로 옮긴다. 불만이 없으면 불만 유형은 없음이다."
)


class ReviewTag(BaseModel):
    """✍️ 리뷰 태깅 결과의 출력 계약. 정의 안 한 필드(extra)와 자동 형 변환(strict)은 막는다."""

    model_config = ConfigDict(strict=True, extra="forbid")
    # 작성 순서
    # ① 감성은 부정·중립·긍정 중 하나 → 불만 유형은 ComplaintType 중 하나 → 긴급도는 1~3 정수
    #    → 근거 문장은 빈칸이 아닌 문자열. 필드 하나에 한 줄씩 4줄 (이름: sentiment, complaint_type, urgency, evidence)
    # ② {"sentiment": "부정", "complaint_type": "배송", "urgency": 3, "evidence": "일주일째 안 와요."} → 통과
    #    urgency가 5면 실패, "3"(문자열)이어도 실패
    # ✍️ 여기에 필드 4줄을 쓴다
    sentiment: Literal["부정", "중립", "긍정"]
    complaint_type: ComplaintType
    urgency: int = Field(ge=1, le=3)
    evidence: str = Field(min_length=1)




def short_errors(error: ValidationError) -> list[str]:
    """ValidationError를 '필드: 오류 종류' 목록으로 줄인다. 필드가 없는 오류(JSON 문법 등)는 <root>."""
    return [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['type']}" for e in error.errors()]


def validate_tag(raw: str) -> tuple[ReviewTag | None, list[str]]:
    """✍️ 답 문자열을 ReviewTag로 검증한다. 성공하면 (객체, []), 실패하면 (None, 오류 목록)."""
    # 작성 순서
    # ① 문자열을 ReviewTag로 바꿔 본다 → 되면 (객체, 빈 목록) → ValidationError가 나면 (None, short_errors(오류))
    # ② validate_tag('{"sentiment": "부정"}')
    #    → (None, ['complaint_type: missing', 'urgency: missing', 'evidence: missing'])
    try:
        tag = ReviewTag.model_validate_json(raw)                 # 문자열 → 검증된 객체
    except ValidationError as error:
        return None, short_errors(error)                         # 실패: 객체 없이 오류 목록만
    return tag, []                                               # 성공: 객체와 빈 목록


def evidence_in_review(tag: ReviewTag, review: str) -> bool:
    """업무 규칙: 근거 문장이 리뷰 원문에 실제로 있는가 (3강 §3). 구조 검증으로는 못 잡는다."""
    return tag.evidence.strip() in review


def ask_tag(client: OpenAI, review: str, use_schema: bool) -> str:
    """리뷰 하나를 태깅해 답 문자열을 그대로 돌려준다. use_schema면 JSON Schema를 response_format으로 넘긴다."""
    extra = {}
    if use_schema:
        schema = {"name": "review_tag", "schema": ReviewTag.model_json_schema()}
        extra = {"response_format": {"type": "json_schema", "json_schema": schema}}
    messages = [{"role": "system", "content": TAG_PROMPT}, {"role": "user", "content": review}]
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, **extra)
    return response.choices[0].message.content


def check(name: str, raw: str, review: str) -> bool:
    """검증 결과를 한 줄로 출력하고, 구조와 업무 규칙을 모두 통과했는지 돌려준다."""
    tag, errors = validate_tag(raw)
    if tag is None:
        print(f"  [{name}] 구조 실패 | {', '.join(errors)}")
        return False
    ok = evidence_in_review(tag, review)
    print(f"  [{name}] 구조 통과 | {tag.sentiment}/{tag.complaint_type}/{tag.urgency} | 근거가 리뷰에 {'있음' if ok else '없음'}")
    return ok


def demo_fixed() -> None:
    review = "주문한 지 일주일째 안 와요. 문의해도 답이 없어요."
    good = '{"sentiment": "부정", "complaint_type": "배송", "urgency": 3, "evidence": "주문한 지 일주일째 안 와요."}'
    cases = {
        "정상": good,
        "코드 펜스": f"```json\n{good}\n```",
        "허용값·범위 밖": good.replace('"부정"', '"매우 부정"').replace("3", "5"),
        "문자열 숫자·추가 필드": good.replace("3", '"3"').replace("}", ', "note": "급함"}'),
        "근거 지어냄": good.replace("주문한 지 일주일째 안 와요.", "배송이 너무 느려요."),
    }
    print("① 고정된 답 5개 검증")
    for name, raw in cases.items():
        check(name, raw, review)


def demo_llm() -> None:
    reviews = [
        "주문한 지 일주일째 안 와요. 문의해도 답이 없어요.",
        "사이즈는 딱 맞는데 박음질이 한 달 만에 풀렸어요.",
        "가볍고 튼튼해요. 아이가 매일 써요.",
    ]
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    for use_schema in (False, True):
        print(f"\n② Ollama: {'스키마도 전달' if use_schema else '프롬프트만'}")
        passed = sum(check(f"리뷰 {i}", ask_tag(client, r, use_schema), r) for i, r in enumerate(reviews, 1))
        print(f"  → 구조와 근거를 모두 통과: {passed}/{len(reviews)}개")


def main() -> None:
    if not ReviewTag.model_fields:
        print("⏸ 건너뜀: ✍️ ReviewTag 필드 작성 전")
        return
    try:
        print(validate_tag('{"sentiment": "부정"}'))
        demo_fixed()
    except NotImplementedError as e:
        print(f"⏸ 건너뜀: {e}")
        return
    try:
        demo_llm()
    except Exception as e:  # Ollama가 꺼져 있으면 연결 오류가 난다
        print(f"\n⏸ Ollama 부분 건너뜀: {type(e).__name__}")


if __name__ == "__main__":
    main()
