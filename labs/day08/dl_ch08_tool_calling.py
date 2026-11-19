"""Day08 · 딥러닝 실전 8장 — Tool Calling (F5 리포트 v0).

모델은 "어떤 도구를 어떤 인자로 부르자"고 제안만 하고, 확인과 실행은 내 코드가 한다.
먼저 모델 없이 고정된 호출 4개를 실행하고, Ollama가 켜져 있으면 같은 질문을 도구 없이 / 도구와 함께 묻는다.

    run_tool        모델이 제안한 (이름, 인자 문자열) → 확인하고 실행한 결과 dict      ✍️
    tool_messages   도구 호출 목록 → 실행 결과를 담은 role="tool" 메시지 목록         ✍️

실행 (프로젝트 루트에서): uv run python labs/day08/dl_ch08_tool_calling.py
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pandas as pd
from openai import APIConnectionError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.csv"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
LABELS = {"부정": 1, "긍정": 0}  # reviews.csv의 label 값
MAX_ROUNDS = 3  # 모델이 도구를 끝없이 부르지 못하게 왕복 횟수를 막는다
SYSTEM_PROMPT = (
    "너는 쇼핑몰 셀러의 리뷰 분석 도우미다. 리뷰 건수는 도구 결과에 있는 숫자만 말하고, 추측하지 않는다. "
    "도구 결과가 ok=false이면 건수를 말하지 말고 왜 못 구했는지 안내한다."
)


class CountArgs(BaseModel):
    """count_reviews 도구의 인자 계약. 모델에는 스키마로 넘기고, 모델이 만든 인자도 이것으로 검증한다."""

    model_config = ConfigDict(strict=True, extra="forbid")
    product_id: str = Field(pattern=r"^product_\d{2}$", description="상품 ID. 예: product_03")
    sentiment: Literal["부정", "긍정"] = Field(description="셀 리뷰의 감성")


def count_reviews(product_id: str, sentiment: str) -> dict:
    """상품의 부정 또는 긍정 리뷰 건수를 센다. 숫자는 모델이 아니라 이 함수(pandas)가 만든다."""
    reviews = pd.read_csv(REVIEWS_PATH)
    product = reviews[reviews["product_id"] == product_id]
    if product.empty:  # 업무 규칙: 형식은 맞아도 데이터에 없는 상품일 수 있다
        return {"ok": False, "error_code": "product_not_found"}
    count = int((product["label"] == LABELS[sentiment]).sum())
    return {"ok": True, "product_id": product_id, "sentiment": sentiment, "count": count, "total": len(product)}


TOOL_ROUTER = {"count_reviews": count_reviews}  # 실행을 허용한 함수만 이름으로 등록한다 (allowlist)
COUNT_TOOL = {  # 모델에 보여 주는 도구 설명서: 이름, 설명, 인자 스키마
    "name": "count_reviews",
    "description": "상품 하나의 부정 또는 긍정 리뷰 건수와 전체 리뷰 건수를 센다.",
    "parameters": CountArgs.model_json_schema(),
}
TOOLS = [{"type": "function", "function": COUNT_TOOL}]


def run_tool(name: str, raw_arguments: str) -> dict:
    """✍️ 모델이 제안한 도구 호출(이름, 인자 문자열)을 확인한 뒤 실행해 결과 dict를 돌려준다."""
    # 작성 순서
    # ① 이름이 TOOL_ROUTER에 없으면 {"ok": False, "error_code": "unknown_tool"}
    #    → 인자 문자열을 CountArgs로 검증 (Day07과 같다). ValidationError면 error_code는 "invalid_arguments"
    #    → 통과하면 TOOL_ROUTER[name](**args.model_dump())를 실행한 결과
    # ② run_tool("count_reviews", '{"product_id": "product_03", "sentiment": "부정"}') → {"ok": True, ..., "count": 27, "total": 64}
    #    run_tool("delete_reviews", "{}") → {"ok": False, "error_code": "unknown_tool"}
    raise NotImplementedError("✍️ run_tool 작성 전")


def tool_messages(tool_calls: list) -> list[dict]:
    """✍️ 도구 호출마다 run_tool로 실행하고, 결과를 모델에 돌려줄 role="tool" 메시지로 만든다."""
    # 작성 순서
    # ① 빈 목록 → 호출(call)마다 run_tool(call.function.name, call.function.arguments)
    #    → {"role": "tool", "tool_call_id": call.id, "content": 결과의 JSON 문자열}을 목록에 넣는다 → 목록
    # ② 호출 1개(id="call_1", 위의 정상 예) → [{"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true, ...}'}]
    #    JSON 문자열은 json.dumps(result, ensure_ascii=False). 이 옵션이 없으면 한글이 부처럼 바뀐다
    raise NotImplementedError("✍️ tool_messages 작성 전")


def ask(client: OpenAI, question: str, use_tools: bool) -> str:
    """질문에 답한다. use_tools면 모델이 도구를 부르자고 할 때마다 실행 결과를 붙여 다시 묻는다."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    extra = {"tools": TOOLS} if use_tools else {}
    for _ in range(MAX_ROUNDS):
        response = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, **extra)
        reply = response.choices[0].message
        if not reply.tool_calls:  # 도구 호출이 없으면 이것이 최종 답이다
            return (reply.content or "").strip()
        results = tool_messages(reply.tool_calls)  # 실행은 모델이 아니라 여기서 일어난다
        for call, result in zip(reply.tool_calls, results):
            print(f"    모델의 제안: {call.function.name} {call.function.arguments}")
            print(f"    내 코드의 실행 결과: {result['content']}")
        messages += [reply.model_dump(exclude_none=True), *results]  # 모델의 제안과 실행 결과를 대화에 붙인다
    return "도구 호출이 너무 많아 멈췄습니다."


def demo_fixed() -> None:
    cases = {
        "정상": ("count_reviews", '{"product_id": "product_03", "sentiment": "부정"}'),
        "없는 도구": ("delete_reviews", '{"product_id": "product_03"}'),
        "인자 틀림": ("count_reviews", '{"product_id": "3번 상품", "sentiment": "나쁨"}'),
        "없는 상품": ("count_reviews", '{"product_id": "product_99", "sentiment": "부정"}'),
    }
    print("① 모델 없이: 고정된 도구 호출 4개")
    for label, (name, raw_arguments) in cases.items():
        print(f"  [{label}] {run_tool(name, raw_arguments)}")
    call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="count_reviews", arguments=cases["정상"][1]))
    print(f"  tool 메시지: {tool_messages([call])}")


def demo_llm() -> None:
    questions = [
        "product_03의 부정 리뷰는 몇 건이야?",
        "product_99의 부정 리뷰는 몇 건이야?",
        "부정 리뷰에는 어떻게 답글을 달면 좋을까? 한 문장으로.",
    ]
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    for use_tools in (False, True):
        print(f"\n② Ollama: {'도구와 함께' if use_tools else '도구 없이'}")
        for question in questions:
            print(f"  Q. {question}")
            print(f"  A. {ask(client, question, use_tools)}")


def main() -> None:
    try:
        demo_fixed()
    except NotImplementedError as e:
        print(f"⏸ 건너뜀: {e}")
        return
    try:
        demo_llm()
    except APIConnectionError:
        print("\n⏸ Ollama 부분 건너뜀: Ollama가 꺼져 있다")


if __name__ == "__main__":
    main()
