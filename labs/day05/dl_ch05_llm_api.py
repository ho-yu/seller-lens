"""Day05 · 딥러닝 실전 5장 — LLM API 호출, System Prompt, temperature.

Ollama에 띄운 qwen3:4b-instruct를 OpenAI SDK로 부른다. 리뷰 감성 분류(F2)와 답글 초안(F4 v0).

    build_messages  System Prompt + 리뷰 → messages 목록   ✍️
    ask_llm         messages → 모델 답 문자열             ✍️

실행 (프로젝트 루트에서, Ollama가 켜져 있어야 한다): uv run python labs/day05/dl_ch05_llm_api.py
"""

from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"  # Ollama의 OpenAI 호환 주소
MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # Day04에서 고른 모델

SENTIMENT_PROMPT = (
    "너는 온라인 쇼핑몰 리뷰 분석가다. "
    "리뷰의 감성을 부정, 중립, 긍정 중 하나로만 답한다. 다른 말은 하지 않는다."
)
REPLY_PROMPT = (
    "너는 인덕션 매트를 파는 쇼핑몰 셀러다. 고객 리뷰에 정중한 답글을 두 문장으로 쓴다. "
    "환불이나 보상은 약속하지 않는다."
)

# (평점, 리뷰) — data/processed/reviews.csv에서 고른 것
REVIEWS = [
    (2, "녹아내렸어요 .. 구매한지 얼마 되지도 않았는데 녹아내렸어요ㅠㅠ"),
    (5, "아주 마음에 들어요 디자인이 인덕션과 거의 같아서 깔끔하고 이쁩니다 ㅎㅎ"),
    (4, "요리하고 나면 누렇게 변하고 냄새가 나는게 정상인가요??"),
    (2, "무게도 질감도 색도 다 맘에 드는데 자꾸 인덕션위에서 고정이 안되고 돌아다녀요ㅠ"),
    (4, "오염 방지는 굳 전원 콘트롤 부분은 잘 보이지 않아 자르던지 해야 할듯"),
]


def build_messages(system_prompt: str, review: str) -> list[dict]:
    """✍️ System Prompt와 리뷰를 LLM에 보낼 messages 목록으로 만든다.

    반환: [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    # 작성 순서
    # ① system 역할 메시지를 만든다 → user 역할 메시지를 만든다 → 둘을 리스트로 묶어 돌려준다
    # ② build_messages("감성만 답해", "좋아요")
    #    → [{"role": "system", "content": "감성만 답해"}, {"role": "user", "content": "좋아요"}]
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": review}]


def ask_llm(client: OpenAI, messages: list[dict], temperature: float) -> str:
    """✍️ messages를 모델에 보내고, 답 문자열만 꺼내 앞뒤 공백을 지운다."""
    # 작성 순서
    # ① 보낸다: client.chat.completions.create(model=, messages=, temperature=)
    #    → 답을 꺼낸다: 응답.choices[0].message.content → 앞뒤 공백을 지운다 (.strip())
    # ② 모델 답이 " 부정\n"이면 → "부정"
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return response.choices[0].message.content.strip()


def demo_sentiment(client: OpenAI) -> None:
    print("\n=== 1. 감성 분류 (temperature 0) ===")
    for rating, review in REVIEWS:
        answer = ask_llm(client, build_messages(SENTIMENT_PROMPT, review), temperature=0)
        print(f"{rating}점 → {answer:<3} | {review[:30]}")


def demo_temperature(client: OpenAI) -> None:
    print("\n=== 2. 답글 초안: temperature 0 vs 1.2 (같은 리뷰, 2번씩) ===")
    messages = build_messages(REPLY_PROMPT, REVIEWS[0][1])
    for temperature in [0, 1.2]:
        for i in range(2):
            print(f"[T={temperature} #{i + 1}] {ask_llm(client, messages, temperature)}")


def main() -> None:
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")  # Ollama는 키를 안 보지만 값은 필요하다
    try:
        print(build_messages("감성만 답해", "좋아요"))
        demo_sentiment(client)
        demo_temperature(client)
    except NotImplementedError as e:
        print(f"⏸ 건너뜀: {e}")


if __name__ == "__main__":
    main()
