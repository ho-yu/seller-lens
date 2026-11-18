"""Day05 · 딥러닝 실전 5장 — LLM API 호출, System Prompt, temperature.

Ollama에 띄운 qwen3:4b-instruct를 OpenAI SDK로 부른다. 리뷰 감성 분류(F2)와 답글 초안(F4 v0).

    build_messages  System Prompt + 리뷰 → messages 목록   ✍️
    ask_llm         messages → 모델 답 문자열             ✍️
    count_blame     답 목록에서 '사용자 탓' 답 개수 세기  ✍️ (프롬프트 튜닝 보충)

실행 (프로젝트 루트에서, Ollama가 켜져 있어야 한다): uv run python labs/day05/dl_ch05_llm_api.py
"""

from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"  # Ollama의 OpenAI 호환 주소
MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # Day04에서 고른 모델

SENTIMENT_PROMPT = (
    "너는 온라인 쇼핑몰 리뷰 분석가다. "
    "리뷰의 감성을 부정, 중립, 긍정 중 하나로만 답한다. 다른 말은 하지 않는다."
)
REPLY_PROMPT_V0 = (  # 처음 프롬프트: 두 문장 + 환불 약속 금지만 있다
    "너는 인덕션 매트를 파는 쇼핑몰 셀러다. 고객 리뷰에 정중한 답글을 두 문장으로 쓴다. "
    "환불이나 보상은 약속하지 않는다."
)
REPLY_PROMPT = (  # V1: 고객 탓 금지·두 문장을 더했다
    "너는 인덕션 매트를 파는 쇼핑몰 셀러다. 고객 리뷰에 정중한 답글을 두 문장으로 쓴다. "
    "환불이나 보상은 약속하지 않는다."
    "고객 탓은 하지 않는다."
    "두 문장으로 대답한다."
)
BLAME_WORDS = ["사용법", "사용 방법", "잘못", "부주의", "주의"]  # 답에 있으면 '사용자 탓'으로 센다

# (평점, 리뷰) — data/processed/reviews.csv에서 고른 것
REVIEWS = [
    (2, "녹아내렸어요 .. 구매한지 얼마 되지도 않았는데 녹아내렸어요ㅠㅠ"),
    (5, "아주 마음에 들어요 디자인이 인덕션과 거의 같아서 깔끔하고 이쁩니다 ㅎㅎ"),
    (4, "요리하고 나면 누렇게 변하고 냄새가 나는게 정상인가요??"),
    (2, "무게도 질감도 색도 다 맘에 드는데 자꾸 인덕션위에서 고정이 안되고 돌아다녀요ㅠ"),
    (4, "오염 방지는 굳 전원 콘트롤 부분은 잘 보이지 않아 자르던지 해야 할듯"),
]

# V2: V1에 좋은 답글 예시 1개를 더한다. 예시 리뷰는 측정에 쓰는 REVIEWS[0]이 아닌 것으로 한다
EXAMPLE_REVIEW = REVIEWS[3][1]
# 두 문장, 공감 → 확인·안내, 고객 탓·환불 약속 없음
EXAMPLE_REPLY = (
    "매트가 인덕션 위에서 움직여 불편을 드린 점 죄송합니다. "
    "말씀해 주신 고정 문제는 담당자에게 전달해 제품 개선에 반영하겠습니다."
)
REPLY_PROMPT_V2 = f"{REPLY_PROMPT}\n\n예시\n리뷰: {EXAMPLE_REVIEW}\n답글: {EXAMPLE_REPLY}"


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


def count_blame(answers: list[str], words: list[str]) -> int:
    """✍️ answers 중 words 가운데 하나라도 들어 있는 답이 몇 개인지 센다."""
    # 작성 순서
    # ① 답 하나씩 본다 → 그 답에 words 중 하나라도 들어 있나 → True인 답만 센다
    # ② answers=["사용법을 확인해 주세요", "불편을 드려 죄송합니다", "고객님이 잘못 놓으셨어요"],
    #    words=["사용법", "잘못"] → 첫째 True, 둘째 False, 셋째 True → 2
    count = 0

    for answer in answers:
        if any(w in answer for w in words):
            count += 1

    return count


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


def demo_prompt_tuning(client: OpenAI) -> None:
    print("\n=== 3. 프롬프트 비교: 같은 리뷰, temperature별 10번씩 (사용자 탓 답 개수 / 서로 다른 답 개수) ===")
    if not EXAMPLE_REPLY:
        raise NotImplementedError("EXAMPLE_REPLY에 좋은 답글 예시를 써 주세요")
    review = REVIEWS[0][1]
    for name, prompt in [("V0", REPLY_PROMPT_V0), ("V1", REPLY_PROMPT), ("V2", REPLY_PROMPT_V2)]:
        messages = build_messages(prompt, review)
        for temperature in [0.3, 0.8, 1.2]:
            answers = [ask_llm(client, messages, temperature) for _ in range(10)]
            print(f"{name} T={temperature}: 탓 {count_blame(answers, BLAME_WORDS)}/10, 서로 다른 답 {len(set(answers))}/10")


def main() -> None:
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")  # Ollama는 키를 안 보지만 값은 필요하다
    try:
        print(build_messages("감성만 답해", "좋아요"))
        demo_sentiment(client)
        demo_temperature(client)
        demo_prompt_tuning(client)
    except NotImplementedError as e:
        print(f"⏸ 건너뜀: {e}")


if __name__ == "__main__":
    main()
