"""LLM 호출 함수 모음. Day05·Day06 실습(`labs/`)에서 직접 쓴 함수를 옮겨 왔다."""

from openai import OpenAI

OLLAMA_URL = "http://localhost:11434/v1"  # Ollama의 OpenAI 호환 주소
MODEL = "qwen3:4b-instruct-2507-q4_K_M"  # Day04에서 고른 모델


def build_messages(system_prompt: str, review: str) -> list[dict]:
    """System Prompt와 리뷰를 LLM에 보낼 messages 목록으로 만든다 (Day05)."""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": review}]


def build_fewshot_messages(system_prompt: str, examples: list[tuple[str, str]], review: str) -> list[dict]:
    """System Prompt, 예시 목록, 리뷰를 messages 목록으로 만든다 (Day06).

    examples: [(예시 리뷰, 정답), ...]. 예시마다 user → assistant 쌍이 되고, 비어 있으면 예시 없는(zero-shot) messages가 된다.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for example_review, answer in examples:
        messages += [{"role": "user", "content": example_review}, {"role": "assistant", "content": answer}]
    return messages + [{"role": "user", "content": review}]


def ask_llm(client: OpenAI, messages: list[dict], temperature: float) -> str:
    """messages를 모델에 보내고, 답 문자열만 꺼내 앞뒤 공백을 지운다 (Day05)."""
    response = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature)
    return response.choices[0].message.content.strip()
