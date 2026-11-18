"""Day07 · 딥러닝 실전 6장 — Few-shot과 정확도 평가.

평점 라벨(1~2점 = 부정, 4~5점 = 긍정)로 LLM 감성 분류를 채점한다. 평가셋은 헷갈리기 쉬운 2점 30건 + 4점 30건.
예시 없이(zero-shot) 물을 때와 예시 4개를 보여 줄 때(few-shot)를 같은 평가셋으로 비교한다.

    build_fewshot_messages  System Prompt + 예시 + 리뷰 → messages 목록   ✍️
    confusion_counts        정답·예측 목록 → (tp, fp, fn)                 ✍️

실행 (프로젝트 루트에서, Ollama가 켜져 있어야 한다): uv run python labs/day07/dl_ch06_fewshot_eval.py
"""

from pathlib import Path

import pandas as pd
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWS_PATH = PROJECT_ROOT / "data" / "processed" / "reviews.csv"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# 라벨에 중립(3점)이 없어서 부정/긍정 두 가지만 묻는다
EVAL_PROMPT = (
    "너는 온라인 쇼핑몰 리뷰 분석가다. "
    "리뷰의 감성을 부정 또는 긍정 중 하나로만 답한다. 다른 말은 하지 않는다."
)
N_EVAL = 30  # 평가셋: 2점(부정)·4점(긍정) 각각
N_SHOT = 2  # 예시: 부정·긍정 각각


def build_fewshot_messages(system_prompt: str, examples: list[tuple[str, str]], review: str) -> list[dict]:
    """✍️ System Prompt, 예시 목록, 리뷰를 messages 목록으로 만든다 (예시는 user → assistant 쌍).

    examples: [(예시 리뷰, 정답), ...]. 비어 있으면 예시 없는(zero-shot) messages가 된다.
    """
    # 작성 순서
    # ① system 메시지로 시작 → 예시마다 user(예시 리뷰), assistant(정답) 두 메시지를 잇는다 → 마지막에 진짜 리뷰를 user로 붙인다
    # ② build_fewshot_messages("감성만", [("별로예요", "부정")], "좋아요")
    #    → [system "감성만", user "별로예요", assistant "부정", user "좋아요"]
    messages = [{"role": "system", "content": system_prompt}]

    for example_review, answer in examples:
        messages += [{"role": "user", "content": example_review}, {"role": "assistant", "content": answer}]

    return messages + [{"role": "user", "content": review}]



def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int]:
    """✍️ 부정(1)을 기준으로 (tp, fp, fn)을 센다.

    tp: 정답 1, 예측 1 (부정을 맞힘) / fp: 정답 0, 예측 1 (긍정을 부정이라 함) / fn: 정답 1, 예측 0 (부정을 놓침)
    """
    # 작성 순서
    # ① 정답·예측을 쌍으로 하나씩 본다 → 세 조건 각각 True인 쌍의 수를 센다
    # ② y_true=[1,1,1,1,0,0], y_pred=[1,0,0,0,1,0]
    #    → 첫 쌍만 tp, 다섯째 쌍이 fp, 둘~넷째 쌍이 fn → (1, 1, 3)
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return tp, fp, fn




def split_eval_and_shots(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """평가셋(2점 부정, 4점 긍정 각 N_EVAL개)을 먼저 뽑고, 남은 리뷰에서 짧은 예시(라벨별 N_SHOT개)를 뽑는다. 둘은 겹치지 않는다."""
    boundary = df[df["평점"].isin([2, 4])]  # 1점 대 5점은 너무 쉬워서 경계 평점만 쓴다
    eval_df = boundary.groupby("평점").sample(N_EVAL, random_state=seed)
    pool = df.drop(eval_df.index)
    pool = pool[pool["text"].str.len() <= 150]
    return eval_df, pool.groupby("label").sample(N_SHOT, random_state=seed)


def make_examples(shots: pd.DataFrame) -> list[tuple[str, str]]:
    """예시를 부정, 긍정, 부정, 긍정 순서로 섞는다 (한쪽이 연달아 나오면 그쪽으로 기운다)."""
    neg = shots[shots["label"] == 1]["text"].tolist()
    pos = shots[shots["label"] == 0]["text"].tolist()
    return [pair for n, p in zip(neg, pos) for pair in ((n, "부정"), (p, "긍정"))]


def predict(client: OpenAI, eval_df: pd.DataFrame, examples: list[tuple[str, str]]) -> list[int]:
    """평가셋 리뷰마다 모델에 묻고, 답에 '부정'이 있으면 1, 아니면 0으로 바꾼다."""
    y_pred = []
    for text in eval_df["text"]:
        messages = build_fewshot_messages(EVAL_PROMPT, examples, text)
        answer = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
        y_pred.append(int("부정" in answer.choices[0].message.content))
    return y_pred


def report(name: str, y_true: list[int], y_pred: list[int]) -> None:
    tp, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    recall = tp / (tp + fn)  # 진짜 부정 중 몇 개를 잡았나
    precision = tp / (tp + fp) if tp + fp else 0.0  # 부정이라고 한 것 중 몇 개가 진짜 부정인가
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{name:<9} 정확도 {accuracy:.2f} | 부정 재현율 {recall:.2f} 정밀도 {precision:.2f} F1 {f1:.2f} (tp={tp} fp={fp} fn={fn})")


def main() -> None:
    client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    try:
        print(build_fewshot_messages("감성만", [("별로예요", "부정")], "좋아요"))
        print("confusion_counts:", confusion_counts([1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 0]))

        eval_df, shots = split_eval_and_shots(pd.read_csv(REVIEWS_PATH))
        examples = make_examples(shots)
        y_true = eval_df["label"].tolist()
        print(f"\n평가셋 {len(eval_df)}건 (부정 {sum(y_true)}), 예시 {len(examples)}개, 겹침 {len(set(shots.index) & set(eval_df.index))}건")
        print(f"전부 긍정이라고 답하면 정확도 {y_true.count(0) / len(y_true):.2f}")

        zero_pred = predict(client, eval_df, [])
        few_pred = predict(client, eval_df, examples)
        print()
        report("zero-shot", y_true, zero_pred)
        report("few-shot", y_true, few_pred)

        print("\n[few-shot] 모델은 부정인데 평점은 높은 리뷰 (최대 3개)")
        wrong = eval_df[[p == 1 and t == 0 for p, t in zip(few_pred, y_true)]]
        for _, row in wrong.head(3).iterrows():
            print(f"  {row['평점']}점 | {row['text'][:50]}")
    except NotImplementedError as e:
        print(f"⏸ 건너뜀: {e}")


if __name__ == "__main__":
    main()
