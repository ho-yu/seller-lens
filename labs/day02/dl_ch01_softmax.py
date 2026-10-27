"""Day02 · 딥러닝 실전 1장 1강 — softmax와 temperature.

모델은 클래스마다 점수(logit)를 내고, softmax가 그 점수를 합이 1인 확률로 바꾼다.
LLM의 temperature는 이 확률을 뾰족하게(T < 1) 또는 평평하게(T > 1) 만든다.

실행 (프로젝트 루트에서): uv run python labs/day02/dl_ch01_softmax.py
"""

import numpy as np

CLASS_NAMES = ["부정", "중립", "긍정"]  # 위치가 곧 클래스 번호 (0, 1, 2)
TEMPERATURES = [0.1, 0.5, 1.0, 2.0, 10.0]


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """logits를 합이 1인 확률로 바꾼다. 최댓값을 먼저 빼서 큰 값에서도 안전하다."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def predict_class(logits: np.ndarray) -> str:
    """✍️ 점수 → 확률 → 가장 큰 확률의 클래스 이름."""
    probs = stable_softmax(logits)
    class_index = np.argmax(probs)
    return CLASS_NAMES[class_index]


def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """✍️ 확률을 거치지 않고 log 확률을 바로 계산한다 (Day03 Cross Entropy에서 쓴다)."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """✍️ logits를 T로 나눈 뒤 softmax. LLM API의 temperature가 하는 일이다."""
    return stable_softmax(logits / temperature)


def demo_softmax() -> None:
    print("\n=== 1. logit과 확률 ===")
    logits = np.array([2.0, 0.5, -1.0])  # [부정, 중립, 긍정]
    probs = stable_softmax(logits)
    print(f"logits {logits} → 확률 {probs} (합 {np.sum(probs):.4f})")
    print(f"+100:  {stable_softmax(logits + 100)}  ← 차이가 그대로라 결과도 그대로")
    print(f"×2:    {stable_softmax(logits * 2)}  ← 차이가 벌어져 1등 확률이 커짐")
    print(f"예측:  {predict_class(logits)}")
    assert predict_class(logits) == "부정"
    assert np.allclose(np.exp(stable_log_softmax(logits)), probs)


def demo_temperature() -> None:
    print("\n=== 2. temperature ===")
    logits = np.array([2.0, 0.5, -1.0])
    for temperature in TEMPERATURES:
        print(f"T={temperature:>4}: {softmax_with_temperature(logits, temperature)}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    demo_softmax()
    demo_temperature()


if __name__ == "__main__":
    main()
