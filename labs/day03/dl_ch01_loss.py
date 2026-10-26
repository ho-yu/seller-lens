"""Day03 · 딥러닝 실전 1장 2강 — 손실 함수 (MSE / BCE / Cross Entropy).

손실(loss) = 예측이 정답에서 얼마나 틀렸는지를 숫자 하나로 낸 것. 많이 틀릴수록 크다.
문제 유형마다 쓰는 손실 함수가 다르다. 코드는 완성되어 있고, 실행 전에 결과를 예상해 본다.

    평점 예측 (숫자 하나)       → mse_loss
    부정 리뷰인가? (예/아니오)  → binary_cross_entropy
    부정/중립/긍정 (3개 중 1)  → cross_entropy

실행 (프로젝트 루트에서): uv run python labs/day03/dl_ch01_loss.py
"""

import numpy as np

CLASS_NAMES = ["부정", "중립", "긍정"]  # 위치가 곧 클래스 번호 (0, 1, 2)
EPS = 1e-12  # log(0)을 피하려고 확률을 [EPS, 1 - EPS]로 자른다


def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """logits에서 log 확률을 바로 계산한다 (Day02에서 작성)."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """차이를 → 제곱해서 → 평균 낸다. 평점 예측처럼 숫자를 맞히는 문제에 쓴다."""
    errors = y_pred - y_true
    return float(np.mean(errors**2))


def binary_cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """정답 쪽 확률에 -log를 씌워 평균 낸다. 정답 쪽 확률이 낮을수록 크게 벌한다.

    probs: 부정일 확률 (0~1), targets: 1 = 부정, 0 = 부정 아님.
    """
    safe_probs = np.clip(probs, EPS, 1.0 - EPS)
    losses = -(targets * np.log(safe_probs) + (1.0 - targets) * np.log(1.0 - safe_probs))
    return float(np.mean(losses))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """리뷰마다 정답 클래스의 log 확률에 -를 붙인다.

    logits: (N, 3) 점수, targets: (N,) 정답 번호. 반환: (N,) 리뷰별 loss.
    """
    log_probs = stable_log_softmax(logits)
    correct_log_probs = log_probs[np.arange(len(targets)), targets]
    return -correct_log_probs


def demo_mse() -> None:
    print("\n=== 1. MSE (평점 예측) ===")
    true_ratings = np.array([5.0, 4.0, 1.0, 3.0])
    good_pred = np.array([4.8, 4.1, 1.5, 3.0])
    bad_pred = np.array([3.0, 5.0, 4.0, 3.0])
    print(f"실제 평점 {true_ratings}")
    print(f"좋은 예측 {good_pred} → MSE {mse_loss(good_pred, true_ratings):.4f}")
    print(f"나쁜 예측 {bad_pred} → MSE {mse_loss(bad_pred, true_ratings):.4f}")
    for error in [1.0, 2.0, 4.0]:
        loss = mse_loss(np.array([3.0 + error]), np.array([3.0]))
        print(f"오차 {error:.0f} → MSE {loss:.0f}")


def demo_bce() -> None:
    print("\n=== 2. BCE (부정 리뷰 판별) ===")
    for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
        loss = binary_cross_entropy(np.array([p]), np.array([1.0]))
        print(f"정답 부정, 부정일 확률 {p:<4} → BCE {loss:.4f}")


def demo_cross_entropy() -> None:
    print("\n=== 3. Cross Entropy (감성 3분류) ===")
    targets = np.array([0])  # 정답: 부정
    for name, logits in [("확신", [[4.0, 0.0, -1.0]]), ("애매", [[1.0, 0.9, -1.0]])]:
        logits = np.array(logits)
        pred = CLASS_NAMES[np.argmax(logits)]
        loss = cross_entropy(logits, targets)[0]
        print(f"{name} {logits[0]} → 예측 {pred}, loss {loss:.4f}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    demo_mse()
    demo_bce()
    demo_cross_entropy()


if __name__ == "__main__":
    main()
