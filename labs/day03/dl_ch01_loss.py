"""Day03 · 딥러닝 실전 1장 2강 — MSE / BCE / Cross Entropy.

이 파일이 하는 일
    문제 유형마다 쓰는 손실 함수 3개를 직접 만들고, "예측이 정답에 가까울수록 loss가 작아지는지"를
    sellerlens 예시(평점 예측, 부정 리뷰 판별, 감성 3분류)로 확인한다.
    사용자는 "✍️ 직접 작성" 계산 함수의 본문만 쓴다. demo 함수(예시 값, 출력, 검증)는 이미 작성되어 있다.
    여기서 만든 cross_entropy_per_sample은 2장(Gradient Descent)에서 분류기를 학습시킬 때 다시 쓴다.

전체 흐름
    평점 예측 (N,)       ── mse_loss ──────────────────────────▶ loss (숫자 하나)
    부정일 확률 (N,)     ── binary_cross_entropy ──────────────▶ loss
    감성 logits (N, 3)  ── cross_entropy_per_sample ─▶ (N,) ── reduce_loss ─▶ loss
    정답 번호 (N,)       ── to_one_hot ─▶ (N, 3)

키워드 ↔ 코드 (✍️ = 직접 작성, 괄호는 참고할 강의안 절)
    1  MSE (1~3절)                  ✍️ mse_loss                  → demo_mse
    2  BCE와 clip (4절)             ✍️ binary_cross_entropy      → demo_bce
    3  Cross Entropy (5절, 9절)     ✍️ cross_entropy_per_sample  → demo_cross_entropy
    4  정답 표현 one-hot (5~6절)    ✍️ to_one_hot                → demo_one_hot
    5  reduction (7절)              ✍️ reduce_loss               → demo_reduction

진행 상황
    [ ] 1  mse_loss  ◀ 지금
    [ ] 2  binary_cross_entropy
    [ ] 3  cross_entropy_per_sample
    [ ] 4  to_one_hot
    [ ] 5  reduce_loss

작성 방법
    함수마다 docstring(무엇을 위한 함수인가, Args, Returns)을 읽고, "작성 순서" 주석의
    줄 모양을 그대로 쓰되 ___ 칸만 채운다.
    다 쓰면 TODO 주석과 raise NotImplementedError 줄을 지우고 실행한다. 작성 전인 키워드는 건너뛴다.
    실행 전 예상, 정답 출력(접힘), 설명해 보기, 변형 문제: labs/day03/TASKS.md
실행 (프로젝트 루트에서): uv run python labs/day03/dl_ch01_loss.py
"""

import numpy as np

# =============================================================================
# 상수
# =============================================================================

# 리스트 위치가 곧 클래스 번호다 (0=부정, 1=중립, 2=긍정).
CLASS_NAMES = ["부정", "중립", "긍정"]

# 키워드 2: log(0)을 피하려고 확률을 [EPS, 1 - EPS]로 자른다.
EPS = 1e-12

# 키워드 5: 지원하는 reduction 방식
REDUCTIONS = ["none", "mean", "sum"]


# =============================================================================
# Day02에서 작성한 함수 (그대로 가져왔다)
# =============================================================================


def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """logits에서 log 확률을 바로 계산한다 (Day02 ✅). Cross Entropy에서 쓴다."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


# =============================================================================
# 계산 함수 — ✍️ 직접 작성하는 곳
# =============================================================================


# --- 키워드 1 ---------------------------------------------------------------


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """✍️ 회귀용 손실. 예측과 정답의 차이를 제곱해 평균 낸다.

    무엇을 위한 함수인가
        숫자 하나를 맞히는 문제(회귀)의 loss다. sellerlens에서는 리뷰 본문만 보고
        평점(1.0~5.0)을 예측할 때 쓴다. 예측이 정답에 가까울수록 0에 가까워진다.

    공식
        MSE = (1/N) Σ (y_pred_i - y_true_i)²

    Args:
        y_pred: 예측한 평점. shape (N,).
        y_true: 실제 평점. shape (N,).

    Returns:
        loss 숫자 하나 (0 이상).
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"y_pred {y_pred.shape}와 y_true {y_true.shape}의 shape이 다르다")

    # TODO(키워드 1) 작성 순서. 참고: 강의안 3절, 8절. 아래 모양 그대로 쓰고 ___ 칸만 채운다.
    # 예시는 y_pred = [2.0, 7.0], y_true = [3.0, 5.0] (강의안 8절)
    # 1. errors = ___ - ___
    #    ___: 예측에서 정답을 뺀다. 예: → [-1.0, 2.0]
    # 2. squared_errors = errors ** ___
    #    ___: 제곱. 음수 오차가 양수 오차를 지우지 않게 하고, 큰 오차를 더 크게 벌한다.
    #    예: → [1.0, 4.0]
    # 3. return float(np.mean(___))
    #    ___: 2에서 만든 값. 예: → 2.5
    raise NotImplementedError("✍️ mse_loss 작성 전")


# --- 키워드 2 ---------------------------------------------------------------


def binary_cross_entropy(probs: np.ndarray, targets: np.ndarray, eps: float = EPS) -> float:
    """✍️ 이진 분류용 손실. 정답 쪽 확률이 낮을수록 크게 벌한다.

    무엇을 위한 함수인가
        "예 / 아니오" 문제의 loss다. sellerlens에서는 "이 리뷰가 부정 리뷰인가?(1 = 부정)"를
        판별할 때 쓴다. 정답이 1이면 p가, 정답이 0이면 1 - p가 클수록 loss가 작다.

    공식
        BCE = -(1/N) Σ [ y·log(p) + (1 - y)·log(1 - p) ]

    Args:
        probs: 부정일 확률 p. shape (N,), 값은 0~1.
        targets: 정답 0 또는 1. shape (N,).
        eps: log(0)을 피하려고 확률을 자를 아주 작은 값.

    Returns:
        loss 숫자 하나 (0 이상).
    """
    if probs.shape != targets.shape:
        raise ValueError(f"probs {probs.shape}와 targets {targets.shape}의 shape이 다르다")

    # TODO(키워드 2) 작성 순서. 참고: 강의안 4절. 아래 모양 그대로 쓰고 ___ 칸만 채운다.
    # 예시는 probs = [0.9, 0.1], targets = [1.0, 0.0] (둘 다 잘 맞힘)
    # 1. safe_probs = np.clip(probs, ___, ___)
    #    ___: 아래 경계 eps, 위 경계 1.0 - eps. 확률이 정확히 0이나 1이어도 log가 -inf가 되지 않는다.
    # 2. losses = -(targets * np.log(___) + (1.0 - targets) * np.log(___))
    #    ___: 앞은 "정답이 1일 때 볼 확률", 뒤는 "정답이 0일 때 볼 확률" (1에서 뺀 값).
    #    예: → [0.1054, 0.1054]   (-log 0.9 = 0.1054)
    # 3. return float(np.mean(losses))
    #    예: → 0.1054
    raise NotImplementedError("✍️ binary_cross_entropy 작성 전")


# --- 키워드 3 ---------------------------------------------------------------


def cross_entropy_per_sample(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """✍️ 다중 분류용 손실을 리뷰마다 하나씩 계산한다.

    무엇을 위한 함수인가
        [부정, 중립, 긍정] 중 하나를 고르는 감성 분류의 loss다. 리뷰마다
        "정답 클래스의 log 확률에 마이너스"를 붙인다. 정답 확률이 1에 가까울수록 0에 가깝다.
        확률이 아니라 logits를 받는 이유는 Day02 키워드 4다 (underflow 없이 log 확률을 구하려고).

    공식
        L_i = -log p_(i, y_i)       y_i = i번째 리뷰의 정답 클래스 번호

    Args:
        logits: 리뷰 N개의 [부정, 중립, 긍정] 점수. shape (N, 3).
        targets: 리뷰마다 정답 클래스 번호 (0, 1, 2). shape (N,).

    Returns:
        리뷰별 loss. shape (N,). 평균은 키워드 5의 reduce_loss가 맡는다.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits는 (N, C)여야 한다. 지금 shape: {logits.shape}")
    if targets.shape != (logits.shape[0],):
        raise ValueError(f"targets는 ({logits.shape[0]},)여야 한다. 지금 shape: {targets.shape}")

    # TODO(키워드 3) 작성 순서. 참고: 강의안 5절. 아래 모양 그대로 쓰고 ___ 칸만 채운다.
    # 예시는 logits = [[2.0, 1.0, -0.5], [0.2, 0.3, 1.5]], targets = [0, 2] (강의안 5절)
    # 1. log_probs = stable_log_softmax(___, axis=-1)
    #    ___: logits. 예: → [[-0.3715, -1.3715, -2.8715], [-1.7534, -1.6534, -0.4534]]
    # 2. row_indices = np.arange(___)
    #    ___: 리뷰 수. logits.shape[0]. 예: → [0, 1]
    # 3. correct_log_probs = log_probs[row_indices, ___]
    #    ___: 리뷰마다 꺼낼 열 번호 = 정답 클래스. 행 번호와 열 번호를 짝지어 한 칸씩 꺼낸다.
    #    예: (0행, 0열), (1행, 2열) → [-0.3715, -0.4534]
    # 4. return -___
    #    ___: 3에서 만든 값. 예: → [0.3715, 0.4534]
    raise NotImplementedError("✍️ cross_entropy_per_sample 작성 전")


# --- 키워드 4 ---------------------------------------------------------------


def to_one_hot(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """✍️ 정답 클래스 번호를 one-hot 표현으로 바꾼다.

    무엇을 위한 함수인가
        같은 정답을 "번호 하나"(class index)로도, "클래스마다 0과 1"(one-hot)로도 적을 수 있다.
        라이브러리마다 받는 형식이 달라서 둘을 오갈 줄 알아야 한다.
        예: 정답 2(긍정) → [0, 0, 1]

    Args:
        targets: 정답 클래스 번호. shape (N,).
        num_classes: 클래스 수 C (sellerlens는 3).

    Returns:
        one-hot 배열. shape (N, C). 행마다 정답 자리만 1.0이고 나머지는 0.0.
    """
    # TODO(키워드 4) 작성 순서. 참고: 강의안 5절 "one-hot과 클래스 인덱스".
    # 아래 모양 그대로 쓰고 ___ 칸만 채운다.
    # 예시는 targets = [0, 2], num_classes = 3
    # 1. one_hot = np.zeros((___, ___))
    #    ___: 리뷰 수 len(targets), 클래스 수. 예: → [[0, 0, 0], [0, 0, 0]]
    # 2. one_hot[np.arange(len(targets)), ___] = 1.0
    #    ___: 리뷰마다 1을 넣을 열 번호 = 정답 클래스.
    #    키워드 3의 3단계 "행·열 짝지어 꺼내기"와 같은 방법으로 넣는다.
    #    예: → [[1, 0, 0], [0, 0, 1]]
    # 3. return one_hot
    raise NotImplementedError("✍️ to_one_hot 작성 전")


# --- 키워드 5 ---------------------------------------------------------------


def reduce_loss(sample_losses: np.ndarray, reduction: str = "mean") -> np.ndarray | float:
    """✍️ 리뷰별 loss를 reduction 방식에 따라 하나로 줄인다 (또는 그대로 둔다).

    무엇을 위한 함수인가
        학습에는 보통 loss 숫자 하나가 필요하다. 평균(mean)을 쓸지 합(sum)을 쓸지에 따라
        배치 크기가 바뀔 때 loss 크기가 달라진다. "none"은 어떤 리뷰가 loss가 큰지 볼 때 쓴다.

    Args:
        sample_losses: 리뷰별 loss. shape (N,).
        reduction: "none"(그대로), "mean"(평균), "sum"(합) 중 하나.

    Returns:
        "none"이면 shape (N,) 배열, "mean"·"sum"이면 숫자 하나.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction은 {REDUCTIONS} 중 하나여야 한다. 지금: {reduction!r}")

    # TODO(키워드 5) 작성 순서. 참고: 강의안 7절. 아래 모양 그대로 쓰고 ___ 칸만 채운다.
    # 예시는 sample_losses = [0.2, 0.8, 0.5] (강의안 7절)
    # 1. if reduction == "none":
    #        return ___                        ← 줄이지 않고 그대로. 예: → [0.2, 0.8, 0.5]
    # 2. if reduction == "mean":
    #        return float(np.mean(___))        예: → 0.5
    # 3. return float(np.___(sample_losses))   ← 남은 경우는 "sum". 예: → 1.5
    raise NotImplementedError("✍️ reduce_loss 작성 전")


# =============================================================================
# 키워드 확인 — 이미 작성되어 있다. 실행해서 결과를 본다.
# =============================================================================
# 예시 값을 넣어 위 함수를 호출하고, 결과를 출력하고, assert로 맞는지 확인한다.


def demo_mse() -> None:
    """키워드 1. 예측이 정답에 가까울수록 MSE가 작고, 큰 오차는 제곱으로 더 크게 벌한다."""
    print("\n=== 키워드 1. MSE (평점 예측) ===")

    # 리뷰 4개의 실제 평점과, 모델 두 개가 본문만 보고 예측한 평점이다.
    true_ratings = np.array([5.0, 4.0, 1.0, 3.0])
    good_pred = np.array([4.8, 4.1, 1.5, 3.0])
    bad_pred = np.array([3.0, 5.0, 4.0, 3.0])
    good_loss = mse_loss(good_pred, true_ratings)
    bad_loss = mse_loss(bad_pred, true_ratings)
    print(f"실제 평점: {true_ratings}")
    print(f"좋은 예측 {good_pred} → MSE {good_loss:.4f}")
    print(f"나쁜 예측 {bad_pred} → MSE {bad_loss:.4f}")
    assert good_loss < bad_loss, "정답에 가까운 예측의 loss가 더 커야 할 이유가 없다"
    assert np.isclose(mse_loss(true_ratings, true_ratings), 0.0), "정답 그대로면 loss는 0이다"
    assert np.isclose(mse_loss(np.array([2.0, 7.0]), np.array([3.0, 5.0])), 2.5), "강의안 8절 값"

    print("\n--- 오차가 2배가 되면 loss는 4배 ---")
    for error in [1.0, 2.0, 4.0]:
        loss = mse_loss(np.array([3.0 + error]), np.array([3.0]))
        print(f"오차 {error:.0f} → MSE {loss:.0f}")


def demo_bce() -> None:
    """키워드 2. 정답 쪽 확률이 낮을수록 BCE가 급격히 커지고, clip이 log(0)을 막는다."""
    print("\n=== 키워드 2. BCE와 clip (부정 리뷰 판별) ===")

    # 정답: 1 = 부정 리뷰, 0 = 부정 아님. 예측은 "부정일 확률"이다.
    targets = np.array([1.0, 0.0, 1.0, 0.0])
    good_probs = np.array([0.9, 0.1, 0.8, 0.2])
    bad_probs = np.array([0.1, 0.9, 0.3, 0.8])
    good_loss = binary_cross_entropy(good_probs, targets)
    bad_loss = binary_cross_entropy(bad_probs, targets)
    print(f"정답:      {targets}")
    print(f"좋은 예측 {good_probs} → BCE {good_loss:.4f}")
    print(f"나쁜 예측 {bad_probs} → BCE {bad_loss:.4f}")
    assert good_loss < bad_loss, "정답 쪽 확률이 높은 예측의 loss가 더 작아야 한다"

    print("\n--- 정답이 1(부정)일 때, 예측 확률 p에 따른 loss ---")
    for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
        loss = binary_cross_entropy(np.array([p]), np.array([1.0]))
        print(f"p={p:<4} → BCE {loss:.4f}")

    print("\n--- clip: 확률이 정확히 0이나 1인데 틀린 경우 ---")
    # 부정 리뷰(1)에 확률 0.0, 부정 아닌 리뷰(0)에 확률 1.0을 줬다. 완전히 확신하고 틀렸다.
    sure_wrong = binary_cross_entropy(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    print(f"BCE {sure_wrong:.4f}  ← clip이 없으면 -log(0) = inf")
    assert np.isfinite(sure_wrong), "clip을 했으면 loss가 유한해야 한다"


def demo_cross_entropy() -> None:
    """키워드 3. 정답 클래스의 확률이 높을수록 loss가 작고, 큰 logits에서도 안전하다."""
    print("\n=== 키워드 3. Cross Entropy (감성 3분류) ===")

    # 리뷰 2개의 [부정, 중립, 긍정] logits와 정답 번호 (강의안 5절 값).
    logits = np.array([[2.0, 1.0, -0.5], [0.2, 0.3, 1.5]])
    targets = np.array([0, 2])
    sample_losses = cross_entropy_per_sample(logits, targets)
    print(f"logits:\n{logits}")
    print(f"정답: {[CLASS_NAMES[t] for t in targets]}")
    print(f"log 확률:\n{stable_log_softmax(logits)}")
    print(f"리뷰별 loss: {sample_losses}")
    print(f"평균 loss:   {np.mean(sample_losses):.4f}")
    assert sample_losses.shape == (2,), "리뷰마다 loss가 하나씩 있어야 한다"
    assert np.all(sample_losses >= 0), "Cross Entropy는 0 이상이다"

    print("\n--- 정답(부정) logit을 키우면 ---")
    for correct_logit in [0.0, 2.0, 4.0, 8.0]:
        one_review = np.array([[correct_logit, 1.0, -0.5]])
        loss = cross_entropy_per_sample(one_review, np.array([0]))[0]
        print(f"부정 logit {correct_logit:>3} → loss {loss:.4f}")

    print("\n--- 예측은 같아도 loss는 다르다 (강의안 11절 오해 2) ---")
    sure = np.array([[4.0, 0.0, -1.0]])
    unsure = np.array([[1.0, 0.9, -1.0]])
    for name, review_logits in [("확신", sure), ("애매", unsure)]:
        pred = CLASS_NAMES[np.argmax(review_logits)]
        loss = cross_entropy_per_sample(review_logits, np.array([0]))[0]
        print(f"{name} {review_logits[0]}: 예측 {pred}, loss {loss:.4f}")

    print("\n--- 큰 logits (강의안 9절) ---")
    big_loss = cross_entropy_per_sample(np.array([[1000.0, 1001.0, 1002.0]]), np.array([2]))[0]
    print(f"[1000, 1001, 1002], 정답 긍정 → loss {big_loss:.4f}  ← log-softmax라 overflow 없음")
    assert np.isfinite(big_loss), "큰 logits에서도 loss가 유한해야 한다"


def demo_one_hot() -> None:
    """키워드 4. 정답 번호와 one-hot은 같은 정보이고, 어느 쪽으로 계산해도 loss가 같다."""
    print("\n=== 키워드 4. 정답 표현: class index와 one-hot ===")

    targets = np.array([0, 2])
    one_hot = to_one_hot(targets, num_classes=len(CLASS_NAMES))
    print(f"class index: {targets}  shape {targets.shape}")
    print(f"one-hot:\n{one_hot}  shape {one_hot.shape}")
    assert one_hot.shape == (2, 3), "one-hot은 (리뷰 수, 클래스 수)여야 한다"
    assert np.allclose(one_hot.sum(axis=-1), 1.0), "행마다 1이 하나뿐이어야 한다"
    assert np.array_equal(np.argmax(one_hot, axis=-1), targets), (
        "argmax로 되돌리면 번호가 나와야 한다"
    )

    # one-hot으로 계산한 CE: 정답 자리만 1이라, 곱하고 더하면 정답 log 확률만 남는다.
    logits = np.array([[2.0, 1.0, -0.5], [0.2, 0.3, 1.5]])
    log_probs = stable_log_softmax(logits)
    loss_from_one_hot = -np.sum(one_hot * log_probs, axis=-1)
    loss_from_index = cross_entropy_per_sample(logits, targets)
    print(f"one-hot으로 계산한 loss: {loss_from_one_hot}")
    print(f"index로 계산한 loss:     {loss_from_index}  ← 같다")
    assert np.allclose(loss_from_one_hot, loss_from_index), "두 표현의 loss가 같아야 한다"


def demo_reduction() -> None:
    """키워드 5. none은 그대로, mean은 평균, sum은 합. 배치가 커지면 sum만 커진다."""
    print("\n=== 키워드 5. reduction ===")

    sample_losses = np.array([0.2, 0.8, 0.5])
    for reduction in REDUCTIONS:
        print(f"{reduction:>4}: {reduce_loss(sample_losses, reduction)}")
    assert np.allclose(reduce_loss(sample_losses, "none"), sample_losses), "none은 그대로여야 한다"
    assert np.isclose(reduce_loss(sample_losses, "mean"), 0.5), "평균은 0.5다"
    assert np.isclose(reduce_loss(sample_losses, "sum"), 1.5), "합은 1.5다"

    print("\n--- 같은 리뷰를 2배로 늘린 배치 ---")
    doubled = np.concatenate([sample_losses, sample_losses])
    print(f"mean: {reduce_loss(sample_losses):.4f} → {reduce_loss(doubled):.4f}  ← 그대로")
    print(
        f"sum:  {reduce_loss(sample_losses, 'sum'):.4f} → {reduce_loss(doubled, 'sum'):.4f}  ← 2배"
    )


DEMOS = [demo_mse, demo_bce, demo_cross_entropy, demo_one_hot, demo_reduction]


def main() -> None:
    # 소수점 4자리까지, 아주 작은 값은 지수 표기 대신 0으로 보여 준다.
    np.set_printoptions(precision=4, suppress=True)
    for demo in DEMOS:
        try:
            demo()
        except NotImplementedError as error:
            print(f"⏸ 건너뜀: {error}")


if __name__ == "__main__":
    main()
