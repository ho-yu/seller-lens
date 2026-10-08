"""Day02 · 딥러닝 실전 1장 1강 — logits / softmax / log-softmax.

이 파일이 하는 일
    1강 핵심 키워드 5개를 코드로 확인한다. 사용자는 "✍️ 직접 작성" 계산 함수의 본문만 쓴다.
    demo 함수(예시 값, 출력, 검증)는 이미 작성되어 있고, 실행하면 결과를 보여 준다.
    여기서 만든 softmax / log-softmax는 2강(Cross Entropy)에서 다시 쓴다.

전체 흐름
    logits (N, 3) ─stable_softmax─────▶ 확률 (N, 3) ─argmax─▶ 예측 클래스 (N,)
                  └stable_log_softmax─▶ log 확률 (N, 3) ─────▶ 2강 Cross Entropy

키워드 ↔ 코드 (✍️ = 직접 작성, 괄호는 참고할 강의안 절)
    1  logit과 확률 (1~3절)          ✍️ predict_class, is_valid_probs  → demo_logit_vs_prob
    2  max trick, overflow (4~5절)   ✍️ naive_softmax                  → demo_overflow
    3  axis와 keepdims (5~6, 9절)    ✍️ softmax_no_keepdims            → demo_axis
    4  log-softmax, underflow (7절)  ✍️ stable_log_softmax             → demo_underflow
    5  temperature (강의 밖, 선택)   ✍️ softmax_with_temperature       → demo_temperature

진행 상황
    [x] stable_softmax
    [x] 1  predict_class, is_valid_probs
    [x] 2  naive_softmax
    [x] 3  softmax_no_keepdims
    [x] 4  stable_log_softmax
    [x] 5  softmax_with_temperature (선택)

작성 방법
    함수마다 docstring(무엇을 위한 함수인가, Args, Returns)을 읽고, "작성 순서" 주석의
    줄 모양을 그대로 쓰되 ___ 칸만 채운다.
    다 쓰면 TODO 주석과 raise 줄을 지우고 실행한다. 작성 전인 키워드는 건너뛰고 나머지는 실행된다.
    실행 결과와 맞춰 볼 정답 출력, 설명해 보기: labs/day02/TASKS.md
실행 (프로젝트 루트에서): uv run python labs/day02/dl_ch01_softmax.py
"""

import numpy as np

# =============================================================================
# 상수
# =============================================================================

# 리스트 위치가 곧 클래스 번호다 (0=부정, 1=중립, 2=긍정).
CLASS_NAMES = ["부정", "중립", "긍정"]

# 키워드 5: 비교할 temperature 값
TEMPERATURES = [0.1, 0.5, 1.0, 2.0, 10.0]


# =============================================================================
# 계산 함수 — ✍️ 직접 작성하는 곳. 2강에서 다시 쓰는 부품
# =============================================================================


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """logits를 합이 1인 확률로 바꾼다 (max trick 적용).

    Args:
        x: logits. 1차원 (C,) 또는 2차원 (N, C).
        axis: 확률의 합이 1이 되어야 하는 축. 클래스가 마지막 축이면 -1.

    Returns:
        x와 같은 shape의 확률 배열.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


# --- 키워드 1 ---------------------------------------------------------------


def predict_class(logits: np.ndarray) -> str:
    """✍️ 리뷰 하나의 logits로 예측 클래스 이름을 고른다.

    무엇을 위한 함수인가
        분류 모델의 마지막 단계다. 점수(logits) → 확률(softmax) → 가장 큰 확률의 클래스.
        1장 2강~2장에서 만드는 분류기가 리뷰마다 이 방식으로 감성을 예측한다.

    Args:
        logits: 리뷰 하나의 [부정, 중립, 긍정] 점수. shape (3,).

    Returns:
        CLASS_NAMES 중 하나 (예: "부정").
    """
    # TODO(키워드 1) 작성 순서. 참고: 강의안 3절
    # 1. probs: logits를 stable_softmax로 확률로 바꾼다.
    probs = stable_softmax(logits)
    # 2. 가장 큰 확률의 위치 번호(0, 1, 2)를 np.argmax로 구한다.
    class_index = np.argmax(probs)
    # 3. 그 번호로 CLASS_NAMES에서 이름을 꺼내 return한다.
    return CLASS_NAMES[class_index]


def is_valid_probs(probs: np.ndarray, axis: int = -1) -> bool:
    """✍️ 배열이 올바른 확률 분포인지 판단한다.

    무엇을 위한 함수인가
        softmax 결과가 확률이 되려면 지켜야 하는 조건을 코드로 쓴 것이다.
        강의안 3절 "가장 먼저 할 검증"이고, 앞으로 softmax를 고칠 때마다 이걸로 확인한다.

    Args:
        probs: 확률 배열. (C,) 또는 (N, C).
        axis: 합이 1이 되어야 하는 축.

    Returns:
        아래 세 조건을 모두 만족하면 True.
    """

    is_non_negative = np.all(probs >= 0)
    is_at_most_one = np.all(probs <= 1)
    sums_to_one = np.allclose(np.sum(probs, axis=axis), 1.0)

    # np.all은 numpy.bool을 돌려줘서, 선언한 반환 타입(파이썬 bool)에 맞게 bool()로 감싼다.
    return bool(is_non_negative and is_at_most_one and sums_to_one)


# --- 키워드 2 ---------------------------------------------------------------


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """✍️ max trick 없이 공식 그대로 계산한 softmax.

    무엇을 위한 함수인가
        max trick이 없으면 큰 logit에서 inf → nan으로 깨진다는 것을 보여 줄 비교 대상이다.
        demo_overflow에서만 쓰고, 서비스 코드에서는 쓰지 않는다.

    Args:
        x: logits. 1차원 (C,) 또는 2차원 (N, C).
        axis: 확률의 합이 1이 되어야 하는 축.

    Returns:
        x와 같은 shape의 확률 배열. logit이 크면 nan이 섞여 나온다 (그게 정상이다).
    """

    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


# --- 키워드 3 ---------------------------------------------------------------


def softmax_no_keepdims(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """✍️ keepdims=True를 빼고 계산한 softmax (일부러 틀리게 만든 비교용).

    무엇을 위한 함수인가
        keepdims가 무엇을 막아 주는지 보려는 비교 대상이다. 1차원에서는 맞게 나오고,
        (N, C)에서는 shape에 따라 에러가 나거나 에러 없이 조용히 틀린다.

    Args:
        x: logits. 1차원 (C,) 또는 2차원 (N, C).
        axis: 확률의 합이 1이 되어야 하는 축.

    Returns:
        계산 결과 배열 (2차원이면 틀린 값일 수 있다).
    """

    shifted = x - np.max(x, axis=axis)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis)


# --- 키워드 4 ---------------------------------------------------------------


def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """✍️ softmax를 거치지 않고 log 확률을 바로 계산한다.

    무엇을 위한 함수인가
        2강 Cross Entropy에 넣을 log 확률을 구한다. np.log(softmax)로 구하면 아주 작은 확률이
        0이 되고 log(0) = -inf가 된다(underflow). 이 함수는 그 문제 없이 정확한 값을 낸다.
        2강에서 실제로 다시 쓰는 부품이다.

    공식
        log(softmax(x)_i) = shifted_i - log(Σ exp(shifted_j)),  shifted = x - max(x)
        softmax의 "나누기"가 log를 씌우면 "빼기"가 된다 (log(a / b) = log a - log b).

    Args:
        x: logits. 1차원 (C,) 또는 2차원 (N, C).
        axis: 클래스 축. 클래스가 마지막 축이면 -1.

    Returns:
        x와 같은 shape의 log 확률 배열. 모든 값이 0 이하이고, np.exp를 씌우면 softmax와 같다.
    """

    shifted = x - np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


# --- 키워드 5 ---------------------------------------------------------------


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """✍️ temperature를 적용한 softmax (강의 밖, 선택).

    무엇을 위한 함수인가
        logits를 T로 나눠 확률 분포를 뾰족하게(T<1) 또는 평평하게(T>1) 만든다.
        LLM API의 temperature 파라미터가 하는 일이 바로 이것이다 (5장에서 정할 때 근거).

    Args:
        logits: (C,) 또는 (N, C).
        temperature: 0보다 큰 실수 T.

    Returns:
        logits와 같은 shape의 확률 배열.
    """

    return stable_softmax(logits / temperature)


# =============================================================================
# 키워드 확인 — 이미 작성되어 있다. 실행해서 결과를 본다.
# =============================================================================
# 예시 값을 넣어 위 함수를 호출하고, 결과를 출력하고, assert로 맞는지 확인한다.


def demo_logit_vs_prob() -> None:
    """키워드 1. logit은 점수, softmax는 그 점수를 합이 1인 확률로 바꾼다."""
    print("\n=== 키워드 1. logit과 확률 ===")

    # 모델이 리뷰 하나에 이런 점수를 냈다고 가정한다 ([부정, 중립, 긍정]).
    logits = np.array([2.0, 0.5, -1.0])
    probs = stable_softmax(logits)
    print(f"logits:    {logits}")
    print(f"확률:      {probs}")
    print(f"확률의 합: {np.sum(probs):.4f}")
    print(f"+100:      {stable_softmax(logits + 100)}  ← 차이가 그대로라 결과도 그대로")
    print(f"×2:        {stable_softmax(logits * 2)}  ← 차이가 벌어져 1등 확률이 커짐")

    print(f"예측:      {predict_class(logits)}")

    bad_probs = np.array([0.7, 0.5, -0.2])
    print(f"is_valid_probs(확률)            → {is_valid_probs(probs)}")
    print(f"is_valid_probs({bad_probs}) → {is_valid_probs(bad_probs)}")
    assert is_valid_probs(probs), "softmax 결과가 확률 조건을 만족하지 않는다"
    assert not is_valid_probs(bad_probs), "음수가 있는 배열을 확률로 판단했다"
    assert predict_class(logits) == "부정", "가장 큰 logit은 부정인데 다른 클래스를 골랐다"


def demo_overflow() -> None:
    """키워드 2. 공식 그대로 계산하면 큰 logit에서 inf → nan이 된다."""
    print("\n=== 키워드 2. max trick과 overflow ===")

    normal_logits = np.array([2.0, 0.5, -1.0])
    assert np.allclose(naive_softmax(normal_logits), stable_softmax(normal_logits)), (
        "보통 값에서는 naive와 stable이 같아야 한다"
    )
    print(f"보통 값 {normal_logits}: naive와 stable이 같다")

    big_logits = np.array([1000.0, 1001.0, 1002.0])
    # overflow 경고를 끄고 결과만 본다.
    with np.errstate(over="ignore", invalid="ignore"):
        naive_probs = naive_softmax(big_logits)
    stable_probs = stable_softmax(big_logits)
    print(f"큰 값 {big_logits}")
    print(f"  naive:  {naive_probs}")
    print(f"  stable: {stable_probs}  ← [0, 1, 2]의 softmax와 같다")
    assert np.all(np.isfinite(stable_probs)), "stable 결과에 inf / nan이 있다"

    print(f"exp가 inf가 되는 경계 (float64): {np.log(np.finfo(np.float64).max):.2f}")
    print(f"exp가 inf가 되는 경계 (float32): {np.log(np.finfo(np.float32).max):.2f}")


def demo_axis() -> None:
    """키워드 3. (N, C)에서는 리뷰마다(행마다) 합이 1이어야 한다."""
    print("\n=== 키워드 3. axis와 keepdims ===")

    # 리뷰 3개 × [부정, 중립, 긍정]. 한 행이 리뷰 하나다 (부정 / 애매 / 긍정 리뷰).
    batch_logits = np.array([[2.0, 0.5, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 3.0]])

    probs_ok = stable_softmax(batch_logits, axis=-1)
    probs_wrong = stable_softmax(batch_logits, axis=0)
    print(f"axis=-1 확률:\n{probs_ok}")
    print(f"axis=-1 행마다 합: {np.sum(probs_ok, axis=-1)}")
    print(f"axis=0  확률:\n{probs_wrong}")
    print(f"axis=0  행마다 합: {np.sum(probs_wrong, axis=-1)}  ← 에러 없이 틀림")
    assert np.allclose(np.sum(probs_ok, axis=-1), 1.0), "리뷰마다 합이 1이 아니다"

    print("\n--- keepdims ---")
    row_max = np.max(batch_logits, axis=-1)
    row_max_kept = np.max(batch_logits, axis=-1, keepdims=True)
    print(f"np.max(axis=-1) shape:                {row_max.shape}")
    print(f"np.max(axis=-1, keepdims=True) shape: {row_max_kept.shape}")

    one_review = batch_logits[0]
    assert np.allclose(softmax_no_keepdims(one_review), stable_softmax(one_review)), (
        "1차원에서는 keepdims가 없어도 같아야 한다"
    )
    print("1차원 (3,): keepdims 없어도 결과가 같다")

    no_keep = softmax_no_keepdims(batch_logits)
    print(f"(3, 3) keepdims 없음:\n{no_keep}")
    print(f"행마다 합: {np.sum(no_keep, axis=-1)}  ← 리뷰 수 = 클래스 수라 에러 없이 틀림")
    assert not np.allclose(no_keep, probs_ok), "keepdims 없이도 맞게 나왔다 (예상과 다름)"

    try:
        softmax_no_keepdims(batch_logits[:2])
    except ValueError as error:
        print(f"(2, 3) keepdims 없음: ValueError ← shape이 안 맞아 에러\n  {error}")


def demo_underflow() -> None:
    """키워드 4. log(softmax)는 작은 확률을 0으로 뭉개고, log-softmax는 정확하다."""
    print("\n=== 키워드 4. log-softmax와 underflow ===")

    logits = np.array([2.0, 1.0, -0.5])
    log_probs = stable_log_softmax(logits)
    print(f"log 확률:      {log_probs}")
    print(f"exp(log 확률): {np.exp(log_probs)}  ← softmax와 같다")
    assert np.allclose(np.exp(log_probs), stable_softmax(logits)), "exp(log-softmax) ≠ softmax"
    assert np.all(log_probs <= 0), "log 확률은 0 이하여야 한다"

    extreme_logits = np.array([0.0, -1000.0])
    # log(0) 경고를 끄고 결과만 본다.
    with np.errstate(divide="ignore"):
        log_of_probs = np.log(stable_softmax(extreme_logits))
    print(f"극단 값 {extreme_logits}")
    print(f"  log(softmax): {log_of_probs}  ← 작은 확률이 0이 되어 -inf")
    print(f"  log-softmax:  {stable_log_softmax(extreme_logits)}  ← 정확")


def demo_temperature() -> None:
    """키워드 5. logits를 T로 나누면 분포가 뾰족해지거나(T<1) 평평해진다(T>1)."""
    print("\n=== 키워드 5. temperature ===")

    logits = np.array([2.0, 0.5, -1.0])
    for temperature in TEMPERATURES:
        print(f"T={temperature:>4}: {softmax_with_temperature(logits, temperature)}")
    assert np.allclose(softmax_with_temperature(logits, 1.0), stable_softmax(logits)), (
        "T=1이면 그냥 softmax와 같아야 한다"
    )


DEMOS = [demo_logit_vs_prob, demo_overflow, demo_axis, demo_underflow, demo_temperature]


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
