"""Day02 · 딥러닝 실전 1장 1강 — logits / softmax / log-softmax.

이 파일이 하는 일
    1강 핵심 키워드 5개를 코드로 확인한다. 강의안 코드를 참고해 직접 작성하고,
    실행해서 결과를 보며 TASKS.md의 "설명해 보기"에 답한다.
    여기서 만든 softmax / log-softmax는 2강(Cross Entropy)에서 다시 쓴다.

전체 흐름
    logits (N, 3) ─stable_softmax─────▶ 확률 (N, 3) ─argmax─▶ 예측 클래스 (N,)
                  └stable_log_softmax─▶ log 확률 (N, 3) ─────▶ 2강 Cross Entropy

키워드 ↔ 코드 (TASKS.md 순서, 괄호는 참고할 강의안 절)
    1  logit과 확률 (3절)            stable_softmax                  → demo_logit_vs_prob
    2  max trick, overflow (4절)     naive_softmax, stable_softmax   → demo_overflow
    3  axis와 keepdims (6, 9절)      stable_softmax(axis=...)        → demo_axis
    4  log-softmax, underflow (7절)  stable_log_softmax              → demo_underflow
    5  temperature (부가, 강의 밖)   stable_softmax(logits / T)      → demo_temperature

진행 상황
    [x] 1-① stable_softmax 구현
    [ ] 1-② demo_logit_vs_prob  ◀ 지금
        리뷰 하나의 [부정, 중립, 긍정] logits에 softmax를 적용해 출력하고,
        강의안 3절 "가장 먼저 할 검증"의 assert 4개를 내 데이터로 쓴다.
    [ ] 2  naive_softmax, demo_overflow
    [ ] 3  demo_axis
    [ ] 4  stable_log_softmax, demo_underflow
    [ ] 5  demo_temperature (부가)

키워드 설명 · 직접 작성할 것 · 설명해 보기: labs/day02/TASKS.md
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
# 계산 함수 — 2강에서 다시 쓰는 부품
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


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """max trick 없이 공식 그대로 계산한 softmax. 키워드 2에서 비교용으로만 쓴다."""
    # TODO(키워드 2) 강의안 3절의 세 단계(exp → 합 → 나누기)를 함수로.
    #                stable_softmax에서 max trick 줄만 뺀 모양이다.
    raise NotImplementedError


def stable_log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """softmax를 거치지 않고 log 확률을 바로 계산한다.

    Returns:
        x와 같은 shape의 log 확률 배열 (값은 0 이하).
    """
    # TODO(키워드 4) 강의안 7절의 stable_log_softmax 참고.
    raise NotImplementedError


# =============================================================================
# 키워드 확인 — 키워드 하나에 함수 하나
# =============================================================================


def demo_logit_vs_prob() -> None:
    """키워드 1. logit은 점수, softmax는 그 점수를 합이 1인 확률로 바꾼다."""
    # TODO(키워드 1) TASKS.md 1번 "직접 작성" 참고 (강의안 3절)


def demo_overflow() -> None:
    """키워드 2. 공식 그대로 계산하면 큰 logit에서 inf → nan이 된다."""
    # TODO(키워드 2) TASKS.md 2번 "직접 작성" 참고 (강의안 4절)


def demo_axis() -> None:
    """키워드 3. (N, C)에서는 리뷰마다(행마다) 합이 1이어야 한다."""
    # TODO(키워드 3) TASKS.md 3번 "직접 작성" 참고 (강의안 6, 9절)


def demo_underflow() -> None:
    """키워드 4. log(softmax)는 작은 확률을 0으로 뭉개고, log-softmax는 정확하다."""
    # TODO(키워드 4) TASKS.md 4번 "직접 작성" 참고 (강의안 7절)


def demo_temperature() -> None:
    """키워드 5. logits를 T로 나누면 분포가 뾰족해지거나(T<1) 평평해진다(T>1)."""
    # TODO(키워드 5, 부가) TASKS.md 5번 "직접 작성" 참고


def main() -> None:
    # 소수점 4자리까지, 아주 작은 값은 지수 표기 대신 0으로 보여 준다.
    np.set_printoptions(precision=4, suppress=True)
    demo_logit_vs_prob()
    demo_overflow()
    demo_axis()
    demo_underflow()
    demo_temperature()


if __name__ == "__main__":
    main()
