"""분류 평가 함수. Day07 실습(`labs/day07/`)에서 직접 쓴 `confusion_counts`를 옮겨 왔다."""


def confusion_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int]:
    """부정(1)을 기준으로 (tp, fp, fn)을 센다 (Day07).

    tp: 정답 1, 예측 1 / fp: 정답 0, 예측 1 / fn: 정답 1, 예측 0
    """
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    return tp, fp, fn


def recall_precision_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """(tp, fp, fn)에서 재현율, 정밀도, F1을 계산한다. 나누는 수가 0이면 0.0."""
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return recall, precision, f1
