# Day02 · 1장 1강 — softmax와 temperature ✅

> 코드: `labs/day02/dl_ch01_softmax.py` · 실행: `uv run python labs/day02/dl_ch01_softmax.py`

## 한 문장 핵심

모델은 클래스마다 점수(logit)를 내고, softmax는 그 점수의 **차이**를 보고 합이 1인 확률로 바꾼다. temperature는 점수를 T로 나눠 그 차이를 줄이거나 벌린다.

| 키워드 | 한 문장 |
|---|---|
| logit과 확률 | logit은 음수도 되는 점수, 확률은 0~1 사이이고 합이 1 |
| softmax는 차이만 본다 | 모두에 100을 더해도 결과가 같고, 2를 곱하면 1등 확률이 커진다 |
| temperature | T가 작으면 1등 독식(일관된 답), 크면 평평(다양한 답) |

## 실행 전 예상

1. logits `[2.0, 0.5, -1.0]`에 100을 더하면 확률은 어떻게 될까?
2. temperature를 0.1로 낮추면? 10으로 높이면?

<details><summary>정답</summary>

1. 그대로 `[0.7856, 0.1753, 0.0391]`. 차이가 그대로라서.
2. T=0.1 → `[1, 0, 0]` (1등 독식), T=10 → `[0.38, 0.33, 0.28]` (거의 평평).

</details>

## 한 문장으로 답하기

**리뷰 태깅과 답글 초안을 LLM으로 만들 때 temperature를 각각 어떻게 둘까?**

<details><summary>모범 답</summary>

태깅은 매번 같은 답이 나와야 하니 낮게(0에 가깝게), 답글 초안은 표현이 다양해야 하니 조금 높게(0.7 안팎) 둔다.

</details>
