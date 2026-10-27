from pathlib import Path

import pandas as pd

# 00_merge_reviews.py 결과 (상품 3개 병합, 개인정보 제거)
INPUT_PATH = Path("data/interim/reviews_merged.csv")

# 전처리 결과 저장 위치
OUTPUT_PATH = Path("data/processed/reviews.csv")


# ==============================
# 1. CSV 읽기
# ==============================

df = pd.read_csv(INPUT_PATH)


# ==============================
# 2. 필요한 컬럼만 선택
# ==============================

# 상품·옵션·날짜는 나중에 상품별/옵션별 분석에 쓰려고 남긴다
df = df[
    [
        "product_id",
        "작성일",
        "상품옵션명",
        "평점",
        "리뷰제목",
        "리뷰내용"
    ]
].copy()


# ==============================
# 3. 리뷰제목 + 리뷰내용 → text
# ==============================

# Day01에서 확인: 내용 없이 제목만 있는 리뷰도 있어서 둘을 합쳐 쓴다
title = (
    df["리뷰제목"]
    .fillna("")
    .astype(str)
    .str.strip()
)

body = (
    df["리뷰내용"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["text"] = (
    title + " " + body
).str.strip()


# ==============================
# 4. 제목·내용이 모두 없는 행 제거
# ==============================

df = df[
    df["text"] != ""
]


# ==============================
# 5. 중복 리뷰 제거
# ==============================

df = df.drop_duplicates(
    subset=["text"]
)


# ==============================
# 6. 3점 제외
# ==============================

df = df[
    df["평점"].isin(
        [1, 2, 4, 5]
    )
]


# ==============================
# 7. label 생성
# ==============================

# 부정 리뷰 = 1
# 긍정 리뷰 = 0

df["label"] = (
    df["평점"] <= 2
).astype(int)


# ==============================
# 8. 저장할 컬럼 정리
# ==============================

df = df[
    [
        "product_id",
        "작성일",
        "상품옵션명",
        "평점",
        "text",
        "label"
    ]
]


# ==============================
# 9. 결과 확인
# ==============================

print("=== 전처리 후 데이터 수 ===")
print(len(df))


print("\n=== label 분포 ===")
print(
    df["label"]
    .value_counts()
    .sort_index()
)


print("\n=== 상품별 label 분포 ===")
print(
    pd.crosstab(
        df["product_id"],
        df["label"]
    )
)


print("\n=== 상위 5개 ===")
print(
    df.head()
)


# ==============================
# 10. CSV 저장
# ==============================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n저장 완료:",
    OUTPUT_PATH
)
