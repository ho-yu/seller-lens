from pathlib import Path

import pandas as pd

# 00_merge_reviews.py 결과 (상품 3개 병합, 개인정보 제거)
INPUT_PATH = Path('data/interim/reviews_merged.csv')

df = pd.read_csv(INPUT_PATH)

# 필요한 컬럼만 선택
df = df[
    [
        '평점'
        ,'리뷰내용'
    ]
]

print('=== 1. 원본 데이터 ===')
print(f'전체 : {len(df)}')

print(
    df['평점']
    .value_counts()
    .sort_index()
)

# 결측치 제거
df = df.dropna(
    subset=['리뷰내용']
)

print('\n=== 2. 결측치 제거 후 ===')

print(f'전체 : {len(df)}')

print(
    df['평점']
    .value_counts()
    .sort_index()
)

# 문자열 정리
df["리뷰내용"] = (
    df["리뷰내용"]
    .astype(str)
    .str.strip()
)


print("\n=== 3. 문자열 정리 후 ===")
print("전체:", len(df))


# 3점 제외
df = df[
    df["평점"].isin(
        [1, 2, 4, 5]
    )
]


print("\n=== 4. 3점 제외 후 ===")
print("전체:", len(df))

print(
    df["평점"]
    .value_counts()
    .sort_index()
)


# ==================================
# 중복 제거 전 부정 리뷰 확인
# ==================================

negative_before = df[
    df["평점"] <= 2
]


print("\n=== 5. 중복 제거 전 부정 리뷰 ===")
print("부정 리뷰 수:", len(negative_before))


print(
    negative_before[
        [
            "평점",
            "리뷰내용"
        ]
    ]
)


# ==================================
# 중복 제거
# ==================================

df_unique = df.drop_duplicates(
    subset=["리뷰내용"]
)


print("\n=== 6. 중복 제거 후 ===")
print("전체:", len(df_unique))


negative_after = df_unique[
    df_unique["평점"] <= 2
]


print(
    "부정 리뷰 수:",
    len(negative_after)
)


print(
    negative_after[
        [
            "평점",
            "리뷰내용"
        ]
    ]
)