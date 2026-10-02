from pathlib import Path

import pandas as pd

# 상품별 원본 엑셀 폴더
REVIEWS_DIR = Path("data/raw/reviews")

# 병합 결과 저장 위치 (정제 전, 개인정보만 제거)
OUTPUT_PATH = Path("data/interim/reviews_merged.csv")


# =====================================
# 1. 엑셀 파일 찾기
# =====================================

excel_files = sorted(
    REVIEWS_DIR.glob('*.xlsx')
)

print('=== 발견한 엑셀 파일 ===')

for file in excel_files:
    print(file.name)


print(
    '\n파일 개수:', len(excel_files)
)


# =====================================
# 2. 필수 컬럼
# =====================================

required_columns = [
    "평점",
    "리뷰제목",
    "리뷰내용"
]


# =====================================
# 3. 개인정보 컬럼
# =====================================

# 리뷰어를 알아볼 수 있는 컬럼은 병합 단계에서 바로 지운다
pii_columns = [
    "이름",
    "리뷰어번호"
]


# =====================================
# 4. 파일별로 읽고 product_id 붙이기
# =====================================

frames = []

for file in excel_files:

    df = pd.read_excel(file)

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file.name}에 필수 컬럼이 없습니다: {missing}"
        )

    # 파일 이름(product_01)을 상품 ID로 사용
    df.insert(
        0,
        "product_id",
        file.stem
    )

    frames.append(df)

    print(f"{file.stem}: {len(df)}개")


# =====================================
# 5. 병합
# =====================================

merged = pd.concat(
    frames,
    ignore_index=True
)


# =====================================
# 6. 개인정보 제거
# =====================================

merged = merged.drop(
    columns=pii_columns,
    errors="ignore"
)


# =====================================
# 7. 결과 확인
# =====================================

print('\n=== 병합 후 데이터 크기 ===')
print(merged.shape)

print('\n=== 컬럼명 ===')
print(merged.columns.to_list())

print('\n=== 상품별 평점 분포 ===')
print(
    pd.crosstab(
        merged["product_id"],
        merged["평점"]
    )
)


# =====================================
# 8. CSV 저장
# =====================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print(
    '\n저장 완료:',
    OUTPUT_PATH
)
