from pathlib import Path
import pandas as pd

FILE_PATH = Path("data/raw/reviews.xlsx")

df = pd.read_excel(FILE_PATH)

print('=== 데이터 크기 ===')
print(df.shape)

print('\n=== 컬럼명 ===')
print(df.columns.to_list())

print('\n=== 상위 5개 데이터 ===')
print(df.head())

print('\n=== 데이터 타입 ===')
print(df.dtypes)

print('\n=== 결측치 개수 ===')
print(df.isna().sum())


print('\n=== 평점 분포 ===')
print(
    df['평점']
    .value_counts()
    .sort_index()
)

print('\n=== 리뷰내용 결측치 ===')
print(
    df['리뷰내용']
    .isna()
    .sum()
)


