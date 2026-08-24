from pathlib import Path
import pandas as pd


REVIEWS_DIR = Path("data/raw/reviews")

excel_files = list(
    REVIEWS_DIR.glob('*.xlsx')
)

print('=== 발견한 엑셀 파일 ===')

for file in excel_files:
    print(file.name)


print(
    '\n파일 개수:', len(excel_files)
)



# =====================================
# 3. 필수 컬럼
# =====================================

required_columns = [
    "평점",
    "리뷰제목",
    "리뷰내용"
]



