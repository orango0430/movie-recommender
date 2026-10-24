"""공공데이터포털 '한국영상자료원_영화상세정보' CSV(통합 파일) 변환."""
import pandas as pd

import collect_kmdb as c


def test_csv_row_mapping():
    df = pd.DataFrame([{
        "제명": "브로커", "제작년도": 2022, "장르": "드라마,가족", "키워드": "세탁소,베이비박스,입양",
        "줄거리": "<b>베이비 박스</b>에 놓인 아기를 둘러싸고 벌어지는 이야기. " * 3,
        "크레딧명": "감독,출연,출연,감독", "인명": "고레에다 히로카즈,송강호,강동원,고레에다 히로카즈",
        "제작국가": "대한민국", "길이구분": "장편", "KMDB영화정보관리번호": "K31502",
    }])
    row = c.csv_to_rows(df)[0]
    assert row["title"] == "브로커" and row["year"] == 2022
    assert row["genres"] == "드라마|가족" and row["keywords"] == "세탁소|베이비박스|입양"
    assert row["director"] == "고레에다 히로카즈"  # 감독만, 중복 없이
    assert "<b>" not in row["overview"]
    assert row["url"].endswith("/movie/K/31502")
