from pathlib import Path

import pandas as pd
import pytest

from recommender import Recommender, build_document, tokenize

SAMPLE = Path(__file__).parents[1] / "data" / "sample_movies.csv"


@pytest.fixture(scope="module")
def rec():
    return Recommender(pd.read_csv(SAMPLE))


def test_tokenize_drops_stopwords_and_particles():
    toks = tokenize("형사들이 조직 범죄를 소탕하는 이야기")
    assert {"형사", "조직", "범죄"} <= set(toks)
    assert "이야기" not in toks and "들" not in toks


def test_character_names_are_dropped():
    # 등장인물 이름(고유명사)이 같다는 이유로 엉뚱한 영화가 이어지지 않게
    assert "소영" not in tokenize("소영은 아기를 베이비박스에 두고 간다")


def test_name_patterns_removed_before_analysis():
    toks = tokenize("사이는 좋은 기택(송강호) 가족. 장남 기우(최우식)와 ‘상현’이 과외 자리를 얻는다")
    assert not {"기택", "기우", "상", "현", "송강호"} & set(toks)
    assert {"가족", "장남", "과외"} <= set(toks)
    # 이름 뒤 조사가 떨어져 단어로 남지 않는다
    assert not {"을", "과", "이"} & set(tokenize("엄마가 아기 ‘우성’을 찾는다. 형사 ‘수진’과 함께"))
    # 띄어쓰기가 있는 따옴표(대사)는 이름이 아니라서 지우지 않는다
    assert "가족" in tokenize("“우리 가족은 행복해야 해” 그는 결심한다")


def test_meta_keywords_are_not_used():
    doc = build_document(pd.Series({"overview": "", "genres": "",
                                    "keywords": "빈부격차|황금종려상|소설원작|해외영화제수상작|반지하"}))
    assert set(doc) == {"키워드:빈부격차", "키워드:반지하"}


def test_document_weights():
    doc = build_document(pd.Series({"overview": "", "genres": "범죄|액션", "keywords": "복수", "director": "봉준호"}))
    assert doc.count("장르:범죄") == 2 and doc.count("키워드:복수") == 2 and doc.count("감독:봉준호") == 1


def test_similar_excludes_itself_and_finds_sequel(rec):
    i, res = rec.similar("겨울왕국")
    titles = [rec.movies.loc[r.index, "title"] for r in res]
    assert rec.movies.loc[i, "title"] == "겨울왕국" and "겨울왕국" not in titles
    assert titles[0] == "겨울왕국 2"


def test_find_with_year_and_missing(rec):
    assert rec.movies.loc[rec.find("인터스텔라 (2014)"), "year"] == 2014
    with pytest.raises(KeyError):
        rec.find("인터스텔라 (1999)")


def test_text_search(rec):
    res = rec.search("복수를 다룬 스릴러", top=2)
    assert {rec.movies.loc[r.index, "title"] for r in res} == {"올드보이", "친절한 금자씨"}


def test_watched_excludes_watched(rec):
    idx, res = rec.from_watched(["인터스텔라", "그래비티"])
    assert not set(idx) & {r.index for r in res}


def test_large_data_uses_df_filters():
    plots = ["형사가 범인을 쫓는다", "우주선이 행성에 간다", "가족이 여행을 떠난다"]
    big = pd.DataFrame({"title": [f"영화{i}" for i in range(250)], "year": 2020, "genres": "드라마",
                        "overview": [plots[i % 3] + (" 외계인" if i == 0 else "") for i in range(250)]})
    r = Recommender(big)
    assert r.max_df == 0.5 and r.min_df == 2
    assert "장르:드라마" not in r.vocab  # 모든 영화에 있는 토큰은 걸러짐 (max_df)
    assert "외계인" not in r.vocab      # 한 영화에만 있는 단어도 걸러짐 (min_df)
    assert "형사" in r.vocab
