"""KMDb 응답 해석과 정제 규칙. 응답 모양은 KMDb 안내 문서 기준이며, --sample로 실제 구조를 확인한 뒤 맞춘다."""
import collect_kmdb as c

PLOT = "가족 모두가 백수인 기택네 장남 기우가 고액 과외 면접을 위해 박사장네 집에 발을 들이면서 두 가족의 만남이 걷잡을 수 없는 사건으로 번진다."


def record(**over):
    r = {
        "DOCID": "K12345",
        "title": " !HS 기생충 !HE ",
        "prodYear": "2019",
        "genre": "드라마,스릴러",
        "plots": {"plot": [{"plotLang": "한국어", "plotText": PLOT}, {"plotLang": "영어", "plotText": "Greed and class..."}]},
        "directors": {"director": [{"directorNm": "봉준호", "directorId": "00001"}]},
        "keywords": "가족,계급,반지하",
        "nation": "대한민국",
        "audiAcc": "10313087",
        "kmdbUrl": "https://www.kmdb.or.kr/db/kor/detail/movie/K/12345",
        "type": "극영화",
    }
    r.update(over)
    return r


def test_to_row_cleans_and_maps():
    row = c.to_row(record())
    assert row["title"] == "기생충"
    assert row["overview"] == PLOT  # 한국어 줄거리만
    assert row["genres"] == "드라마|스릴러"
    assert row["keywords"] == "가족|계급|반지하"
    assert row["director"] == "봉준호"
    assert row["year"] == 2019 and row["audience"] == 10313087


def test_single_object_instead_of_list():
    r = record(plots={"plot": {"plotLang": "한국어", "plotText": PLOT}},
               directors={"director": {"directorNm": "봉준호"}})
    row = c.to_row(r)
    assert row["overview"] == PLOT and row["director"] == "봉준호"


def test_html_and_empty_genre():
    row = c.to_row(record(title="<b>기생충</b>", genre=""))
    assert row["title"] == "기생충" and row["genres"] == "기타"


def test_refine_steps():
    rows = [
        c.to_row(record()),
        c.to_row(record()),  # 같은 docid
        c.to_row(record(DOCID="K99999", audiAcc="")),  # 재개봉판: 제목·연도 같음
        c.to_row(record(DOCID="K2", title="짧은 영화", plots={"plot": [{"plotLang": "한국어", "plotText": "짧다"}]})),
        c.to_row(record(DOCID="K3", title="제외", genre="에로,드라마")),
    ]
    df, steps = c.refine(rows)
    assert [n for _, n in steps] == [5, 4, 2, 1, 1]
    assert df.iloc[0]["id"] == "K12345"  # 관객 수 있는 쪽이 남음
