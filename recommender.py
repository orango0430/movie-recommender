"""TF-IDF + 코사인 유사도 영화 추천기 (터미널용).

    python recommender.py --similar 기생충
    python recommender.py --similar "인터스텔라 (2014)"
    python recommender.py --watched "기생충,올드보이,추격자"
    python recommender.py --text "복수를 다룬 스릴러"
    python recommender.py --data data/sample_movies.csv --similar 기생충   # 샘플 35편 버전

문서 = 줄거리 명사·동사 + 장르 토큰 x2 + 키워드 토큰 x2 + 감독 토큰 x1
"""
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA = Path(__file__).parent / "data" / "movies.csv"

GENRE_WEIGHT = 2
KEYWORD_WEIGHT = 2
DIRECTOR_WEIGHT = 1

# 줄거리에 흔하지만 영화를 구분해 주지 못하는 말. 추천 이유에 자주 보이면 여기에 더한다
STOPWORDS = {
    "영화", "이야기", "사람", "자신", "시작", "모든", "그녀", "그들", "하나", "사이", "이후", "당시",
    "모습", "생각", "세상", "인생", "앞", "뒤", "속", "날", "때", "곳", "것", "수", "등", "중", "년",
    "하다", "되다", "있다", "없다", "보다", "같다", "가다", "오다", "알다", "만나다", "위하다",
}
CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "SL"}  # 일반·고유명사, 동사, 형용사, 외국어

_kiwi = None


def kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    out = []
    for t in kiwi().tokenize(text):
        if t.tag not in CONTENT_TAGS:
            continue
        form = t.form + "다" if t.tag in ("VV", "VA") else t.form
        # 한 글자 명사도 남긴다 (꿈, 왕, 돈, 섬). 의미 없는 것은 STOPWORDS로 뺀다
        if form in STOPWORDS or (t.tag == "SL" and len(t.form) < 2):
            continue
        out.append(form.lower())
    return out


def _split(value) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [v.strip() for v in re.split(r"[|,]", value) if v.strip()]


def build_document(row: pd.Series) -> list[str]:
    tokens = tokenize(row.get("overview", ""))
    for g in _split(row.get("genres")):
        tokens += [f"장르:{g}"] * GENRE_WEIGHT
    for k in _split(row.get("keywords")):
        tokens += [f"키워드:{k}"] * KEYWORD_WEIGHT
    for d in _split(row.get("director")):
        tokens += [f"감독:{d}"] * DIRECTOR_WEIGHT
    return tokens


@dataclass
class Result:
    index: int
    score: float
    reasons: list[str]


class Recommender:
    def __init__(self, movies: pd.DataFrame, max_df: float | None = None, min_df: int | None = None):
        self.movies = movies.reset_index(drop=True)
        n = len(self.movies)
        # 데이터가 적으면(샘플 35편) 흔한/희귀 단어 거르기를 끈다. 다 걸러져 버리기 때문
        self.max_df = max_df if max_df is not None else (0.5 if n >= 200 else 1.0)
        self.min_df = min_df if min_df is not None else (2 if n >= 200 else 1)
        self.docs = [build_document(r) for _, r in self.movies.iterrows()]
        self.vectorizer = TfidfVectorizer(analyzer=lambda doc: doc, max_df=self.max_df, min_df=self.min_df)
        self.matrix = self.vectorizer.fit_transform(self.docs)
        self.vocab = np.array(self.vectorizer.get_feature_names_out())

    # ---- 찾기 ----

    def find(self, query: str) -> int:
        """'인터스텔라' 또는 '인터스텔라 (2014)'. 같은 제목이 여럿이면 관객 수(있으면)가 많은 쪽, 없으면 최신작."""
        m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", query.strip())
        title, year = (m.group(1), int(m.group(2))) if m else (query.strip(), None)
        norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
        cand = self.movies[self.movies["title"].map(norm) == norm(title)]
        if year is not None:
            cand = cand[cand["year"].astype("Int64") == year]
        if cand.empty:
            partial = self.movies[self.movies["title"].map(norm).str.contains(re.escape(norm(title)))]
            hint = ", ".join(self.label(i) for i in partial.index[:5])
            raise KeyError(f"'{query}'을(를) 찾지 못했습니다." + (f" 비슷한 제목: {hint}" if hint else ""))
        if len(cand) > 1:
            key = "audience" if "audience" in cand and cand["audience"].notna().any() else "year"
            cand = cand.sort_values(key, ascending=False)
        return int(cand.index[0])

    def label(self, i: int) -> str:
        r = self.movies.loc[i]
        year = "" if pd.isna(r.get("year")) else f" ({int(r['year'])})"
        return f"{r['title']}{year}"

    # ---- 추천 ----

    def _rank(self, vec, exclude: set[int], top: int) -> list[Result]:
        scores = cosine_similarity(vec, self.matrix).ravel()
        order = [i for i in np.argsort(-scores) if i not in exclude and scores[i] > 0][:top]
        return [Result(int(i), float(scores[i]), self._reasons(vec, i)) for i in order]

    def _reasons(self, vec, i: int, k: int = 4) -> list[str]:
        """두 벡터에서 곱이 큰 단어 = 유사도에 가장 많이 기여한 단어"""
        contrib = vec.multiply(self.matrix[i]).toarray().ravel()
        idx = [j for j in np.argsort(-contrib)[:k] if contrib[j] > 0]
        return [self.vocab[j] for j in idx]

    def similar(self, query: str, top: int = 5) -> tuple[int, list[Result]]:
        i = self.find(query)
        return i, self._rank(self.matrix[i], {i}, top)

    def from_watched(self, titles: list[str], top: int = 5) -> tuple[list[int], list[Result]]:
        idx = [self.find(t) for t in titles]
        profile = csr_matrix(self.matrix[idx].mean(axis=0))
        return idx, self._rank(profile, set(idx), top)

    def search(self, text: str, top: int = 5) -> list[Result]:
        tokens = tokenize(text)
        # "스릴러", "애니메이션"처럼 장르 이름이 문장에 있으면 장르 토큰으로도 넣는다
        genres = {g for gs in self.movies["genres"].dropna() for g in _split(gs)}
        tokens += [f"장르:{g}" for g in genres if g in text]
        vec = self.vectorizer.transform([tokens])
        return self._rank(vec, set(), top)


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path} 이(가) 없습니다. 먼저 python collect_kmdb.py 로 수집하거나 --data 로 파일을 지정하세요.")
    return pd.read_csv(path, encoding="utf-8-sig")


def print_results(rec: Recommender, results: list[Result]):
    for n, r in enumerate(results, 1):
        row = rec.movies.loc[r.index]
        genres = row.get("genres", "")
        director = row.get("director", "")
        extra = f" · {director}" if isinstance(director, str) and director else ""
        print(f"{n}. {rec.label(r.index)}  [{genres}{extra}]  유사도 {r.score:.3f}")
        print(f"   이유: {', '.join(r.reasons) if r.reasons else '-'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="TF-IDF 영화 추천기")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--similar", metavar="제목", help='비슷한 영화. 동명 영화는 "제목 (연도)"')
    g.add_argument("--watched", metavar="제목들", help="본 영화들(쉼표 구분)을 바탕으로 추천")
    g.add_argument("--text", metavar="문장", help="문장으로 검색")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)

    rec = Recommender(load(a.data))
    print(f"영화 {len(rec.movies):,}편 · 단어 {len(rec.vocab):,}개 (max_df={rec.max_df}, min_df={rec.min_df})\n")
    try:
        if a.similar:
            i, res = rec.similar(a.similar, a.top)
            print(f"'{rec.label(i)}'와 비슷한 영화")
        elif a.watched:
            idx, res = rec.from_watched([t for t in a.watched.split(",") if t.strip()], a.top)
            print("본 영화: " + ", ".join(rec.label(i) for i in idx))
        else:
            res = rec.search(a.text, a.top)
            print(f"'{a.text}' 검색 결과")
    except KeyError as e:
        sys.exit(str(e.args[0]))
    print_results(rec, res)


if __name__ == "__main__":
    main()
