"""TF-IDF + 코사인 유사도 영화 추천기 (터미널용).

    python recommender.py --similar 기생충
    python recommender.py --similar "인터스텔라 (2014)"
    python recommender.py --watched "기생충,올드보이,추격자"
    python recommender.py --text "복수를 다룬 스릴러"
    python recommender.py --data data/sample_movies.csv --similar 기생충   # 샘플 35편 버전

문서 = 줄거리 일반명사·동사·형용사 + 장르 토큰 x2 + 키워드 토큰 x2 + 감독 토큰 x1
내부 동작은 docs/HOW_IT_WORKS.md
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
    # 독립영화 3,991편 점검에서 추천 이유로 자주 나왔지만 내용과 무관했던 말
    "특별", "과정", "반", "지나다", "보이다", "돌아오다", "살다", "늦다", "다녀오다", "후배",
    # 문장 검색에서 "~를 다룬", "~를 그린"처럼 내용이 아니라 말투로 들어오는 동사
    "다루다", "그리다",
}
# 영화 내용이 아니라 출처·이력을 말하는 KMDb 키워드. 가중치 2배라 남겨 두면 주제를 덮는다
# (기생충의 키워드 20개 중 해외영화제수상작, 황금종려상, 아카데미상이 가족·빈부격차보다 강하게 작용했다)
META_KEYWORDS = {
    "실화바탕", "시리즈", "극장판", "리메이크", "옴니버스영화", "사사로운영화리스트", "넷플릭스",
    "황금종려상", "아카데미상", "모션캡처", "1인2역", "1인다역", "3D", "IMAX",
}
_META_SUFFIX = ("원작", "수상작", "영화제")


def is_meta_keyword(k: str) -> bool:
    return k in META_KEYWORDS or k.endswith(_META_SUFFIX)


# KMDb 줄거리는 등장인물을 '기택(송강호)', '상현'처럼 쓴다. 이름은 형태소 분석기가 일반명사로 보기도 해서
# (기우, 동수) 품사만으로는 다 걸러지지 않는다. 괄호 앞 단어와 짧은 따옴표 안 단어를 먼저 지운다
_NAME_WITH_ACTOR = re.compile(r"[^\s(（]+\s?[(（][^)）]{1,20}[)）]")
_QUOTED_NAME = re.compile(r"[‘'\"“]([^’'\"”\s]{1,6})[’'\"”]")
# 일반명사, 동사, 형용사, 외국어. 고유명사(NNP)는 뺀다: 대부분 등장인물 이름(소영, 경수)이라
# 이름이 같은 엉뚱한 영화끼리 이어졌다. 지명·주제어는 키워드 토큰으로 따로 들어간다
CONTENT_TAGS = {"NNG", "VV", "VA", "SL"}

_kiwi = None


def kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def clean_plot(text: str) -> str:
    """인물 이름을 '사람'으로 바꾼다. 그냥 지우면 뒤의 조사('우성'을 → 을)가 떨어져 단어로 잡힌다.
    TF-IDF와 임베딩(embedder.py)이 같은 규칙을 쓴다."""
    if not isinstance(text, str):
        return ""
    return _QUOTED_NAME.sub("사람", _NAME_WITH_ACTOR.sub("사람", text))


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = clean_plot(text)
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
        if not is_meta_keyword(k):
            tokens += [f"키워드:{k}"] * KEYWORD_WEIGHT
    for d in _split(row.get("director")):
        tokens += [f"감독:{d}"] * DIRECTOR_WEIGHT
    return tokens


@dataclass
class Result:
    index: int
    score: float
    reasons: list[str]


METHODS = ("tfidf", "embed", "hybrid")
# 혼합 방식에서 임베딩 점수의 비중. 나머지는 TF-IDF(장르·키워드·감독 포함)
HYBRID_ALPHA = 0.7


class Recommender:
    """method: tfidf(단어 겹침) | embed(줄거리 뜻, embedder.py) | hybrid(둘을 섞음).
    embeddings는 embedder.load_or_build()로 만든 '영화 수 × 768' 행렬. 없으면 tfidf만 쓸 수 있다."""

    def __init__(self, movies: pd.DataFrame, max_df: float | None = None, min_df: int | None = None,
                 embeddings: np.ndarray | None = None):
        self.movies = movies.reset_index(drop=True)
        if embeddings is not None and len(embeddings) != len(self.movies):
            raise ValueError("임베딩 행 수가 영화 수와 다릅니다. embedder.py로 다시 만드세요")
        self.embeddings = embeddings
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

    def _check(self, method: str):
        if method not in METHODS:
            raise ValueError(f"method는 {', '.join(METHODS)} 중 하나")
        if method != "tfidf" and self.embeddings is None:
            raise ValueError("임베딩이 없습니다. 먼저 python embedder.py --data <데이터> 를 실행하세요")

    def _rank(self, vec, emb_vec, exclude: set[int], top: int, method: str = "tfidf") -> list[Result]:
        """vec: TF-IDF 벡터, emb_vec: 길이 1인 임베딩 벡터. 추천 이유는 방식과 상관없이 겹치는 단어로 보여 준다."""
        tfidf = cosine_similarity(vec, self.matrix).ravel()
        if method == "tfidf":
            scores = tfidf
        else:
            emb = self.embeddings @ emb_vec
            scores = emb if method == "embed" else HYBRID_ALPHA * emb + (1 - HYBRID_ALPHA) * tfidf
        order = [i for i in np.argsort(-scores) if i not in exclude and scores[i] > 0][:top]
        return [Result(int(i), float(scores[i]), self._reasons(vec, i)) for i in order]

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n else v

    def _reasons(self, vec, i: int, k: int = 4) -> list[str]:
        """두 벡터에서 곱이 큰 단어 = 유사도에 가장 많이 기여한 단어"""
        contrib = vec.multiply(self.matrix[i]).toarray().ravel()
        idx = [j for j in np.argsort(-contrib)[:k] if contrib[j] > 0]
        return [self.vocab[j] for j in idx]

    def similar(self, query: str, top: int = 5, method: str = "tfidf") -> tuple[int, list[Result]]:
        return self.similar_index(self.find(query), top, method)

    def similar_index(self, i: int, top: int = 5, method: str = "tfidf") -> tuple[int, list[Result]]:
        """제목 대신 행 번호로. 평가(evaluate.py)에서 수천 편을 돌릴 때 쓴다."""
        self._check(method)
        emb = None if method == "tfidf" else self.embeddings[i]
        return i, self._rank(self.matrix[i], emb, {i}, top, method)

    def from_watched(self, titles: list[str], top: int = 5, method: str = "tfidf") -> tuple[list[int], list[Result]]:
        self._check(method)
        idx = [self.find(t) for t in titles]
        profile = csr_matrix(self.matrix[idx].mean(axis=0))
        emb = None if method == "tfidf" else self._unit(self.embeddings[idx].mean(axis=0))
        return idx, self._rank(profile, emb, set(idx), top, method)

    def search(self, text: str, top: int = 5, method: str = "tfidf") -> list[Result]:
        self._check(method)
        tokens = tokenize(text)
        # "스릴러", "애니메이션"처럼 장르 이름이 문장에 있으면 장르 토큰으로도 넣는다
        genres = {g for gs in self.movies["genres"].dropna() for g in _split(gs)}
        tokens += [f"장르:{g}" for g in genres if g in text]
        vec = self.vectorizer.transform([tokens])
        emb = None
        if method != "tfidf":
            from embedder import embed_query  # 모델(torch)은 임베딩을 쓸 때만 불러온다
            emb = embed_query(text)
        return self._rank(vec, emb, set(), top, method)


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
    ap = argparse.ArgumentParser(description="영화 추천기 (TF-IDF / 문장 임베딩)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--similar", metavar="제목", help='비슷한 영화. 동명 영화는 "제목 (연도)"')
    g.add_argument("--watched", metavar="제목들", help="본 영화들(쉼표 구분)을 바탕으로 추천")
    g.add_argument("--text", metavar="문장", help="문장으로 검색")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--method", choices=METHODS, default="tfidf",
                    help="tfidf: 단어 겹침, embed: 줄거리 뜻(임베딩), hybrid: 둘을 섞음")
    a = ap.parse_args(argv)

    movies = load(a.data)
    emb = None
    if a.method != "tfidf":
        from embedder import load_or_build
        emb = load_or_build(movies.reset_index(drop=True), a.data)
        if emb is None:
            sys.exit(f"임베딩이 없습니다. 먼저 python embedder.py --data {a.data} 를 실행하세요 (한 번만)")
    rec = Recommender(movies, embeddings=emb)
    print(f"영화 {len(rec.movies):,}편 · 단어 {len(rec.vocab):,}개 (max_df={rec.max_df}, min_df={rec.min_df}) · 방식 {a.method}\n")
    try:
        if a.similar:
            i, res = rec.similar(a.similar, a.top, a.method)
            print(f"'{rec.label(i)}'와 비슷한 영화")
        elif a.watched:
            idx, res = rec.from_watched([t for t in a.watched.split(",") if t.strip()], a.top, a.method)
            print("본 영화: " + ", ".join(rec.label(i) for i in idx))
        else:
            res = rec.search(a.text, a.top, a.method)
            print(f"'{a.text}' 검색 결과")
    except KeyError as e:
        sys.exit(str(e.args[0]))
    print_results(rec, res)


if __name__ == "__main__":
    main()
