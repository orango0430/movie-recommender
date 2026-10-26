"""로컬 검색 화면. 내장 http.server로 static/index.html과 추천 API를 띄운다.

    python web.py            # http://localhost:8501
    python web.py --port 9000
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

import embedder
from recommender import METHODS, Recommender

ROOT = Path(__file__).parent
DATASETS = {
    "api": ("KMDb API (2015~2025)", ROOT / "data" / "movies.csv"),
    "csv": ("독립영화 (공공데이터포털)", ROOT / "data" / "movies_kmdb_csv.csv"),
    "sample": ("샘플 35편", ROOT / "data" / "sample_movies.csv"),
}

_cache: dict[str, Recommender] = {}
_lock = threading.Lock()


def read(dataset: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS[dataset][1], encoding="utf-8-sig").reset_index(drop=True)


def has_embeddings(dataset: str) -> bool:
    """임베딩은 python embedder.py 로 미리 만들어 둔 것만 쓴다. 화면에서 수십 분짜리 계산을 시작하지 않는다."""
    return embedder.cache_path(read(dataset), DATASETS[dataset][1]).exists()


def available() -> list[dict]:
    return [{"id": k, "name": name, "embed": has_embeddings(k)}
            for k, (name, path) in DATASETS.items() if path.exists()]


def recommender(dataset: str) -> Recommender:
    with _lock:  # 처음 한 번 형태소 분석에 몇 초 걸린다
        if dataset not in _cache:
            movies = read(dataset)
            emb = embedder.load_or_build(movies, DATASETS[dataset][1])
            _cache[dataset] = Recommender(movies, embeddings=emb)
        return _cache[dataset]


def value(v):
    return None if pd.isna(v) else v


def movie_json(rec: Recommender, i: int, score=None, reasons=None) -> dict:
    r = rec.movies.loc[i]
    overview = str(value(r.get("overview")) or "")
    return {
        "label": rec.label(i),
        "title": r["title"],
        "year": None if pd.isna(r.get("year")) else int(r["year"]),
        "genres": [g for g in str(value(r.get("genres")) or "").split("|") if g],
        "director": value(r.get("director")),
        "overview": overview[:220] + ("…" if len(overview) > 220 else ""),
        "url": value(r.get("url")),
        "score": None if score is None else round(score, 3),
        # 내부 토큰 접두사는 화면에서 읽기 쉽게 바꾼다
        "reasons": [x.replace("장르:", "장르 ").replace("키워드:", "#").replace("감독:", "감독 ") for x in (reasons or [])],
    }


def search(params: dict) -> dict:
    dataset = params.get("data", "csv")
    mode = params.get("mode", "text")
    method = params.get("method", "tfidf")
    q = params.get("q", "").strip()
    top = min(int(params.get("top", 10)), 30)
    if dataset not in DATASETS or not DATASETS[dataset][1].exists():
        return {"error": "데이터 파일이 없습니다"}
    if method not in METHODS:
        return {"error": "알 수 없는 방식입니다"}
    if not q:
        return {"error": "검색어를 입력하세요"}
    rec = recommender(dataset)
    try:
        if mode == "similar":
            i, res = rec.similar(q, top, method)
            base = movie_json(rec, i)
        else:
            res, base = rec.search(q, top, method), None
    except KeyError as e:
        return {"error": e.args[0]}
    except ValueError as e:
        return {"error": str(e)}
    return {"base": base, "method": method, "results": [movie_json(rec, r.index, r.score, r.reasons) for r in res],
            "count": len(rec.movies)}


def titles(params: dict) -> list[str]:
    dataset = params.get("data", "csv")
    q = params.get("q", "").replace(" ", "")
    if not q or dataset not in DATASETS:
        return []
    rec = recommender(dataset)
    hit = rec.movies[rec.movies["title"].str.replace(" ", "").str.contains(q, regex=False, na=False)]
    return [rec.label(i) for i in hit.index[:15]]


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self):
        url = urlparse(self.path)
        params = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            self._send((ROOT / "static" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/datasets":
            self._json(available())
        elif url.path == "/api/search":
            self._json(search(params))
        elif url.path == "/api/titles":
            self._json(titles(params))
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # 요청마다 찍히는 로그는 끈다
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args()
    # 기본 데이터와 임베딩 모델은 미리 읽어 두어 첫 검색이 느리지 않게 한다
    ds = available()
    if ds:
        print(f"{DATASETS[ds[0]['id']][0]} 불러오는 중…")
        recommender(ds[0]["id"])
    if any(d["embed"] for d in ds):
        print("임베딩 모델 불러오는 중… (문장 검색용)")
        embedder.model()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"http://localhost:{a.port}  (종료: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
