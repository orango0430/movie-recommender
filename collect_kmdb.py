"""KMDb(한국영상자료원) API로 영화를 수집·정제해 data/movies.csv로 저장한다.

    python collect_kmdb.py --sample                       # 1건만 받아 data/raw/sample.json (응답 구조 확인용)
    python collect_kmdb.py --from 20150101 --to 20251231  # 본 수집 (극영화, 애니메이션 각각)
    python collect_kmdb.py --from-raw                     # API를 다시 부르지 않고 data/raw/로 다시 정제

API 키는 .env의 KMDB_API_KEY에서 읽는다. KMDb 사이트(kmdb.or.kr)에서 발급한 키여야 하며,
공공데이터포털 개인 키로는 '인증키가 필요합니다'가 돌아온다.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "movies.csv"

URL = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json2.jsp"
TYPES = ["극영화", "애니메이션"]
PAGE = 500
PAUSE = 0.5
MIN_PLOT = 50
MAX_MOVIES = 1000
EXCLUDED_GENRES = {"에로"}


# ---------------- 수집 ----------------

def api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("KMDB_API_KEY", "").strip()
    if not key:
        sys.exit(".env에 KMDB_API_KEY=발급받은키 를 넣어 주세요")
    return key


def call(params: dict) -> dict:
    params = {"collection": "kmdb_new2", "detail": "Y", "ServiceKey": api_key(), **params}
    r = requests.get(URL, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 오류 응답은 { "Error" : 인증키가 필요합니다. } 처럼 JSON이 아니다
        sys.exit(f"KMDb 응답을 읽을 수 없습니다: {text[:200]}")


def total_count(js: dict) -> int:
    data = js.get("Data") or [{}]
    return int(data[0].get("TotalCount") or js.get("TotalCount") or 0)


def fetch_sample():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    js = call({"type": "극영화", "releaseDts": "20190101", "releaseDte": "20191231",
               "title": "기생충", "listCount": 1, "startCount": 0})
    path = RAW_DIR / "sample.json"
    path.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = results(js)
    print(f"저장: {path} (TotalCount={total_count(js)}, 받은 건수={len(rows)})")
    if rows:
        print("줄거리·감독·키워드가 어디에 오는지 확인하세요:")
        for k in ("DOCID", "title", "prodYear", "genre", "plots", "directors", "keywords", "nation", "audiAcc", "kmdbUrl"):
            print(f"  {k}: {json.dumps(rows[0].get(k), ensure_ascii=False)[:150]}")


def fetch_all(start: str, end: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for t in TYPES:
        offset, total = 0, None
        while total is None or offset < total:
            path = RAW_DIR / f"{t}_{start}_{end}_{offset:05d}.json"
            if path.exists():  # 같은 요청은 다시 보내지 않는다
                js = json.loads(path.read_text(encoding="utf-8"))
            else:
                js = call({"type": t, "releaseDts": start, "releaseDte": end,
                           "listCount": PAGE, "startCount": offset})
                path.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
                time.sleep(PAUSE)
            total = total_count(js)
            print(f"{t} {offset}~{offset + PAGE} / {total}")
            if not results(js):
                break
            offset += PAGE


# ---------------- 응답 해석 ----------------

def results(js: dict) -> list[dict]:
    out = []
    for d in js.get("Data") or []:
        out += d.get("Result") or []
    return out


def as_list(v, key: str) -> list:
    """KMDb는 같은 필드가 {'plot': [..]}, {'plot': {..}}, [..] 처럼 다양하게 온다"""
    if v is None or v == "":
        return []
    if isinstance(v, dict):
        v = v.get(key, v)
    if isinstance(v, dict):
        return [v]
    return v if isinstance(v, list) else [v]


_HS = re.compile(r"!H[SE]")
_TAG = re.compile(r"<[^>]+>")


def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = _HS.sub("", _TAG.sub("", s))
    return re.sub(r"\s+", " ", s).strip()


def korean_plot(r: dict) -> str:
    """여러 언어 줄거리 중 한국어만. 언어 표시가 없으면 그 줄거리를 쓴다."""
    for p in as_list(r.get("plots"), "plot"):
        if isinstance(p, str):
            return clean_text(p)
        lang = str(p.get("plotLang", ""))
        if not lang or "한국" in lang:
            return clean_text(p.get("plotText", ""))
    return ""


def directors(r: dict) -> str:
    names = [clean_text(d.get("directorNm")) for d in as_list(r.get("directors"), "director") if isinstance(d, dict)]
    return "|".join(n for n in names if n)


def to_int(v):
    try:
        return int(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def to_row(r: dict) -> dict:
    genres = "|".join(g.strip() for g in re.split(r"[,|]", clean_text(r.get("genre"))) if g.strip())
    keywords = "|".join(k.strip() for k in re.split(r"[,|]", clean_text(r.get("keywords"))) if k.strip())
    return {
        "id": clean_text(r.get("DOCID")) or f"{r.get('movieId', '')}{r.get('movieSeq', '')}",
        "title": clean_text(r.get("title")),
        "year": to_int(r.get("prodYear")),
        "genres": genres or "기타",
        "overview": korean_plot(r),
        "keywords": keywords,
        "director": directors(r),
        "nation": clean_text(r.get("nation")),
        "audience": to_int(r.get("audiAcc")),
        "url": clean_text(r.get("kmdbUrl")),
        "type": clean_text(r.get("type")),
    }


# ---------------- 정제 ----------------

def refine(rows: list[dict], max_movies: int = MAX_MOVIES) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    df = pd.DataFrame(rows)
    steps = [("수집", len(df))]

    df = df[df["overview"].str.len() >= MIN_PLOT]
    steps.append((f"줄거리 {MIN_PLOT}자 이상", len(df)))

    df = df.drop_duplicates("id")
    # 재개봉판: 제목과 연도가 같으면 관객 수가 많은 쪽, 줄거리가 긴 쪽을 남긴다
    df = (df.assign(_len=df["overview"].str.len(), _aud=df["audience"].fillna(-1))
            .sort_values(["_aud", "_len"], ascending=False)
            .drop_duplicates(["title", "year"])
            .drop(columns=["_len", "_aud"]))
    steps.append(("중복 제거", len(df)))

    df = df[~df["genres"].map(lambda g: any(x in g.split("|") for x in EXCLUDED_GENRES))]
    steps.append(("제외 장르 빼기", len(df)))

    # 관객 수가 대부분 있으면 많은 순으로 자르고, 대부분 비어 있으면 전부 쓴다
    if len(df) > max_movies and df["audience"].notna().mean() >= 0.5:
        df = df.sort_values("audience", ascending=False, na_position="last").head(max_movies)
    steps.append(("최종", len(df)))
    return df.sort_values(["year", "title"]).reset_index(drop=True), steps


def load_raw() -> list[dict]:
    rows = []
    for p in sorted(RAW_DIR.glob("*_*_*_*.json")):
        rows += [to_row(r) for r in results(json.loads(p.read_text(encoding="utf-8")))]
    if not rows:
        sys.exit(f"{RAW_DIR}에 원본이 없습니다. 먼저 수집하세요")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="1건만 받아 응답 구조 확인")
    ap.add_argument("--from", dest="start", default="20150101")
    ap.add_argument("--to", dest="end", default="20251231")
    ap.add_argument("--from-raw", action="store_true", help="저장된 원본으로 다시 정제")
    ap.add_argument("--max", type=int, default=MAX_MOVIES)
    a = ap.parse_args()

    if a.sample:
        fetch_sample()
        return
    if not a.from_raw:
        fetch_all(a.start, a.end)
    df, steps = refine(load_raw(), a.max)
    print(" → ".join(f"{name} {n:,}건" for name, n in steps))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
