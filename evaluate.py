"""추천 방식 비교: TF-IDF vs 임베딩 vs 혼합.

    python evaluate.py                      # data/movies.csv 기준, 결과는 화면과 docs/evaluation_result.md
    python evaluate.py --sample 1000

정답 데이터가 없어서, 어느 방식도 직접 쓰지 않는 정보로 채점하는 대리 지표를 쓴다.
- 시리즈 찾기: 제목이 같은 시리즈(범죄도시 ↔ 범죄도시2)가 상위 10개 안에 있는 비율. 두 방식 모두 제목은 보지 않는다.
- 애니메이션 일치: 애니메이션을 넣었을 때 상위 5개 중 애니메이션 비율. KMDb '유형'은 두 방식 모두 쓰지 않는다.
- 장르 일치: 상위 5개가 기준 영화와 장르를 하나 이상 공유하는 비율. TF-IDF는 장르 토큰을 직접 쓰므로 TF-IDF에 유리하다.
"""
import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import embedder
from recommender import Recommender

ROOT = Path(__file__).parent
METHODS = ["tfidf", "embed", "hybrid"]
NAMES = {"tfidf": "단어 겹침(TF-IDF)", "embed": "줄거리 뜻(임베딩)", "hybrid": "혼합"}

CHECKS = {
    "data/movies.csv": ["범죄도시 (2017)", "극한직업", "기생충", "마션", "겨울왕국 2"],
    "data/movies_kmdb_csv.csv": ["보이스 (2021)", "브로커 (2022)", "이 안에 외계인이 있다", "휴게소 (2016)", "22 (2015)"],
}
# 키워드와 다른 단어로 바꿔 말한 문장. 뜻을 아는지 본다
QUERIES = ["복수를 다룬 스릴러", "깡패를 소탕하는 경찰", "죽은 사람이 유령이 되어 돌아온다",
           "시간을 되돌리는 사랑 이야기", "학교에서 괴롭힘을 당하는 학생", "외계 생명체가 지구를 침공한다"]


def series_key(title: str) -> str:
    """'범죄도시2', '겨울왕국 2', '나쁜 녀석들: 더 무비' → 시리즈 이름. 부제와 끝 숫자를 뗀다."""
    t = re.split(r"\s*[:：]\s*", str(title))[0]
    t = re.sub(r"(극장판|더 무비|시즌\s*\d+|파트\s*\d+|part\s*\d+)", "", t, flags=re.I)
    t = re.sub(r"[\s\d.ⅡⅢⅣ]+$", "", t)
    return re.sub(r"\s+", "", t).lower()


def load(path: Path) -> tuple[pd.DataFrame, Recommender | None]:
    movies = pd.read_csv(path, encoding="utf-8-sig").reset_index(drop=True)
    emb = embedder.load_or_build(movies, path)
    return movies, (Recommender(movies, embeddings=emb) if emb is not None else None)


def genres(s) -> set:
    return set(str(s).split("|")) - {"기타", "nan", ""}


def proxy_metrics(movies: pd.DataFrame, rec: Recommender, n_sample: int, seed=0) -> dict:
    out = {}
    # 시리즈: 같은 시리즈 이름을 가진 영화가 2편 이상인 묶음
    groups = defaultdict(list)
    for i, t in enumerate(movies["title"]):
        k = series_key(t)
        if len(k) >= 2:
            groups[k].append(i)
    series_items = [i for g in groups.values() if len(g) >= 2 for i in g]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(movies), size=min(n_sample, len(movies)), replace=False)
    anim = movies["type"].astype(str).str.contains("애니메이션").to_numpy()
    anim_items = np.flatnonzero(anim)

    for m in METHODS:
        hit = 0
        for i in series_items:
            sib = set(groups[series_key(movies.loc[i, "title"])]) - {i}
            _, res = rec.similar_index(i, 10, m)
            hit += bool(sib & {r.index for r in res})
        g_agree, a_agree = [], []
        for i in sample:
            _, res = rec.similar_index(int(i), 5, m)
            gi = genres(movies.loc[i, "genres"])
            g_agree += [bool(gi & genres(movies.loc[r.index, "genres"])) for r in res]
        for i in anim_items:
            _, res = rec.similar_index(int(i), 5, m)
            a_agree += [anim[r.index] for r in res]
        out[m] = {"series": hit / max(len(series_items), 1), "genre": np.mean(g_agree), "anim": np.mean(a_agree)}
    out["_n"] = {"series": len(series_items), "series_groups": sum(len(g) >= 2 for g in groups.values()),
                 "genre": len(sample), "anim": len(anim_items), "anim_base": anim.mean()}
    return out


def fmt_results(rec: Recommender, res) -> str:
    return " / ".join(f"{rec.label(r.index)}" + (f" ({', '.join(x.split(':')[-1] for x in r.reasons[:2])})" if r.reasons else "")
                      for r in res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500, help="장르 일치에 쓸 무작위 영화 수")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "evaluation_result.md")
    a = ap.parse_args()
    lines = [f"# 추천 방식 비교 결과 (자동 생성: python evaluate.py)\n",
             f"모델: `{embedder.MODEL_NAME}` · 혼합 비중: 임베딩 {int(__import__('recommender').HYBRID_ALPHA * 100)}%\n"]

    for path, checks in CHECKS.items():
        p = ROOT / path
        if not p.exists():
            continue
        movies, rec = load(p)
        lines.append(f"\n## {path} ({len(movies):,}편)\n")
        if rec is None:
            lines.append(f"임베딩이 없어 건너뜀: `python embedder.py --data {path}`\n")
            continue
        lines.append("### 점검표 (상위 5개)\n")
        for q in checks:
            i = rec.find(q)
            lines.append(f"**{rec.label(i)}** [{movies.loc[i, 'genres']}]\n")
            for m in METHODS:
                _, res = rec.similar(q, 5, m)
                lines.append(f"- {NAMES[m]}: {fmt_results(rec, res)}")
            lines.append("")
        lines.append("### 바꿔 말한 문장 검색 (상위 5개)\n")
        for q in QUERIES:
            lines.append(f"**'{q}'**\n")
            for m in METHODS:
                lines.append(f"- {NAMES[m]}: {fmt_results(rec, rec.search(q, 5, m))}")
            lines.append("")
        if "type" in movies and movies["type"].astype(str).str.contains("애니메이션").any():
            t0 = time.time()
            pm = proxy_metrics(movies, rec, a.sample)
            n = pm.pop("_n")
            lines.append("### 대리 지표\n")
            lines.append(f"| 지표 | 대상 | {' | '.join(NAMES[m] for m in METHODS)} |")
            lines.append("|---|---|" + "---|" * len(METHODS))
            lines.append(f"| 시리즈 찾기 (상위 10개에 같은 시리즈) | {n['series']}편 ({n['series_groups']}개 시리즈) | "
                         + " | ".join(f"{pm[m]['series']:.1%}" for m in METHODS) + " |")
            lines.append(f"| 애니메이션 일치 (상위 5개 중) | 애니메이션 {n['anim']}편, 전체 비율 {n['anim_base']:.1%} | "
                         + " | ".join(f"{pm[m]['anim']:.1%}" for m in METHODS) + " |")
            lines.append(f"| 장르 일치 (상위 5개 중) ※TF-IDF에 유리 | 무작위 {n['genre']}편 | "
                         + " | ".join(f"{pm[m]['genre']:.1%}" for m in METHODS) + " |")
            lines.append(f"\n(계산 {time.time() - t0:.0f}초)\n")

    text = "\n".join(lines)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text, encoding="utf-8")
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
