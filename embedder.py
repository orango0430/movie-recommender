"""줄거리 문장 임베딩. 학습 없이 공개 모델로 줄거리를 '뜻 벡터'로 바꾼다.

    python embedder.py --data data/movies.csv     # 처음 한 번: 벡터를 계산해 data/emb/에 저장 (7,456편 약 15분, CPU)

모델은 128토큰까지만 읽는데 KMDb 줄거리는 절반 이상이 더 길다. 그래서 줄거리를 문장 단위로 묶어 128토큰 이하 조각으로 나누고,
조각 벡터를 길이 가중 평균한다. 줄거리에 없는 장르·키워드·감독은 넣지 않는다. 그 정보는 TF-IDF 쪽이 맡는다(혼합 방식).
"""
import argparse
import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recommender import clean_plot

MODEL_NAME = "jhgan/ko-sroberta-multitask"
EMB_DIR = Path(__file__).parent / "data" / "emb"
CHUNK_TOKENS = 120  # 모델 한도 128에서 특수 토큰 자리를 뺀 값
_SENT_END = re.compile(r"(?<=[.!?…。])\s+|(?<=[다요죠])\.\s*")

_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # torch를 쓸 때만 불러온다
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def chunks(text: str) -> list[str]:
    """문장을 이어 붙이다가 CHUNK_TOKENS를 넘기기 직전에 자른다. 한 문장이 너무 길면 그 문장은 모델이 자른다."""
    text = clean_plot(text).strip()
    if not text:
        return [""]
    tok = model().tokenizer
    out, cur, cur_len = [], [], 0
    for sent in [s for s in _SENT_END.split(text) if s and s.strip()]:
        n = len(tok.tokenize(sent))
        if cur and cur_len + n > CHUNK_TOKENS:
            out.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent.strip())
        cur_len += n
    if cur:
        out.append(" ".join(cur))
    return out


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """여러 문서를 벡터로. 문서마다 조각을 나눠 길이 가중 평균 후 길이 1로 맞춘다."""
    pieces, owner, weight = [], [], []
    for i, t in enumerate(texts):
        for c in chunks(t):
            pieces.append(c)
            owner.append(i)
            weight.append(max(len(c), 1))
    vecs = model().encode(pieces, batch_size=batch_size, normalize_embeddings=True,
                          show_progress_bar=show_progress, convert_to_numpy=True)
    out = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)[:, None]
    np.add.at(out, np.asarray(owner), vecs * w)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def cache_path(movies: pd.DataFrame, source: Path) -> Path:
    """데이터 내용(id·줄거리)이 바뀌면 다른 파일이 되도록 해시를 붙인다."""
    # pandas 3은 astype(str)이 빈 값(NaN)을 그대로 두므로 먼저 빈 문자열로 바꾼다
    text = movies["title"].fillna("").astype(str) + movies["overview"].fillna("").astype(str)
    key = hashlib.sha1((MODEL_NAME + "".join(text)).encode("utf-8")).hexdigest()[:10]
    return EMB_DIR / f"{Path(source).stem}.{key}.npy"


def load_or_build(movies: pd.DataFrame, source: Path, build: bool = False) -> np.ndarray | None:
    """저장된 벡터가 있으면 읽고, 없으면 build=True일 때만 계산한다 (오래 걸리므로 화면에서 몰래 계산하지 않는다)."""
    path = cache_path(movies, source)
    if path.exists():
        return np.load(path)
    if not build:
        return None
    t0 = time.time()
    emb = embed_texts(movies["overview"].fillna("").tolist(), show_progress=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    print(f"{len(movies):,}편 임베딩 {time.time() - t0:.0f}초 -> {path}")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "movies.csv")
    a = ap.parse_args()
    movies = pd.read_csv(a.data, encoding="utf-8-sig").reset_index(drop=True)
    emb = load_or_build(movies, a.data, build=True)
    print(f"벡터 {emb.shape} ({MODEL_NAME})")


if __name__ == "__main__":
    main()
