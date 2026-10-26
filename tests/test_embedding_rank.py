"""임베딩·혼합 방식의 순위 계산. 실제 모델 대신 손으로 만든 벡터를 써서 빠르게 확인한다."""
import numpy as np
import pandas as pd
import pytest

import recommender as R
from recommender import Recommender

MOVIES = pd.DataFrame({
    "title": ["범죄A", "범죄B", "우주C", "우주D"],
    "year": [2020, 2021, 2020, 2021],
    "genres": ["범죄", "범죄", "SF", "SF"],
    # 줄거리 단어는 일부러 엇갈리게: A와 C가 단어를 공유하지만 뜻(벡터)은 A와 B가 가깝다
    "overview": ["형사가 우주선 밀수 조직을 쫓는다", "경찰이 깡패를 잡는다", "우주선이 형사를 태우고 행성에 간다", "우주 탐사대가 행성을 조사한다"],
})
EMB = np.array([[1, 0, 0], [0.95, 0.31, 0], [0, 0, 1], [0.1, 0, 0.99]], dtype=np.float32)
EMB /= np.linalg.norm(EMB, axis=1, keepdims=True)


@pytest.fixture()
def rec():
    return Recommender(MOVIES, embeddings=EMB)


def titles(rec, res):
    return [rec.movies.loc[r.index, "title"] for r in res]


def test_embed_ranks_by_meaning(rec):
    _, res = rec.similar("범죄A", 3, method="embed")
    assert titles(rec, res)[0] == "범죄B"


def test_tfidf_unchanged_by_default(rec):
    i, res = rec.similar("범죄A", 3)
    assert i == 0 and "범죄A" not in titles(rec, res)


def test_hybrid_is_weighted_mix(rec):
    _, emb = rec.similar("범죄A", 4, method="embed")
    _, tf = rec.similar("범죄A", 4, method="tfidf")
    _, hy = rec.similar("범죄A", 4, method="hybrid")
    e = {r.index: r.score for r in emb}
    t = {r.index: r.score for r in tf}
    for r in hy:
        assert r.score == pytest.approx(R.HYBRID_ALPHA * e.get(r.index, 0) + (1 - R.HYBRID_ALPHA) * t.get(r.index, 0), abs=1e-5)


def test_watched_profile_excludes_watched(rec):
    idx, res = rec.from_watched(["범죄A", "범죄B"], 2, method="embed")
    assert not set(idx) & {r.index for r in res}


def test_embed_requires_embeddings():
    with pytest.raises(ValueError):
        Recommender(MOVIES).similar("범죄A", method="embed")
    with pytest.raises(ValueError):
        Recommender(MOVIES, embeddings=EMB[:2])
