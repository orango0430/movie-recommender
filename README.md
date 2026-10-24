# KMDb 영화 추천 (로컬 예제)

한국영상자료원 KMDb API로 영화 줄거리를 모아 TF-IDF + 코사인 유사도로 추천한다. 서버·DB 없이 터미널에서 실행한다.

## 실행

```powershell
py -3.13 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env                       # KMDB_API_KEY 채우기 (kmdb.or.kr에서 발급)
python collect_kmdb.py --sample              # 응답 구조 확인 -> data/raw/sample.json
python collect_kmdb.py --from 20150101 --to 20251231
python recommender.py --similar 기생충
python recommender.py --watched "기생충,올드보이"
python recommender.py --text "복수를 다룬 스릴러"
```

샘플 35편 버전: `python recommender.py --data data/sample_movies.csv --similar 기생충`

## 추천 방식

문서 = 줄거리 명사·동사(kiwipiepy) + `장르:` 토큰 x2 + `키워드:` 토큰 x2 + `감독:` 토큰 x1.
영화가 200편 이상이면 `max_df=0.5`(절반 넘게 나오는 단어 제외), `min_df=2`(한 영화에만 나오는 단어 제외).
추천 이유 = 두 영화 TF-IDF 벡터에서 곱이 큰 단어.

## 정제 규칙

HTML 태그·검색 강조 문자열(!HS, !HE) 제거 → 한국어 줄거리만 → 50자 미만 제외 → docid 중복·제목+연도 중복(재개봉) 제거 → 장르 구분자 `|`, 빈 장르는 기타 → 에로 장르 제외 → 1,000편 초과 시 관객 수 순.

## 결과 점검

(수집 후 작성 예정: 범죄도시, 극한직업, 기생충, 인터스텔라, 겨울왕국 점검표와 35편 버전 비교)

## 출처

영화 데이터: 한국영상자료원 KMDb (https://www.kmdb.or.kr). `data/sample_movies.csv`의 줄거리는 테스트용으로 직접 요약한 것이다.