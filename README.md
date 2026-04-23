# 🇺🇸 미국 대통령 취임사 검색 시스템

NLTK Inaugural Address Corpus (60개 문서)를 기반으로 한 정보 검색 시스템입니다.  
**Cosine Similarity**와 **BM25** 랭킹 알고리즘을 지원하며, 질의어별 IDF 및 문서별 TF 정보를 제공합니다.

---

## 실행 방법

```bash
# 의존성 설치
pip install nltk flask

# NLTK 데이터 다운로드
python3 -c "import nltk; nltk.download('inaugural'); nltk.download('punkt_tab')"

# 서버 실행
python3 app.py
# → http://127.0.0.1:5000 접속
```

---

## 시스템 인터페이스

| 요구사항                                         | 구현 |
| ------------------------------------------------ | ---- |
| 검색창 + 결과 (대통령 이름, 핵심 문장 1줄, URL)  | ✅    |
| Cosine Similarity / BM25 선택 (k1=1.2, b=0.9)    | ✅    |
| 질의어별 IDF 표시 + 문서별 TF 표시 (결과와 별도) | ✅    |

---

## 프로그램 구조

### 파일 구성

| 파일                   | 역할                     |
| ---------------------- | ------------------------ |
| `app.py`               | 검색 엔진 백엔드 (Flask) |
| `templates/index.html` | 웹 UI                    |

### 모듈별 기능 (코드 내 주석 포함)

| #   | 기능               | 설명                                                                              |
| --- | ------------------ | --------------------------------------------------------------------------------- |
| 1   | **문서 ID 부여**   | 파일명에서 연도·대통령 이름 추출, 동명이인(Adams, Harrison, Roosevelt, Bush) 구분 |
| 2   | **토큰화**         | 정규식으로 알파벳 토큰 추출 (`re.findall(r"[a-z]+")`)                             |
| 3   | **스테밍**         | NLTK PorterStemmer 사용                                                           |
| 4   | **사전 구성**      | 전체 문서의 stemmed 토큰에서 정렬된 vocabulary 생성 (5,569개)                     |
| 5   | **IDF / 통계**     | `IDF = log₁₀(N / df)`, 평균 문서 길이 ≈ 2,354.2                                   |
| 6   | **역파일**         | `term → [(doc_id, tf), ...]` 구조의 Inverted Index                                |
| 7   | **랭킹**           | Cosine Similarity (tf-idf 가중치) 및 BM25 (k1=1.2, b=0.9)                         |
| 8   | **핵심 문장 추출** | 질의어 매칭 점수 기반 문장 선택                                                   |
| 9   | **결과 제공**      | Flask REST API + 반응형 웹 UI                                                     |

---

## 사용 말뭉치

- **출처**: NLTK 제공 `Inaugural Address` 코퍼스
- **규모**: 문서 60개 (George Washington ~ Donald Trump 2nd Term)
- **형식**: UTF-8 인코딩 텍스트 파일

---

## 랭킹 알고리즘

### Cosine Similarity

질의와 문서를 tf-idf 가중치 벡터로 표현한 뒤, 코사인 유사도를 계산합니다.

```
w(t, d) = (1 + log₁₀(tf)) × idf(t)
sim(q, d) = (q · d) / (|q| × |d|)
```

### BM25

Okapi BM25 랭킹 함수를 사용합니다.

```
score(D, Q) = Σ idf(q) × tf(q,D) × (k1 + 1) / (tf(q,D) + k1 × (1 - b + b × |D| / avgdl))

k1 = 1.2,  b = 0.9
```

---

## 테스트 결과

질의어: **"freedom democracy"**

### IDF 정보

| Term      | Stemmed   | DF  | IDF    |
| --------- | --------- | --- | ------ |
| freedom   | freedom   | 38  | 0.1984 |
| democracy | democraci | 19  | 0.4994 |

### Cosine Similarity 상위 5

| 순위 | 대통령                | 연도 | Score  |
| ---- | --------------------- | ---- | ------ |
| 1    | Franklin D. Roosevelt | 1945 | 0.0724 |
| 2    | Franklin D. Roosevelt | 1941 | 0.0657 |
| 3    | Harry S. Truman       | 1949 | 0.0569 |
| 4    | Bill Clinton          | 1993 | 0.0478 |
| 5    | Franklin D. Roosevelt | 1937 | 0.0449 |

### BM25 상위 5

| 순위 | 대통령                | 연도 | Score  |
| ---- | --------------------- | ---- | ------ |
| 1    | Franklin D. Roosevelt | 1941 | 1.4041 |
| 2    | Harry S. Truman       | 1949 | 1.3730 |
| 3    | George H. W. Bush     | 1989 | 1.2514 |
| 4    | Bill Clinton          | 1993 | 1.2438 |
| 5    | Franklin D. Roosevelt | 1945 | 1.2420 |

---

## 기술 스택

- **Backend**: Python 3, Flask
- **NLP**: NLTK (PorterStemmer, Inaugural Corpus)
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **검색 모델**: Vector Space Model (Cosine), BM25
