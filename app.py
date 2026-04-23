"""
미국 대통령 취임사 검색 시스템
- NLTK Inaugural Address Corpus 사용
- Cosine Similarity 및 BM25 랭킹 지원
- TF/IDF 상세 정보 제공
"""

import os
import re
import math
import glob
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify

# ============================================================
# 1. 문서 id를 부여, 문서 id에 대한 취임사 대통령 이름 저장
# ============================================================

CORPUS_DIR = os.path.expanduser("~/nltk_data/corpora/inaugural")

# 대통령 성(last name)에서 전체 이름으로 매핑
PRESIDENT_FULL_NAMES = {
    "Washington": "George Washington", "Adams": "John Adams",
    "Jefferson": "Thomas Jefferson", "Madison": "James Madison",
    "Monroe": "James Monroe", "Jackson": "Andrew Jackson",
    "VanBuren": "Martin Van Buren", "Harrison": "William Henry Harrison",
    "Polk": "James K. Polk", "Taylor": "Zachary Taylor",
    "Pierce": "Franklin Pierce", "Buchanan": "James Buchanan",
    "Lincoln": "Abraham Lincoln", "Grant": "Ulysses S. Grant",
    "Hayes": "Rutherford B. Hayes", "Garfield": "James A. Garfield",
    "Cleveland": "Grover Cleveland", "Harrison-1889": "Benjamin Harrison",
    "McKinley": "William McKinley", "Roosevelt": "Theodore Roosevelt",
    "Taft": "William Howard Taft", "Wilson": "Woodrow Wilson",
    "Harding": "Warren G. Harding", "Coolidge": "Calvin Coolidge",
    "Hoover": "Herbert Hoover", "Roosevelt-1933": "Franklin D. Roosevelt",
    "Truman": "Harry S. Truman", "Eisenhower": "Dwight D. Eisenhower",
    "Kennedy": "John F. Kennedy", "Johnson": "Lyndon B. Johnson",
    "Nixon": "Richard Nixon", "Carter": "Jimmy Carter",
    "Reagan": "Ronald Reagan", "Bush": "George H. W. Bush",
    "Clinton": "Bill Clinton", "Bush-2001": "George W. Bush",
    "Obama": "Barack Obama", "Trump": "Donald Trump",
    "Biden": "Joe Biden",
}

def get_president_name(filename):
    """파일명에서 대통령 이름과 연도를 추출"""
    base = os.path.basename(filename).replace(".txt", "")
    year, last = base.split("-", 1)
    year = int(year)
    # 동명이인 구분
    key = last
    if last == "Harrison" and year >= 1889:
        key = "Harrison-1889"
    elif last == "Roosevelt" and year >= 1933:
        key = "Roosevelt-1933"
    elif last == "Bush" and year >= 2001:
        key = "Bush-2001"
    elif last == "Adams" and year >= 1825:
        key = "Adams"  # John Quincy Adams
        return year, "John Quincy Adams"
    full = PRESIDENT_FULL_NAMES.get(key, last)
    return year, full

def load_documents():
    """문서를 로드하고 각 문서에 고유 id를 부여"""
    documents = {}  # doc_id -> { year, president, text, filename }
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    for doc_id, fpath in enumerate(files):
        year, president = get_president_name(fpath)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        documents[doc_id] = {
            "year": year,
            "president": president,
            "text": text,
            "filename": os.path.basename(fpath),
        }
    return documents


# ============================================================
# 2. 각 문서를 token으로 바꿈
# ============================================================

def tokenize(text):
    """텍스트를 소문자로 변환 후 알파벳 토큰으로 분리"""
    return re.findall(r"[a-z]+", text.lower())


# ============================================================
# 3. stemming을 함 (Porter Stemmer 직접 구현 - 간소화 버전)
# ============================================================

# 간단한 suffix-stripping stemmer
def stem(word):
    """간단한 suffix-stripping stemmer"""
    if len(word) <= 3:
        return word
    # 기본 접미사 제거 규칙들
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("alli", "al"),
        ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
        ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
        ("biliti", "ble"),
        ("ness", ""), ("ment", ""), ("ing", ""), ("ings", ""),
        ("tion", "t"), ("sion", "s"),
        ("ies", "i"), ("ied", "i"),
        ("able", ""), ("ible", ""),
        ("ly", ""), ("ed", ""), ("er", ""),
        ("es", ""), ("s", ""),
    ]
    for suffix, replacement in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)] + replacement
    return word

# NLTK PorterStemmer 사용 (가능한 경우)
try:
    from nltk.stem import PorterStemmer
    _stemmer = PorterStemmer()
    def stem(word):
        return _stemmer.stem(word)
except ImportError:
    pass  # 위의 간소화 버전 사용


# ============================================================
# 4. 사전을 구성함 (Dictionary / Vocabulary)
# ============================================================

def build_vocabulary(doc_tokens):
    """전체 문서의 토큰에서 어휘 사전(vocabulary)을 구성"""
    vocab = set()
    for tokens in doc_tokens.values():
        vocab.update(tokens)
    return sorted(vocab)


# ============================================================
# 5. idf, 문서 평균 길이 등을 구함
# ============================================================

def compute_df(doc_tokens):
    """각 term의 document frequency를 계산"""
    df = defaultdict(int)
    for tokens in doc_tokens.values():
        seen = set(tokens)
        for t in seen:
            df[t] += 1
    return df

def compute_idf(df, N):
    """IDF = log10(N / df_t)"""
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log10(N / freq)
    return idf

def compute_avg_dl(doc_tokens):
    """문서 평균 길이 계산"""
    total = sum(len(tokens) for tokens in doc_tokens.values())
    return total / len(doc_tokens)


# ============================================================
# 6. 역파일을 만듦 (Inverted Index)
# ============================================================

def build_inverted_index(doc_tokens):
    """역파일(inverted index) 생성: term -> [(doc_id, tf), ...]"""
    index = defaultdict(list)
    for doc_id, tokens in doc_tokens.items():
        tf_counter = Counter(tokens)
        for term, tf in tf_counter.items():
            index[term].append((doc_id, tf))
    return index


# ============================================================
# 7. cosine measure와 BM25로 순서화함
# ============================================================

def search_cosine(query_terms, inverted_index, idf, doc_tokens, N):
    """
    Cosine Similarity를 이용한 검색
    - 질의 벡터: tf-idf (질의 내 각 term의 tf * idf)
    - 문서 벡터: tf-idf (문서 내 각 term의 tf * idf)
    - 유사도: dot product / (|q| * |d|)
    """
    # 질의 tf 계산
    query_tf = Counter(query_terms)
    
    # 질의 벡터 (tf-idf)
    query_vec = {}
    for t in query_tf:
        if t in idf:
            query_vec[t] = (1 + math.log10(query_tf[t])) * idf[t]
    
    if not query_vec:
        return []
    
    # 질의 벡터 크기
    q_norm = math.sqrt(sum(v * v for v in query_vec.values()))
    if q_norm == 0:
        return []
    
    # 문서별 점수 계산
    scores = defaultdict(float)
    doc_norms = {}
    
    # 문서 벡터 크기 미리 계산
    for doc_id, tokens in doc_tokens.items():
        tf_counter = Counter(tokens)
        norm_sq = 0
        for t, tf in tf_counter.items():
            if t in idf:
                w = (1 + math.log10(tf)) * idf[t]
                norm_sq += w * w
        doc_norms[doc_id] = math.sqrt(norm_sq) if norm_sq > 0 else 0
    
    # dot product 계산
    for t, q_w in query_vec.items():
        if t in inverted_index:
            for doc_id, tf in inverted_index[t]:
                d_w = (1 + math.log10(tf)) * idf[t]
                scores[doc_id] += q_w * d_w
    
    # cosine similarity 정규화
    results = []
    for doc_id, dot_prod in scores.items():
        d_norm = doc_norms.get(doc_id, 0)
        if d_norm > 0 and q_norm > 0:
            sim = dot_prod / (q_norm * d_norm)
            results.append((doc_id, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def search_bm25(query_terms, inverted_index, idf, doc_tokens, N, avg_dl, k1=1.2, b=0.9):
    """
    BM25 랭킹 함수
    - k1 = 1.2, b = 0.9
    - score(D, Q) = sum over q in Q: idf(q) * (tf(q,D) * (k1+1)) / (tf(q,D) + k1 * (1 - b + b * |D|/avgdl))
    """
    scores = defaultdict(float)
    
    for t in set(query_terms):
        if t not in inverted_index or t not in idf:
            continue
        idf_val = idf[t]
        for doc_id, tf in inverted_index[t]:
            dl = len(doc_tokens[doc_id])
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (dl / avg_dl))
            scores[doc_id] += idf_val * (numerator / denominator)
    
    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return results


# ============================================================
# 8. 찾아진 문서에서 중요 문장 추출
# ============================================================

def extract_key_sentence(text, query_terms):
    """
    질의어와 가장 관련 깊은 문장을 추출
    방법: 각 문장에 포함된 질의어(stemmed) 수를 기준으로 점수 매김
    """
    # 문장 분리
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if not sentences:
        return text[:200] + "..."
    
    best_score = -1
    best_sentence = sentences[0]
    
    for sent in sentences:
        tokens = [stem(t) for t in tokenize(sent)]
        # 질의어 매칭 점수
        score = sum(1 for t in query_terms if t in tokens)
        # 보너스: 여러 번 등장하는 질의어
        score += sum(tokens.count(t) for t in query_terms) * 0.1
        if score > best_score:
            best_score = score
            best_sentence = sent
    
    # 문장이 너무 길면 자르기
    if len(best_sentence) > 300:
        best_sentence = best_sentence[:300] + "..."
    
    return best_sentence


# ============================================================
# 시스템 초기화: 문서 로드 및 인덱스 구축
# ============================================================

print("문서 로딩 중...")
documents = load_documents()
N = len(documents)
print(f"총 {N}개 문서 로드 완료")

# 2. 토큰화
print("토큰화 중...")
doc_raw_tokens = {}
for doc_id, doc in documents.items():
    doc_raw_tokens[doc_id] = tokenize(doc["text"])

# 3. 스테밍
print("스테밍 중...")
doc_stemmed_tokens = {}
for doc_id, tokens in doc_raw_tokens.items():
    doc_stemmed_tokens[doc_id] = [stem(t) for t in tokens]

# 4. 사전 구성
print("사전 구성 중...")
vocabulary = build_vocabulary(doc_stemmed_tokens)
print(f"사전 크기: {len(vocabulary)}")

# 5. IDF 및 통계 계산
print("IDF 및 통계 계산 중...")
df = compute_df(doc_stemmed_tokens)
idf = compute_idf(df, N)
avg_dl = compute_avg_dl(doc_stemmed_tokens)
print(f"평균 문서 길이: {avg_dl:.1f}")

# 6. 역파일 구축
print("역파일 구축 중...")
inverted_index = build_inverted_index(doc_stemmed_tokens)
print(f"역파일 항목 수: {len(inverted_index)}")

# 문서별 TF 미리 계산 (빠른 조회용)
doc_tf = {}
for doc_id, tokens in doc_stemmed_tokens.items():
    doc_tf[doc_id] = Counter(tokens)

print("시스템 준비 완료!\n")


# ============================================================
# 9. 결과 제공 (Flask 웹 서버)
# ============================================================

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query_text = data.get("query", "").strip()
    method = data.get("method", "cosine")
    
    if not query_text:
        return jsonify({"error": "검색어를 입력하세요.", "results": [], "idf_info": [], "tf_info": []})
    
    # 질의어 처리: 토큰화 + 스테밍
    raw_terms = tokenize(query_text)
    stemmed_terms = [stem(t) for t in raw_terms]
    
    if not stemmed_terms:
        return jsonify({"error": "유효한 검색어가 없습니다.", "results": [], "idf_info": [], "tf_info": []})
    
    # 질의어 IDF 정보
    idf_info = []
    for raw, stemmed in zip(raw_terms, stemmed_terms):
        idf_val = idf.get(stemmed, 0)
        df_val = df.get(stemmed, 0)
        idf_info.append({
            "raw_term": raw,
            "stemmed_term": stemmed,
            "df": df_val,
            "idf": round(idf_val, 4),
        })
    
    # 7. 검색 수행
    if method == "bm25":
        results = search_bm25(stemmed_terms, inverted_index, idf, doc_stemmed_tokens, N, avg_dl, k1=1.2, b=0.9)
    else:
        results = search_cosine(stemmed_terms, inverted_index, idf, doc_stemmed_tokens, N)
    
    # 결과 정리
    search_results = []
    tf_info = []
    
    for doc_id, score in results[:20]:  # 상위 20개
        doc = documents[doc_id]
        # 8. 중요 문장 추출
        key_sentence = extract_key_sentence(doc["text"], stemmed_terms)
        
        url = f"https://avalon.law.yale.edu/subject_menus/inaug.asp"
        
        search_results.append({
            "doc_id": doc_id,
            "president": doc["president"],
            "year": doc["year"],
            "score": round(score, 6),
            "key_sentence": key_sentence,
            "url": url,
            "filename": doc["filename"],
            "doc_length": len(doc_stemmed_tokens[doc_id]),
        })
        
        # TF 정보
        doc_tf_info = {"doc_id": doc_id, "president": doc["president"], "year": doc["year"], "terms": []}
        for raw, stemmed in zip(raw_terms, stemmed_terms):
            tf_val = doc_tf[doc_id].get(stemmed, 0)
            doc_tf_info["terms"].append({
                "raw_term": raw,
                "stemmed_term": stemmed,
                "tf": tf_val,
            })
        tf_info.append(doc_tf_info)
    
    return jsonify({
        "results": search_results,
        "idf_info": idf_info,
        "tf_info": tf_info,
        "method": method,
        "query": query_text,
        "total_docs": N,
        "avg_dl": round(avg_dl, 1),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
