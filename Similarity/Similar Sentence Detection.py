from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 두 문서의 유사한 문장을 검출하는 함수
def find_similar_sentences(doc1, doc2, threshold=0.8):
    """
    두 문서에서 유사한 문장을 검출하는 함수

    Parameters:
    doc1 (str): 첫 번째 문서
    doc2 (str): 두 번째 문서
    threshold (float): 유사도를 결정하는 임계값 (0.0 ~ 1.0)

    Returns:
    list of tuple: 유사한 문장의 쌍과 유사도 점수 [(문장1, 문장2, 유사도), ...]
    """
    # 문서를 문장 단위로 분리
    sentences1 = doc1.split(". ")
    sentences2 = doc2.split(". ")

    # 모든 문장을 TF-IDF로 변환
    all_sentences = sentences1 + sentences2
    vectorizer = TfidfVectorizer().fit(all_sentences)
    tfidf_matrix1 = vectorizer.transform(sentences1)
    tfidf_matrix2 = vectorizer.transform(sentences2)

    # 문장 쌍 간의 코사인 유사도 계산
    similarities = cosine_similarity(tfidf_matrix1, tfidf_matrix2)

    # 임계값을 초과하는 유사한 문장 쌍 찾기
    similar_pairs = []
    for i, row in enumerate(similarities):
        for j, score in enumerate(row):
            if score >= threshold:
                similar_pairs.append((sentences1[i], sentences2[j], score))

    return similar_pairs

# 테스트용 문서
doc1 = "Machine learning is a field of artificial intelligence. It focuses on learning from data."
doc2 = "Artificial intelligence includes machine learning. It emphasizes understanding data."

# 유사한 문장 검출
similar_sentences = find_similar_sentences(doc1, doc2, threshold=0.5)

# 결과 출력
for sent1, sent2, score in similar_sentences:
    print(f"문장1: {sent1}\n문장2: {sent2}\n유사도: {score:.2f}\n")
