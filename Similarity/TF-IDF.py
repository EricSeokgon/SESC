from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess_document(text):
    # 여기에 문서 전처리 로직 추가 (예: 소문자 변환, 특수문자 제거 등)
    return text.lower()

def check_copy(doc1, doc2, similarity_threshold=0.6):
    # 문서 전처리
    preprocessed_doc1 = preprocess_document(doc1)
    preprocessed_doc2 = preprocess_document(doc2)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([preprocessed_doc1, preprocessed_doc2])

    # 코사인 유사도 계산
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 유사도에 따른 카피 여부 판단
    is_copy = similarity >= similarity_threshold

    return is_copy, similarity

# 사용 예시
doc1 = "This is the first document. It contains some text."
doc2 = "This is the second document. It also contains some text, but it's slightly different."

is_copy, similarity = check_copy(doc1, doc2)
print(f"Is copy: {is_copy}")
print(f"Similarity: {similarity:.2f}")
