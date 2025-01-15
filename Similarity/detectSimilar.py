from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def detect_similar_sentences(doc1, doc2, similarity_threshold=0.5):
    """
    두 문서에서 유사한 문장들을 찾아내는 함수

    Parameters:
    doc1 (str): 첫 번째 문서
    doc2 (str): 두 번째 문서
    similarity_threshold (float): 유사도 임계값 (기본값: 0.7)

    Returns:
    list: 유사한 문장 쌍과 유사도 점수를 포함하는 리스트
    """

    # 문장 분리
    def split_into_sentences(text):
        # 기본적인 문장 구분자로 분리
        sentences = text.replace('?', '.').replace('!', '.').split('.')
        # 빈 문자열 제거 및 공백 제거
        return [s.strip() for s in sentences if s.strip()]

    # 각 문서를 문장으로 분리
    sentences1 = split_into_sentences(doc1)
    sentences2 = split_into_sentences(doc2)

    # 모든 문장을 하나의 리스트로 결합
    all_sentences = sentences1 + sentences2

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix = vectorizer.fit_transform(all_sentences)

    # 문장 간 코사인 유사도 계산
    n_sentences1 = len(sentences1)
    similarity_matrix = cosine_similarity(tfidf_matrix[:n_sentences1], tfidf_matrix[n_sentences1:])

    # 유사한 문장 쌍 찾기
    similar_pairs = []
    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            similarity = similarity_matrix[i][j]
            if similarity >= similarity_threshold:
                similar_pairs.append({
                    'doc1_sentence': sentences1[i],
                    'doc2_sentence': sentences2[j],
                    'similarity_score': round(similarity, 3)
                })

    # 유사도 점수 기준으로 정렬
    similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)

    return similar_pairs

# 사용 예시
if __name__ == "__main__":
    # 예시 문서
    document1 = """
    인공지능은 현대 기술의 핵심입니다.
    딥러닝은 인공지능의 중요한 분야입니다.
    머신러닝은 데이터로부터 학습합니다.
    """

    document2 = """
    인공지능 기술은 현대 사회의 핵심 기술입니다.
    딥러닝은 인공지능 발전의 핵심 분야입니다.
    데이터 과학은 중요한 연구 분야입니다.
    """

    # 유사한 문장 찾기
    similar_sentences = detect_similar_sentences(document1, document2)

    # 결과 출력
    print("유사한 문장 쌍:")
    for pair in similar_sentences:
        print(f"\n유사도: {pair['similarity_score']}")
        print(f"문서1 문장: {pair['doc1_sentence']}")
        print(f"문서2 문장: {pair['doc2_sentence']}")
