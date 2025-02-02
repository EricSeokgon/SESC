import nltk

nltk.download("punkt_tab")  # Downloading 'punkt_tab' dataset
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 문서 전처리
def preprocess_documents(doc1, doc2):
    sentences1 = sent_tokenize(doc1)
    sentences2 = sent_tokenize(doc2)
    return sentences1, sentences2


# 문장 벡터화
def vectorize_sentences(sentences1, sentences2):
    vectorizer = TfidfVectorizer()
    all_sentences = sentences1 + sentences2
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    return tfidf_matrix


# 유사도 계산 및 유사한 문장 검출
def find_similar_sentences(tfidf_matrix, sentences1, sentences2, threshold=0.5):
    # Compute the cosine similarity matrix for the TF-IDF representations
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Number of sentences in the first document
    n = len(sentences1)

    # List to hold pairs of similar sentences
    similar_pairs = []

    # Iterate over each sentence in the first document
    for i in range(n):
        # Iterate over each sentence in the second document
        for j in range(n, len(sentences1) + len(sentences2)):
            # Check if the similarity score is above the threshold
            if similarity_matrix[i][j] > threshold:
                # Append the pair of sentences and their similarity score to the list
                similar_pairs.append(
                    (sentences1[i], sentences2[j - n], similarity_matrix[i][j])
                )

    return similar_pairs


# 메인 함수
def compare_documents(doc1, doc2, threshold=0.5):
    sentences1, sentences2 = preprocess_documents(doc1, doc2)
    tfidf_matrix = vectorize_sentences(sentences1, sentences2)
    similar_pairs = find_similar_sentences(
        tfidf_matrix, sentences1, sentences2, threshold
    )

    return similar_pairs


# 사용 예시
doc1 = "This is the first document. It contains some text."
doc2 = "This is the second document. It also contains some text, but it's slightly different."

similar_pairs = compare_documents(doc1, doc2, threshold=0.5)

for pair in similar_pairs:
    print(f"Sentence 1: {pair[0]}")
    print(f"Sentence 2: {pair[1]}")
    print(f"Similarity: {pair[2]:.2f}")
    print()
