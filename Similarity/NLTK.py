import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 두 문서의 내용
doc1 = """
딥러닝 기술은 최근에 매우 빠르게 발전하고 있습니다. 이 기술은 이미지 인식, 음성 인식 등 다양한 분야에서 활용되고 있으며, 의료, 금융, 교육 등 다양한 분야에서도 활용될 가능성이 높습니다.
"""

doc2 = """
딥러닝 기술은 최근에 매우 빠르게 발전하고 있습니다. 이 기술은 이미지 인식, 음성 인식 등 다양한 분야에서 활용되고 있으며, 의료, 금융, 교육 등 다양한 분야에서도 활용될 가능성이 높습니다.
"""


# 문서를 토큰화하고 불용어 제거
def preprocess(doc):
    tokens = nltk.word_tokenize(doc)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


preprocessed_doc1 = preprocess(doc1)
preprocessed_doc2 = preprocess(doc2)

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([preprocessed_doc1, preprocessed_doc2])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)

# 유사한 문장 검출
similar_scores = list(enumerate(cosine_sim[0]))
similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

for score in similar_scores:
    print(
        f"Score: {score[1]}, Sentence: {vectorizer.get_feature_names_out()[score[0]]}"
    )
