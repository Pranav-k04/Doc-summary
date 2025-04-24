from sklearn.feature_extraction.text import TfidfVectorizer

def classify_text(text, topics):
    if not topics or not text:
        return []

    corpus = [text] + topics
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    similarities = (vectorizer[0] @ vectorizer[1:].T).toarray()[0]
    results = list(zip(topics, similarities))
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'topic': t, 'score': float(f'{s:.4f}')} for t, s in results[:5]]
