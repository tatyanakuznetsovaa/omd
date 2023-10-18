class CountVectorizer:
    """Convert a collection of text documents to a matrix of token counts"""

    def __init__(self):
        self.words = {}

    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """Learn the vocabulary dictionary and return document-term matrix"""
        order = 0
        for elem in corpus:
            word_of_sent = elem.lower().split()
            for word in word_of_sent:
                if word not in self.words:
                    self.words[word] = order
                    order += 1

        matrix = []
        for sent in corpus:
            row = [0] * len(self.words)
            for key_word in self.words:
                row[self.words[key_word]] = sent.lower().count(key_word)

            matrix.append(row)

        return matrix

    def get_feature_names(self) -> list[str]:
        """Get output feature names for transformation"""
        return list(self.words.keys())


if __name__ == "__main__":
    corpus = [
        "Crock Pot Pasta Never boil pasta again",
        "Pasta Pomodoro Fresh ingredients Parmesan to taste",
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)