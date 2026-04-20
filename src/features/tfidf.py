"""TF-IDF featurizer with word + char-n-gram union."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf(
    word_ngram_range: tuple = (1, 2),
    char_ngram_range: tuple = (3, 5),
    max_features_word: int = 100_000,
    max_features_char: int = 100_000,
    min_df: int = 2,
) -> FeatureUnion:
    """Union of word and char TF-IDF. Robust default for short support text."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features_word,
        lowercase=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        sublinear_tf=True,
        min_df=min_df,
        max_features=max_features_char,
        lowercase=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
