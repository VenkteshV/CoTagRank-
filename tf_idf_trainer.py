import os
import sys
import csv
import math
import glob
import pickle
import gzip
import json
from multiprocessing import Pool,Process,Manager

from sklearn.model_selection import GridSearchCV
import joblib

from pke.base import LoadFile
from pke.base import ISO_to_language

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def compute_tfidf_model(input_dir,
                      output_file,
                      extension="txt",
                      language="en",
                      normalization="stemming",
                      max_length=10**6):
    """Compute a TF-IDF model from a collection of documents.
    Args:
        input_dir (str): the input directory.
        output_file (str): the output file.
        extension (str): file extension for input documents, defaults to xml.
        language (str): language of the input documents, used for stop_words
            in sklearn CountVectorizer, defaults to 'en'.
        normalization (str): word normalization method, defaults to 'stemming'.
            Other possible values are 'lemmatization' or 'None' for using word
            surface forms instead of stems/lemmas.
    """

    # texts container
    texts = []
    file_list = []
    pool =Pool(20)

    for input_file in glob.iglob(input_dir + '/*.' + extension):
        file_list.append(input_file)

    for input_file in file_list:
        print("input_file", input_file)

        # initialize load file object to load the text files
        doc = LoadFile()

        # read the input file using utils from pke repository
        doc.load_document(input=input_file,
                          language=language,
                          normalization=normalization,
                          max_length=max_length)

        # current document placeholder
        text = []

        # loop through sentences
        for sentence in doc.sentences:
            # get the tokens (stems) from the sentence if they are not
            # punctuation marks 
            text.extend([sentence.stems[i] for i in range(sentence.length)
                         if sentence.pos[i] != 'PUNCT' and
                         sentence.pos[i].isalpha()])

        # add the document to the texts container
        texts.append(' '.join(text))
    
    tf_idf_vectorizer  = TfidfVectorizer()

    out =  tf_idf_vectorizer.fit_transform(texts)

    word2tfidf = dict(zip(tf_idf_vectorizer.get_feature_names(), tf_idf_vectorizer.idf_))

    # tf_idf_scores = dict(zip(tf_idf_vectorizer.get_feature_names(), out.toarray()[0]))

    for word, score in word2tfidf.items():
        print(word, score)

    joblib.dump(word2tfidf, "word2idf_inspec")

    joblib.dump(tf_idf_vectorizer, "tf_idf_vectorizer_inspec")

if __name__ == "__main__":
    input_path = "src/data/Datasets/Inspec/docsutf8"
    output_path = "keyphrase_expansion_lda.gz"
    compute_tfidf_model(
        input_path,
        output_file = output_path
    )