import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from sklearn.feature_extraction.text import CountVectorizer
import statistics
import joblib
import networkx as nx
from nltk.stem.snowball import SnowballStemmer
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

word_to_idf = joblib.load(dir_path+"/../../data/Models/Unsupervised/lda/word2idf_semeval2010")
tf_idf_vectorizer = joblib.load(dir_path+"/../../data/Models/Unsupervised/lda/tf_idf_vectorizer_semeval2010")
stemmer = SnowballStemmer("porter")
class CoTagRank:
    """Implementation of unsupervised `phrase` extraction method using USE and topic embeddings or topically guided pre trained sentence embeddings and our custom ranking algorithm. This method tries to
    find important phrases in text using analysis of their cosine similarity to original text and using reranking method to choose most relevant and also diverse phrases.

         phrase: i.e. `noun phrases`  which are actually chunks of nouns that represent
         important parts of sentence. This is assumed to be good selection of candidates for phrases."""
    def  __init__(self, emb_method='NAIVE', mmr_beta=0.55, top_n=15, alias_threshold=0.8):
        self.top_n = top_n
        self.alias_threshold = alias_threshold
        self.graph = nx.Graph()


    def run(self, text, phrases, text_emb, phrase_embs, highlight=False):
        top_phrases, aliases = self.RankPhrases(text_emb, phrases, phrase_embs,
                                                   self.top_n, self.alias_threshold, highlight, text)

        return top_phrases, aliases

    def RankPhrases(self, text_emb, phrases, phrase_embs, top_n=10, alias_threshold=0.8, highlight = False, og_doc_text=None):
        print("in rank",phrase_embs.shape,text_emb.shape)
        manhaatan_dist = manhattan_distances(phrase_embs, [text_emb])
        text_sims = cosine_similarity(phrase_embs, [text_emb])
        phrase_sims = cosine_similarity(phrase_embs)
        
        # normalize cosine similarities
        text_sims_norm = self.standardize_normalize_cosine_similarities(text_sims)
        # keep indices of selected and unselected phrases in list
        selected_phrase_indices = []
        unselected_phrase_indices = list(range(len(phrases)))

        doc_rel_manhaatan_dist = (manhaatan_dist[unselected_phrase_indices]).squeeze(axis=1).tolist()
        document_relevance = (text_sims_norm[unselected_phrase_indices]).squeeze(axis=1).tolist()
        # aliases_phrases = self.get_alias_phrases(phrase_sims[unselected_phrase_indices, :], phrases, alias_threshold)
        top_phrases = [phrases[idx] for idx in unselected_phrase_indices]

        relevance_dict = {}
        sim_importance = 0.80

        idf_phrase_dict ={}

        tf_phrase_dict = {}

        text =[]

        vectorizer = CountVectorizer()
        freq_counts = vectorizer.fit_transform([og_doc_text])
        freq_counts = freq_counts.toarray()
        feature_names = vectorizer.get_feature_names()

        position_scores ={}



        for (keyword, start_pos, end_pos), score, manhattan in zip(top_phrases, document_relevance, doc_rel_manhaatan_dist):
            words = keyword.split()
            phrase_idf_score = 0.0
            phrase_tf_score = 0.0
            for word in words:
                if word in feature_names:
                    word_index = feature_names.index(word)
                    word_freq = freq_counts[0][word_index]
                else:
                    word_freq = 1   
                if stemmer.stem(word) in word_to_idf:
                    word_idf_score = word_to_idf[stemmer.stem(word)]
                    # word_idf_score = (word_idf_score - min(word_to_idf.values())) / (max(word_to_idf.values()) - min(word_to_idf.values()))
                    phrase_idf_score += (word_idf_score)
                    phrase_tf_score += word_freq
                else:
                    phrase_idf_score = 0.00001
            position_scores[keyword] = 1/(1.0+start_pos)
            idf_phrase_dict[keyword] = phrase_idf_score
            tf_phrase_dict[keyword] = phrase_tf_score
            relevance_dict[keyword] = ((1/manhattan)) * phrase_idf_score * phrase_tf_score * (1/len(keyword.split()))
        # for key, value in relevance_dict.items():
        #     position_scores[key] = np.exp(position_scores[key]) / np.sum(np.exp(list(position_scores.values())))
        #     relevance_dict[key] = relevance_dict[key]

# sim_importance * score + (1-sim_importance) * (1.0/(0.000001+start_pos))
        phrase_to_embedding = {}
        for index, phrase in enumerate(phrases):
            phrase_to_embedding[phrase[0]] = phrase_embs[index]
        # norm = sum(relevance_dict.values())
        # # for word in relevance_dict:
        # #     relevance_dict[word] /= norm
        graph = nx.Graph()
        minx =-1
        maxx =1
        # graph.add_weighted_edges_from(
        #     [(v[0], u[0], np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])) for v in top_phrases for u in top_phrases if u != v])
        # graph.add_weighted_edges_from(
        #     [(v[0], u[0], (0.8*(((np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])/((np.linalg.norm(phrase_to_embedding[v[0]]) * np.linalg.norm(phrase_to_embedding[u[0]]))+1e-07)) - minx) / (maxx-minx) ) + 0.2 * (1.0/(1+u[1])) ) ) for v in top_phrases for u in top_phrases if u != v])
        # graph.add_weighted_edges_from(
        #     [(v[0], u[0], ( 0.8 * (((np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])) - minx) / (maxx-minx) ) + (0.2* position_scores[u[0]] ) ) ) for v in top_phrases for u in top_phrases if u != v])       
        graph.add_weighted_edges_from(
            [(v[0], u[0], ( (((np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])/((np.linalg.norm(phrase_to_embedding[v[0]]) * np.linalg.norm(phrase_to_embedding[u[0]]))+1e-07)) - minx) / (maxx-minx) ) ) ) for v in top_phrases for u in top_phrases if u != v])       
   
        # print("edges",graph.edges.data(),len(top_phrases), len(graph.edges.data()))
        pr = nx.pagerank(graph, personalization=relevance_dict,
                        alpha=0.85,
                        tol=0.0001, weight='weight')

        concepts = sorted([(b, a.lstrip()) for a, b in pr.items()], reverse=True)[:top_n]
        # concepts = [(score, keyword.lstrip()) for (keyword, _, _), score in zip(top_phrases, relevance)]
        
        if highlight:
            phrases_only = [phrase[0].lstrip() for phrase in phrases]
            # print("concepts", concepts,'phrases', phrases_only)
            phrases_selected = [phrases[phrases_only.index(phrase[1])] for phrase in concepts ]
            return concepts, phrases_selected

        return concepts, None

    # def rerank(self,text_emb, phrases, phrase_embs, top_n=10, alias_threshold=0.8, highlight = False):

        # text_sims = cosine_similarity(phrase_embs, [text_emb])
        # phrase_sims = cosine_similarity(phrase_embs)
        # # normalize cosine similarities
        # text_sims_norm = self.standardize_normalize_cosine_similarities(text_sims)

        # # keep indices of selected and unselected phrases in list
        # selected_phrase_indices = []
        # unselected_phrase_indices = list(range(len(phrases)))

        # document_relevance = (text_sims_norm[unselected_phrase_indices]).squeeze(axis=1).tolist()
        # # aliases_phrases = self.get_alias_phrases(phrase_sims[unselected_phrase_indices, :], phrases, alias_threshold)
        # top_phrases = [phrases[idx] for idx in unselected_phrase_indices]

        # relevance_dict = {}
        # print("top_phrases", top_phrases)

        # for (keyword, _, _), score in zip(top_phrases, document_relevance):
        #     relevance_dict[keyword] = score

        # phrase_to_embedding = {}
        # for index, phrase in enumerate(phrases):
        #     phrase_to_embedding[phrase[0]] = phrase_embs[index]
        # # norm = sum(relevance_dict.values())
        # # # for word in relevance_dict:
        # # #     relevance_dict[word] /= norm
        # graph = nx.Graph()
        # graph.add_weighted_edges_from(
        #     [(v[0], u[0], np.dot(phrase_to_embedding[v[0]], phrase_to_embedding[u[0]])) for v in top_phrases for u in top_phrases if u != v])
        # pr = nx.pagerank(graph, personalization=relevance_dict,
        #                 alpha=0.85,
        #                 tol=0.0001, weight='weight')

        # concepts = sorted([(b, a.lstrip()) for a, b in pr.items()], reverse=True)[:top_n]
        # # concepts = [(score, keyword.lstrip()) for (keyword, _, _), score in zip(top_phrases, relevance)]
        
        # if highlight:
        #     phrases_only = [phrase[0] for phrase in phrases]
        #     # print("concepts", concepts,'phrases', phrases)
        #     phrases_selected = [phrases[phrases_only.index(phrase[1])] for phrase in concepts ]
        #     return concepts, phrases_selected

        # return concepts, None
    @staticmethod
    def standardize_normalize_cosine_similarities(cosine_similarities):
        """Normalized and standardized (or z score) cosine similarities"""
        # normalize into 0-1 range
        cosine_sims_norm = (cosine_similarities - np.min(cosine_similarities)) / (
                np.max(cosine_similarities) - np.min(cosine_similarities))

        # standardize and shift by 0.5 following the method in embedrank
        cosine_sims_norm = 0.5 + (cosine_sims_norm - np.mean(cosine_sims_norm)) / np.std(cosine_sims_norm)

        return cosine_sims_norm
