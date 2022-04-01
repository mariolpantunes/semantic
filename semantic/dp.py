# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import copy
import nltk
import pprint
import logging
import operator
import functools
import numpy as np
import nmf.nmf as nmf
import knee.lmethod as lmethod
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

from semantic.corpus import Corpus

from typing import Dict, List
from joblib import Parallel, delayed


logger = logging.getLogger(__name__)


class Cutoff(enum.Enum):
    pareto20 = 'pareto20'
    pareto80 = 'pareto80'
    knee = 'knee'
    none = 'none'

    def __str__(self):
        return self.value


def cutoff_pareto20(neighborhood:Dict) -> int:
    return int(len(neighborhood)*0.2)


def cutoff_pareto80(neighborhood:Dict) -> int:
    limit = len(neighborhood)
    neighborhood_size = 0
    for _, v in neighborhood:
        neighborhood_size += v

    goal = neighborhood_size*0.8

    partial_goal = 0
    for i in range(len(neighborhood)):
        partial_goal += neighborhood[i][1]
        if partial_goal >= goal:
            limit = i
            break
    return limit


def cutoff_knee(neighborhood:Dict) -> int:
    points = []
    for i in range(len(neighborhood)):
        points.append([i, neighborhood[i][1]])
    points = np.array(points)
    limit = lmethod.knee(points)
    return limit


def sentences_to_tokens(s:str, stem_target_word:str, stemmer, stop_words, l:int)->List[str]:
    temp_tokens = nltk.word_tokenize(s)
    filtered_tokens = []
    for w in temp_tokens:
        wl = w.lower()
        ws = stemmer.stem(wl)
        if ws is stem_target_word or (wl not in stop_words and wl.isalpha() and len(wl) >= l):
            filtered_tokens.append(wl)
    return filtered_tokens


def extract_neighborhood(target_word: str, corpus:List[str], n: int, stemmer, stop_words, l:int=3, c: Cutoff = Cutoff.pareto80) -> Dict:
    switcher = {
        Cutoff.knee: cutoff_knee,
        Cutoff.pareto20: cutoff_pareto20,
        Cutoff.pareto80: cutoff_pareto80,
        Cutoff.none: lambda n: len(n)
    }
    
    #snippets = ws.search(target_word)
    stem_target_word = stemmer.stem(target_word.lower())
    
    # Text Mining Pipeline
    tokens = Parallel(n_jobs=-1)(delayed(sentences_to_tokens)(s, stem_target_word, stemmer, stop_words, l) for s in corpus)
    tokens = functools.reduce(operator.iconcat, tokens, [])

    # Search for target word
    neighborhood = {}
    for i in range(len(tokens)):
        st = stemmer.stem(tokens[i])
        #st = tokens[i]
        if st == stem_target_word:
            start = max(0, i-n)
            stop = min(len(tokens), i+n+1)
            # neighbors = tokens[i-n:i+n+1]
            neighbors = tokens[start:stop]
            for t in neighbors:
                if t not in neighborhood:
                    neighborhood[t] = 0
                neighborhood[t] += 1

    # Convert neighborhood into a list of tuples
    neighborhood = [(k, v) for k, v in neighborhood.items() if v > l]
    neighborhood.sort(key=lambda tup: tup[1], reverse=True)

    # Reduce the size of the vector
    limit = switcher[c](neighborhood)
    neighborhood = neighborhood[:limit]

    return neighborhood


def dpw_similarity(n_a: dict, n_b: dict) -> float:
    features_a = set(n_a.keys())
    features_b = set(n_b.keys())
    features = features_a.union(features_b)
    vector_a = np.zeros(len(features))
    vector_b = np.zeros(len(features))
    i = 0
    for f in features:
        if f in n_a:
            vector_a[i] = n_a[f]

        if f in n_b:
            vector_b[i] = (n_b[f])
        i += 1

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    else:
        return np.dot(vector_a, vector_b)/(norm_a*norm_b)


class DPW:
    def __init__(self, word: str, neighborhood: list):
        ps = nltk.stem.PorterStemmer()
        self.word = (ps.stem(word), word)
        # reduce neighborhood using the stem transformation
        self.neighborhood = {}
        self.names = {}
        t = max_value = 0.0
        for k, v in neighborhood:
            stem = ps.stem(k)
            if stem not in self.neighborhood:
                self.neighborhood[stem] = 0
                self.names[stem] = k
            elif len(self.names[stem]) > len(k):
                self.names[stem] = k
            self.neighborhood[stem] += v
            t += v
            if v > max_value:
                max_value = v

        # add itself if not in the profile
        if self.word[0] not in self.neighborhood:
            self.neighborhood[self.word[0]] = max_value
        self.names[self.word[0]] = word

        # normalize the profile
        if t > 0:
            for k in self.neighborhood:
                self.neighborhood[k] /= t

    def get_names(self):
        return [(k, v) for k, v in self.names.items()]

    def similarity(self, dpw: 'DPW') -> float:
        if self.word == dpw.word:
            return 1.0
        else:
            return dpw_similarity(self.neighborhood, dpw.neighborhood)

    def __getitem__(self, key):
        if key in self.neighborhood:
            return self.neighborhood[key]
        else:
            return 0

    def __len__(self):
        return len(self.neighborhood)

    def __str__(self):
        return f'Profile: {self.word}\nNeighborhood: {self.neighborhood}\nNames: {self.names}'

    def __repr__(self):
        return self.__str__()


class DPWC:
    def __init__(self, word: tuple, names: dict, neighborhood: list):
        self.word = word
        self.names = names
        self.neighborhood = neighborhood

    #TODO: improve the performance of this method
    def similarity(self, dpwc: 'DPWC') -> float:
        if self.word == dpwc.word:
            return 1.0
        else:
            best_similarity = 0.0
            for n_a, a_a in self.neighborhood:
                for n_b, a_b in dpwc.neighborhood:
                    similarity = dpw_similarity(n_a, n_b)
                    similarity_with_affinity = similarity*((a_a + a_b)/2.0)
                    if similarity_with_affinity > best_similarity:
                        best_similarity = similarity_with_affinity

            return best_similarity

    def __str__(self):
        names = pprint.pformat(self.names)
        neighborhood = pprint.pformat(self.neighborhood)
        return f'Profile: {self.word}\nNames: {names}\nNeighborhood({len(self.neighborhood)}): {neighborhood}'

    def __repr__(self):
        return self.__str__()


def nmf_optimization(dpw: DPW, Vr: np.ndarray) -> DPW:
    # load names
    names = dpw.get_names()

    idx_word = names.index(dpw.word)
    new_values = Vr[idx_word, :]

    # update DPW
    new_neighborhood = {}
    for i in range(len(names)):
        new_neighborhood[names[i][0]] = new_values[i]

    # Create new DPW
    new_dpw = copy.copy(dpw)
    new_dpw.neighborhood = new_neighborhood

    return new_dpw


def build_neighborhoods(names, values, n, labels):
    if max(labels) > n:
        raise Exception(f'Labels {labels} should not have value bigger than n ({n})')

    neighborhoods = [[{}, 0] for i in range(n)]
    for i in range(len(labels)):
        n, _ = neighborhoods[labels[i]]
        n[names[i][0]] = values[i]
        neighborhoods[labels[i]][1] += values[i]

    # Rescale and normalize affinity
    sum_aff = 0
    for i in range(len(neighborhoods)):
        neighborhoods[i][1] /= len(neighborhoods[i][0])
        sum_aff += neighborhoods[i][1]

    for i in range(len(neighborhoods)):
        neighborhoods[i][1] /= sum_aff

    return neighborhoods


def build_neighborhoods_fuzzy(names, values, weights):
    neighborhoods = [[{}, a] for a, _ in weights]

    for c in range(len(weights)):
        a, w = weights[c]
        n, _ = neighborhoods[c]
        for i in range(len(w)):
            n[names[i][0]] = values[i] * w[i]

    return neighborhoods


def build_fuzzy_weights(u, idx_word):
    weights = []
    for i in range(len(u)):
        w = u[i]
        aff = w[idx_word]
        weights.append((aff, w))
    return weights


def rank(array, reverse=False):
    return [sorted(array, reverse=reverse).index(x) for x in array]


def co_occurrence_matrix(dpw: DPW, dpwm: 'DPWModel'):
    # pre-load all neighboors and remove neighborhood with weak profiles
    names = dpw.get_names()
    for s, w in names:
        temp_dpw = dpwm[w]
        if temp_dpw is None:
            dpw.neighborhood.pop(s, None)
            dpw.names.pop(s, None)
    
    # reload names from valid profiles only
    names = dpw.get_names()
    size_names = len(names)

    # Create a square matrix
    V = np.zeros(shape=(size_names, size_names))

    # Fill the matrix
    for i in range(0, size_names-1):
        for j in range(i+1, size_names):
            dpw_i = dpwm[names[i][1]]
            dpw_j = dpwm[names[j][1]]
            value = max(dpw_i[names[j][0]], dpw_j[names[i][0]])
            V[i, j] = value
            V[j, i] = value
    
    np.fill_diagonal(V, 1.0)

    return V


def latent_analysis(V:np.ndarray, d: int=1, seeds:List[int]=[19, 23, 29, 31, 37, 41, 43]):
    # remove the diagonal (learned by the latent features)
    np.fill_diagonal(V, 0)

    # Learn the dimensions in latent space and reconstruct into token space
    k = max(len(V)//d, 1)

    nmf_results = Parallel(n_jobs=-1)(delayed(nmf.nmf_mu_kl)(V, k, 100, 0.1, s) for s in seeds)
    nmf_results.sort(key=lambda x:x[3])
    Vr = nmf_results[0][0]

    # Recreate the simmetric matrix
    for i in range(0, len(V)-1):
        for j in range(i+1, len(V)):
            value = max(Vr[i, j], Vr[j, i])
            Vr[i, j] = value
            Vr[j, i] = value

    # normalize the similarity matrix
    Vr = np.clip(Vr, 0, 1)
    
    return Vr


def learn_dpwc(dpw: DPW, V: np.ndarray, kl:int=0, linkage: str = 'average'):
    # load names and the right index
    names = dpw.get_names()
    idx_word = names.index(dpw.word)
    size_names = len(names)

    values = V[idx_word, :]
    D = 1.0 - V
    np.fill_diagonal(D, 0)

    best_score = -1
    best_n = 1 if size_names > 0 else 0
    labels = [0]*size_names
    limit = (size_names-1) if kl <= 0 else kl
    limit = min(size_names-1, limit)

    for n in range(2, limit):
        agg = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage = linkage)
        cluster_labels = agg.fit_predict(D)
        score = silhouette_score(D, cluster_labels, metric='precomputed')
        if score > best_score:
            best_n = n
            best_score = score
            labels = cluster_labels

    neighborhoods = build_neighborhoods(names, values, best_n, labels)

    # Build DPWCs
    return DPWC(dpw.word, dpw.names, neighborhoods)


class DPWModel:
    def __init__(self, corpus: Corpus, n:int=3, l:int=3,  c: Cutoff = Cutoff.pareto20, latent:bool=False, k:int=1):
        self.corpus = corpus
        self.n = n
        self.l = l
        self.c = c
        self.k = k
        self.latent = latent
        self.profiles = {}
        if latent:
            self.cache = {}
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()

    def _fit(self, term:str):
        st = self.stemmer.stem(term)
        dpw = self[term]
        if self.latent:
            if dpw is None:
                self.profiles[st] = None
            else:
                V = co_occurrence_matrix(dpw, self)
                Vr = latent_analysis(V, d = self.k)
                self.profiles[st] = nmf_optimization(dpw, Vr)

    def fit(self, terms:List[str]):
        for t in terms:
            # check if the term already exists in the cache
            st = self.stemmer.stem(t)
            if st not in self.profiles:
                self._fit(t)

    def similarity(self, w0:str, w1:str):
        self.fit([w0, w1])
        
        sw0 = self.stemmer.stem(w0)
        sw1 = self.stemmer.stem(w1)

        if self.profiles[sw0] is None or  self.profiles[sw1] is None:
            return 0.0
        else:
            return self.profiles[sw0].similarity(self.profiles[sw1])
    
    def predict(self, w0:str, w1:str):
        return self.similarity(w0, w1)
    
    def __getitem__(self, key):
        logger.debug(f'DPW[{key}]')
        st = self.stemmer.stem(key)

        # select the correct dictionary
        if self.latent:
            d = self.cache
        else:
            d = self.profiles

        if st not in d:
            # get the corpus
            corpus = self.corpus.get(key)
            # create the original dpw
            n = extract_neighborhood(key, corpus, self.n, self.stemmer, self.stop_words, c=self.c, l=self.l)
            if len(n) == 0:
                d[st] = None
            else:
                d[st] = DPW(key, n)
        
        return d[st]

    def __len__(self):
        if self.latent:
            d = self.cache
        else:
            d = self.profiles
        return len(d)

    def __str__(self):
        txt = ''
        for _, p in self.profiles.items():
            txt += f'{p.__str__()}\n'
        return txt

    def __repr__(self):
        return self.__str__()


class DPWCModel:
    def __init__(self, corpus: Corpus, n: int = 3, l: int = 3, kl:int=0, c: Cutoff = Cutoff.pareto20, latent=False, k:int=1):
        self.latent = latent
        self.kl = kl
        self.profiles = {}
        self.dpws = DPWModel(corpus, n=n, l=l, c=c, latent=False, k=k)
        self.stemmer = nltk.stem.PorterStemmer()

    def _fit(self, term:str):
        st = self.stemmer.stem(term)
        dpw = self.dpws[term]
        if dpw is not None:
            V = co_occurrence_matrix(dpw, self.dpws)
            if self.latent:
                V = latent_analysis(V, d = self.dpws.k)
            self.profiles[st] = learn_dpwc(dpw, V, kl=self.kl)
        else:
            self.profiles[st] = None

    def fit(self, terms: List[str]):
        for t in terms:
            # check if the term already exists in the cache
            st = self.stemmer.stem(t)
            if st not in self.profiles:
                self._fit(t)

    def similarity(self, w0:str, w1:str):
        self.fit([w0, w1])
        
        sw0 = self.stemmer.stem(w0)
        sw1 = self.stemmer.stem(w1)
        
        if self.profiles[sw0] is None or  self.profiles[sw1] is None:
            return 0.0
        else:
            return self.profiles[sw0].similarity(self.profiles[sw1])
    
    def predict(self, w0:str, w1:str):
        return self.similarity(w0, w1)

    def __str__(self):
        txt = ''
        for _, p in self.profiles.items():
            txt += f'{p.__str__()}\n'
        return txt

    def __repr__(self):
        return self.__str__()
