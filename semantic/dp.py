# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import copy
import enum
import math
import nltk
import pprint
import logging
import tempfile
import operator
import functools
import numpy as np
import pynnmf.pynnmf as pynnmf
import kneeliverse.lmethod as lmethod


from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from semantic.corpus import Corpus
from typing import Dict, List
from functools import lru_cache
from numba import jit


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
        ws = stemmer(wl)
        if ws is stem_target_word or (wl not in stop_words and wl.isalpha() and len(wl) >= l):
            filtered_tokens.append(wl)
    return filtered_tokens


def _nltk_pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif nltk_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:          
        return None


def _nltk_pos_lemmatizer(token, tag=None):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    if tag is None:
        return lemmatizer.lemmatize(token)
    else:        
        return lemmatizer.lemmatize(token, tag)


def _text_pre_processing(txt, m=2):
    if txt is not None:
        stop_words = set(nltk.corpus.stopwords.words('english'))

        tokens = nltk.word_tokenize(txt)
        tokens = [w.lower() for w in tokens]
        tokens = [w for w in tokens if w not in stop_words]
        tokens = [w for w in tokens if w.isalpha()]
        tokens = [w for w in tokens if len(w) > m]
        tokens = nltk.pos_tag(tokens)
        tokens = [(t[0], _nltk_pos_tagger(t[1])) for t in tokens]
        tokens = [_nltk_pos_lemmatizer(w, t) for w,t in tokens]
    else:
        tokens = []
    
    return tokens


def extract_neighborhood(target_word: str, corpus:List[str], n: int, l:int=3, c: Cutoff = Cutoff.pareto80) -> Dict:
    switcher = {
        Cutoff.knee: cutoff_knee,
        Cutoff.pareto20: cutoff_pareto20,
        Cutoff.pareto80: cutoff_pareto80,
        Cutoff.none: lambda n: len(n)
    }
    
    #snippets = ws.search(target_word)
    lemma_target_word = _nltk_pos_lemmatizer(target_word.lower())
    
    # Text Mining Pipeline
    #tokens = Parallel(n_jobs=-1)(delayed(sentences_to_tokens)(s, stem_target_word, stemmer, stop_words, l) for s in corpus)
    tokens = [_text_pre_processing(s) for s in corpus]
    tokens = [t for t in tokens if lemma_target_word in t]
    tokens = functools.reduce(operator.iconcat, tokens, [])

    # Search for target word
    neighborhood = {}
    for i in range(len(tokens)):
        st = tokens[i]
        if st == lemma_target_word:
            start = max(0, i-n)
            stop = min(len(tokens), i+n+1)
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


#TODO: Apply this to more areas of the code
@jit(nopython=True)
def _dpw_similarity(vector_a:np.ndarray, vector_b:np.ndarray, eps:float) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return np.dot(vector_a, vector_b)/max((norm_a*norm_b), eps)


def dpw_similarity(n_a: dict, n_b: dict) -> float:
    features = set(n_a) | set(n_b)
    vector_a = np.zeros(len(features))
    vector_b = np.zeros(len(features))
    i = 0
    for f in features:
        if f in n_a:
            vector_a[i] = n_a[f]

        if f in n_b:
            vector_b[i] = n_b[f]
        i += 1

    eps = math.ulp(1.0)
    return _dpw_similarity(vector_a, vector_b, eps)


class DPW:
    def __init__(self, word: str, neighborhood: list):
        self.word = _nltk_pos_lemmatizer(word)
        self.neighborhood = {}
        t = max_value = 0.0
        for k, v in neighborhood:
            if k not in self.neighborhood:
                self.neighborhood[k] = 0
            self.neighborhood[k] += v
            t += v
            if v > max_value:
                max_value = v
        
        # add itself if not in the profile
        if self.word not in self.neighborhood:
            self.neighborhood[self.word] = max_value

        # normalize the profile
        if t > 0:
            for k in self.neighborhood:
                self.neighborhood[k] /= t

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
        return f'Profile: {self.word}\nNeighborhood: {self.neighborhood}'

    def __repr__(self):
        return self.__str__()


class DPWC:
    def __init__(self, word: str, names: dict, neighborhood: list):
        self.word = _nltk_pos_lemmatizer(word)
        self.names = names
        self.neighborhood = neighborhood

    def _dpw_similarity(n_a, a_a, n_b, a_b):
        return dpw_similarity(n_a, n_b)*((a_a + a_b)/2.0)

    def similarity(self, dpwc: 'DPWC') -> float:
        if self.word == dpwc.word:
            return 1.0
        else:
            # Step up with joblib
            #ctx_similarity = Parallel(n_jobs=2)(delayed(DPWC._similarity)(n_a, a_a, n_b, a_b) for n_a, a_a in self.neighborhood for n_b, a_b in dpwc.neighborhood)
            ctx_similarity = []
            for n_a, a_a in self.neighborhood:
                for n_b, a_b in dpwc.neighborhood:
                    ctx_similarity.append(DPWC._dpw_similarity(n_a, a_a, n_b, a_b))
            return max(ctx_similarity)
            #return DPWC._similarity(self.neighborhood, dpwc.neighborhood)

    def __str__(self):
        names = pprint.pformat(self.names)
        neighborhood = pprint.pformat(self.neighborhood)
        return f'Profile: {self.word}\nNames: {names}\nNeighborhood({len(self.neighborhood)}): {neighborhood}'

    def __repr__(self):
        return self.__str__()


def nmf_optimization(dpw: DPW, Vr: np.ndarray) -> DPW:
    # load names
    #names = dpw.get_names()
    names = list(dpw.neighborhood)

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
        n[names[i]] = values[i]
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
    
    for t in list(dpw.neighborhood):
        # load RAW DPW
        temp_dpw = dpwm.get_RAW_DPW(t)
        if temp_dpw is None:
            dpw.neighborhood.pop(t, None)
    
    # reload names from valid profiles only
    #names = dpw.get_names()
    names = list(dpw.neighborhood)
    #print(f'DPW Names: {names}')
    size_names = len(names)

    # Create a square matrix
    V = np.zeros(shape=(size_names, size_names))

    # Fill the matrix
    for i in range(0, size_names-1):
        for j in range(i+1, size_names):
            dpw_i = dpwm.get_RAW_DPW(names[i])
            dpw_j = dpwm.get_RAW_DPW(names[j])
            value = max(dpw_i[names[j]], dpw_j[names[i]])
            V[i, j] = value
            V[j, i] = value
    
    np.fill_diagonal(V, 1.0)

    return V


def latent_analysis(V:np.ndarray, d: int=1, seeds:List[int]=[19, 23, 29, 31, 37, 41, 43]):
    # remove the diagonal (learned by the latent features)
    np.fill_diagonal(V, 0)

    # Learn the dimensions in latent space and reconstruct into token space
    k = max(len(V)//d, 1)

    #nmf_results = Parallel(n_jobs=-1)(delayed(pynnmf.nmf_mu_kl)(V, k, 100, 0.1, s) for s in seeds)
    nmf_results = [pynnmf.nmf_mu_kl(V, k, 100, 0.1, s) for s in seeds]
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
    #names = dpw.get_names()
    names = list(dpw.neighborhood)
    idx_word = names.index(dpw.word)
    size_names = len(names)

    logger.debug(f'Learn_DPWC({dpw.word}) {names} {idx_word} {size_names}')

    values = V[idx_word, :]
    D = 1.0 - V
    np.fill_diagonal(D, 0)

    best_score = -1
    best_n = 1 if size_names > 0 else 0
    labels = [0]*size_names
    limit = (size_names-1) if kl <= 0 else kl
    limit = min(size_names-1, limit)

    for n in range(2, limit):
        agg = AgglomerativeClustering(n_clusters=n, metric='precomputed', linkage = linkage)
        cluster_labels = agg.fit_predict(D)
        score = silhouette_score(D, cluster_labels, metric='precomputed')
        if score > best_score:
            best_n = n
            best_score = score
            labels = cluster_labels

    neighborhoods = build_neighborhoods(names, values, best_n, labels)

    # Build DPWCs
    return DPWC(dpw.word, names, neighborhoods)


class DPWModel:
    def __init__(self, corpus: Corpus, n:int=3, l:int=3,
    c: Cutoff = Cutoff.pareto20, continuous:bool=False,
    latent:bool=False, k:int=1):
        self.corpus = corpus
        self.n = n
        self.l = l
        self.c = c
        self.k = k
        self.continuous = continuous
        self.latent = latent
        self.profiles = {}
        if latent:
            self.cache = {}
        #self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.bias = 0.0

    def _fit_RAW_DPW(self, term:str):
        logger.debug(f'_fit_RAW_DPW({term})')
        #term = _nltk_pos_lemmatizer(term)
        # load the data from the corpus
        corpus = self.corpus.get(term)
        # create the original dpw
        n = extract_neighborhood(term, corpus, n=self.n, l=self.l, c=self.c)
        return DPW(term, n)

    def _fit(self, term:str):
        logger.debug(f'_fit({term})')
        rawDPW = self._fit_RAW_DPW(term)
        # store the profile in the rigth profile
        if self.latent:
            self.cache[term] = rawDPW
            for t in rawDPW.neighborhood:
                if t not in self.cache:
                    self.cache[t] = self._fit_RAW_DPW(t)
            V = co_occurrence_matrix(rawDPW, self)
            Vr = latent_analysis(V, d = self.k)
            self.profiles[term] = nmf_optimization(rawDPW, Vr)
        else:
            self.profiles[term] = rawDPW
        
        # compute the bias
        dp_len = len(self)
        if dp_len == 1:
            self.bias = 0.5
        elif dp_len == 2:
            w0, w1 = self.vocabulary()
            self.bias = self.similarity(w0, w1)
        else:
            term_bias = 0.0
            for w in self.vocabulary():
                if w != term:
                    term_bias += self.similarity(term, w)
            tot_pairs = (dp_len**2-dp_len)/2
            old_pairs = ((dp_len-1)**2-(dp_len-1))/2
            self.bias = (self.bias*old_pairs/tot_pairs) + (term_bias/tot_pairs)

    def fit(self, terms:List[str]):
        logger.debug(f'fit({terms})')
        for t in terms:
            # check if the term already exists in the cache
            lt = _nltk_pos_lemmatizer(t)
            if lt not in self.profiles:
                self._fit(lt)

    def similarity(self, w0:str, w1:str):
        w0 = _nltk_pos_lemmatizer(w0)
        w1 = _nltk_pos_lemmatizer(w1)
        
        logger.debug(f'similarity({w0}, {w1})')
        if self.continuous:
            self.fit([w0, w1])

        if w0 == w1:
            return 1.0
        elif self.profiles.get(w0,None) is None or self.profiles.get(w1, None) is None:
            return self.bias
        else:
            return self.profiles[w0].similarity(self.profiles[w1])
    
    def predict(self, w0:str, w1:str):
        logger.debug(f'predict({w0}, {w1})')
        return self.similarity(w0, w1)
    
    def get_RAW_DPW(self, term:str):
        logger.debug(f'get_RAW_DPW({term})')
        term = _nltk_pos_lemmatizer(term)
        return self.cache.get(term, None) if self.latent else self.profiles.get(term, None)
    
    def vocabulary(self):
        logger.debug(f'vocabulary(self)')
        rv = []
        for dp in self.profiles:
            if self.profiles[dp] is not None:
                rv.append(self.profiles[dp].word)
        return rv
        #return [self.profiles[dp].word[1] for dp in self.profiles]
    
    def __getitem__(self, key):
        logger.debug(f'DPW[{key}]')
        return self.profiles.get(key, None)

    def __len__(self):
        logger.debug(f'len(self) = {len(self.profiles)}')
        return len(self.profiles)
    
    def __contains__(self, key):
        logger.debug(f'in {key}')
        return key in self.profiles

    def __str__(self):
        txt = ''
        for _, p in self.profiles.items():
            txt += f'{p.__str__()}\n'
        return txt

    def __repr__(self):
        return self.__str__()


class DPWCModel:
    def __init__(self, corpus: Corpus, n: int = 3, l: int = 3, kl:int=0,
    c: Cutoff = Cutoff.pareto20, continuous=False, latent=False, k:int=1):
        self.latent = latent
        self.continuous = continuous
        self.kl = kl
        self.profiles = {}
        self.dpws = DPWModel(corpus, n=n, l=l, c=c, latent=False, k=k)
        self.bias = 0.0

    def _fit(self, term:str):
        logger.debug(f'DPWCModel _fit({term})')
        #term = _nltk_pos_lemmatizer(term)
        # train the DPW
        self.dpws._fit(term)
        dpw = self.dpws[term]
        logger.debug(f'DPW({term}) {dpw}')
        # train the DPW from the Neighbourhood
        logger.debug(f'NAMES: {list(dpw.neighborhood)}')
        for w in dpw.neighborhood:
            logger.debug(f'NAME {w}')
            if w not in self.dpws:
                self.dpws._fit(w)
        V = co_occurrence_matrix(dpw, self.dpws)
        if self.latent:
            V = latent_analysis(V, d = self.dpws.k)
        self.profiles[dpw.word] = learn_dpwc(dpw, V, kl=self.kl)

        logger.debug(f'DPWC({term}) {self.profiles[dpw.word]}')

        # compute the bias
        dp_len = len(self)
        if dp_len == 1:
            self.bias = 0.5
        elif dp_len == 2:
            w0, w1 = self.vocabulary()
            self.bias = self.similarity(w0, w1)
        else:
            term_bias = 0.0
            for w in self.vocabulary():
                if w != term:
                    term_bias += self.similarity(term, w)
            tot_pairs = (dp_len**2-dp_len)/2
            old_pairs = ((dp_len-1)**2-(dp_len-1))/2
            self.bias = (self.bias*old_pairs/tot_pairs) + (term_bias/tot_pairs)
        logger.debug(f'BIAS = {self.bias}')

    def fit(self, terms: List[str]):
        logger.debug(f'DPWCModel fit({terms})')
        for t in terms:
            # check if the term already exists in the 
            lt = _nltk_pos_lemmatizer(t)
            if lt not in self.profiles:
                self._fit(lt)

    def similarity(self, w0:str, w1:str):
        logger.debug(f'similarity({w0}, {w1})')
        #if self.continuous:
        #   self.fit([w0, w1])
        
        sw0 = _nltk_pos_lemmatizer(w0)
        sw1 = _nltk_pos_lemmatizer(w1)

        if sw0 == sw1:
            return 1.0
        elif self.profiles.get(sw0, None) is None or self.profiles.get(sw1,None) is None:
            return self.bias
        else:
            return self.profiles[sw0].similarity(self.profiles[sw1])
    
    def predict(self, w0:str, w1:str):
        logger.debug(f'predict({w0}, {w1})')
        return self.similarity(w0, w1)
    
    def vocabulary(self):
        logger.debug(f'vocabulary(self)')
        rv = []
        for dp in self.profiles:
            if self.profiles[dp] is not None:
                rv.append(self.profiles[dp].word)
        return rv
        #return [self.profiles[dp].word[1] for dp in self.profiles]

    def __getitem__(self, key):
        logger.debug(f'DPWC[{key}]')
        st = DPW.stem(key)
        return self.profiles.get(st, None)

    def __len__(self):
        logger.debug(f'len(self) = {len(self.profiles)}')
        return len(self.profiles)

    def __str__(self):
        txt = ''
        for _, p in self.profiles.items():
            txt += f'{p.__str__()}\n'
        return txt

    def __repr__(self):
        return self.__str__()
