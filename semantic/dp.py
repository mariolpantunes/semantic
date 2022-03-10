# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import enum
import copy
import nltk
import scipy
import pprint
import logging
import numpy as np

import nmf.nmf as nmf

import knee.lmethod as lmethod
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score

from semantic.corpus import Corpus  
# , davies_bouldin_score
#import skfuzzy as fuzz

from typing import Dict, List


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


def extract_neighborhood(target_word: str, corpus:List[str], n: int, stemmer, stop_words, l:int=1, c: Cutoff = Cutoff.pareto80) -> dict:
    switcher = {
        Cutoff.knee: cutoff_knee,
        Cutoff.pareto20: cutoff_pareto20,
        Cutoff.pareto80: cutoff_pareto80,
        Cutoff.none: lambda n: len(n)
    }
    
    #snippets = ws.search(target_word)
    stem_target_word = stemmer.stem(target_word)
    tokens = []
    # Text Mining Pipeline
    for s in corpus:
        temp_tokens = nltk.word_tokenize(s)
        filtered_tokens = [w.lower() for w in temp_tokens if not w in stop_words and w.isalpha() and len(w) > 2]
        tokens.extend(filtered_tokens)
    # Search for target word
    neighborhood = {}
    for i in range(len(tokens)):
        st = stemmer.stem(tokens[i])
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
    features_a = list(n_a.keys())
    features_b = list(n_b.keys())
    features = list(set(features_a + features_b))
    vector_a = []
    vector_b = []
    for f in features:
        if f in n_a:
            vector_a.append(n_a[f])
        else:
            vector_a.append(0.0)

        if f in n_b:
            vector_b.append(n_b[f])
        else:
            vector_b.append(0.0)

    a = np.array(vector_a)
    b = np.array(vector_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        #raise Exception(f'Dict_A\n{n_a}\nDict_B\n{n_b}\nFeatures\n{features}\nA: {a}\nB: {b}')
        return 0.0

    return np.dot(a, b)/(norm_a*norm_b)


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


class DPW_Cache():
    def __init__(self, n: int, ws, c: Cutoff = Cutoff.pareto20):
        self.n = n
        self.ws = ws
        self.cache = {}
        self.cutoff = c

    def __getitem__(self, key):
        if key not in self.cache:
            word_neighborhood = extract_neighborhood(
                key, self.ws, self.n, self.cutoff)
            if len(word_neighborhood) == 0:
                self.cache[key] = None
            else:
                self.cache[key] = DPW(key, word_neighborhood)
        return self.cache[key]

    def __len__(self):
        return len(self.cache)

    def __str__(self):
        return f'Cache: {list(self.cache.keys())}'

    def __repr__(self):
        return self.__str__()


class DPWC:
    def __init__(self, word: tuple, names: dict, neighborhood: list):
        self.word = word
        self.names = names
        self.neighborhood = neighborhood

    def similarity(self, dpwc: 'DPWC') -> float:
        similarities = []
        for n_a, a_a in self.neighborhood:
            for n_b, a_b in dpwc.neighborhood:
                similarity = dpw_similarity(n_a, n_b)
                similarity_with_affinity = similarity*((a_a+a_b)/2.0)
                #logger.debug('%s\n%s\nS=%s/%s\n\n', n_a, n_b, similarity, similarity_with_affinity)
                similarities.append(similarity_with_affinity)

        return max(similarities)

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
    if max(labels) >= n:
        raise Exception(
            f'Labels {labels} should not have value bigger than n ({n})')

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


def latent_analysis(dpw: DPW, d: int, dpw_cache: DPW_Cache):
    # pre-load all neighboors and remove neighborhood with weak profiles
    names = dpw.get_names()
    for s, w in names:
        temp_dpw = dpw_cache[w]
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
            dpw_i = dpw_cache[names[i][1]]
            dpw_j = dpw_cache[names[j][1]]
            value = max(dpw_i[names[j][0]], dpw_j[names[i][0]])
            V[i, j] = value
            V[j, i] = value

    # normalize the similarity matrix
    #sum_of_matrix = V.sum()
    # print(V)
    # print(sum_of_matrix)
    #V = V / sum_of_matrix

    # print(V)

    np.fill_diagonal(V, 1.0)

    # Learn the dimensions in latent space and reconstruct into token space
    k = len(names)//d

    seeds = [23, 29, 67, 71, 863, 937, 941, 997]
    best_Vr = V
    best_cost = float('inf')
    for s in seeds:
        Vr, _, _, cost = nmf.nmf_mu_kl(V, k, seed=s)
        #Vr, _, _, cost = nmf.rwnmf(V, k, seed=s)
        if cost < best_cost:
            best_Vr = Vr
            best_cost = cost

    # Recreate the simmetric matrix
    for i in range(0, size_names-1):
        for j in range(i+1, size_names):
            value = max(best_Vr[i, j], best_Vr[j, i])
            best_Vr[i, j] = value
            best_Vr[j, i] = value

    # normalize the similarity matrix
    #sum_of_matrix = best_Vr.sum()
    #best_Vr = best_Vr / sum_of_matrix
    best_Vr = np.clip(best_Vr, 0, 1)
    np.fill_diagonal(best_Vr, 1)

    #print(f'Cost = {best_cost}')

    # print(best_Vr)
    # input('wait...')

    return V, best_Vr


def learn_dpwc(dpw: DPW, V: np.ndarray, Vr: np.ndarray, m: str = 'average'):
    # load names and the right index
    names = dpw.get_names()
    idx_word = names.index(dpw.word)
    size_names = len(names)

    values = V[idx_word, :]
    D = 1.0 - V

    values_nmf = Vr[idx_word, :]
    D_nmf = 1.0 - Vr

    best_score = best_score_nmf = -1.0
    best_n = best_n_nmf = 0
    labels = labels_nmf = None

    #best_fpc = best_fpc_nmf = 0.0
    #best_u = best_u_nmf = 0

    scores = []
    scores_nmf = []

    # Compute HC
    ddgm = linkage(ssd.squareform(D), method=m)
    ddgm_nmf = linkage(ssd.squareform(D_nmf), method=m)

    # Hard and Soft Cluster
    for n in range(2, size_names-1):
        cluster_labels = scipy.cluster.hierarchy.fcluster(
            ddgm, n, criterion="maxclust")
        cluster_labels_nmf = scipy.cluster.hierarchy.fcluster(
            ddgm_nmf, n, criterion="maxclust")

        # TODO: Use this: https://stackoverflow.com/questions/47535256/how-to-make-fcluster-to-return-the-same-output-as-cut-tree
        try:
            score = silhouette_score(D, cluster_labels, metric='precomputed')
            scores.append(score)
            if score > best_score:
                best_n = n
                best_score = score
                labels = cluster_labels
        except:
            pass

        try:
            score_nmf = silhouette_score(
                D_nmf, cluster_labels_nmf, metric='precomputed')
            scores_nmf.append(score_nmf)
            if score_nmf > best_score_nmf:
                best_n_nmf = n
                best_score_nmf = score
                labels_nmf = cluster_labels
        except:
            pass

        # Fuzzy
        #_, u, _, _, _, _, fpc = fuzz.cluster.cmeans(V, n, 2, error=0.005, maxiter=1000, init=None)

        # if fpc > best_fpc:
        #    best_fpc = fpc
        #    best_u = u

        #_, u, _, _, _, _, fpc = fuzz.cluster.cmeans(Vr, n, 2, error=0.005, maxiter=1000, init=None)

        # if fpc > best_fpc_nmf:
        #    best_fpc_nmf = fpc
        #    best_u_nmf = u

    #logger.debug('Labels: %s (%s; %s)', labels, best_score, best_n)
    #logger.debug('Names: %s', names)

    labels = [x-1 for x in labels]
    labels_nmf = [x-1 for x in labels_nmf]
    neighborhoods_kmeans = build_neighborhoods(names, values, best_n, labels)
    neighborhoods_kmeans_nmf = build_neighborhoods(
        names, values_nmf, best_n_nmf, labels_nmf)

    # Fuzzy
    #weights = build_fuzzy_weights(best_u, idx_word)
    #weights_nmf = build_fuzzy_weights(best_u_nmf, idx_word)
    #neighborhoods_fuzzy = build_neighborhoods_fuzzy(names, values, weights)
    #neighborhoods_fuzzy_nmf = build_neighborhoods_fuzzy(names, values_nmf, weights_nmf)

    # Build DPWCs
    dpwc = (DPWC(dpw.word, dpw.names, neighborhoods_kmeans),
            DPWC(dpw.word, dpw.names, neighborhoods_kmeans_nmf))
    #DPWC(dpw.word, dpw.names, neighborhoods_fuzzy),
    # DPWC(dpw.word, dpw.names, neighborhoods_fuzzy_nmf))

    return dpwc


class DPWModel:
    def __init__(self, corpus: Corpus, n:int=3, l:int=1):
        self.corpus = corpus
        self.n = n
        self.l = l
        self.profiles = {}
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.stemmer = nltk.stem.PorterStemmer()

    def fit(self, terms:List[str]):
        for t in terms:
            # check if the term already exists in the cache
            if t not in self.profiles:
                # get the corpus
                c = self.corpus.get(t)
                n = extract_neighborhood(t, c, self.n, self.stemmer, self.stop_words, l=self.l)
                self.profiles[t] = DPW(t, n)

    def similarity(self, w0:str, w1:str):
        return self.profiles[w0].similarity(self.profiles[w1])
    
    def predict(self, w0:str, w1:str):
        return self.similarity(w0, w1)

    def __str__(self):
        txt = ''
        for _, p in self.profiles.items():
            txt += f'{p.__str__()}\n'
        return txt

    def __repr__(self):
        return self.__str__()


class DPWCModel:

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.profiles = {}

    def fit(self, terms:List[str]):
        for t in terms:
            # check if the term already exists in the cache
            if t not in self.profiles:
                pass
