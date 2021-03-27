# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import nmf
import copy
import nltk
import scipy
import pprint
import logging
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from typing import List, Dict, Tuple
from sklearn.metrics import silhouette_score,  davies_bouldin_score


import skfuzzy as fuzz


logger = logging.getLogger(__name__)


def extract_neighborhood(target_word: str, ws, n: int) -> Dict:
    snippets = ws.search(target_word)
    ps = nltk.stem.PorterStemmer()
    stem_target_word = ps.stem(target_word)
    tokens = []
    # Text Mining Pipeline
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    for s in snippets:
        temp_tokens = nltk.word_tokenize(s)
        filtered_tokens = [w.lower() for w in temp_tokens if not w in stop_words and w.isalpha() and len(w) > 2] 
        tokens.extend(filtered_tokens)
    logger.debug(tokens)
    logger.debug('Total number of tokens: %s', len(tokens))
    # Search for target word
    neighborhood = {} 
    for i in range(0, len(tokens)):
        st = ps.stem(tokens[i])
        if st == stem_target_word:
            neighbors = tokens[i-n: i+n+1]
            logger.debug(neighbors)
            for t in neighbors:
                if t not in neighborhood:
                    neighborhood[t] = 0
                neighborhood[t] += 1
    logger.debug(neighborhood)
    # Convert neighborhood into a list of tuples
    neighborhood = [(k, v) for k, v in neighborhood.items() if v > 1] 
    neighborhood.sort(key=lambda tup: tup[1], reverse=True)
    logger.debug(neighborhood)
    limit = int(len(neighborhood)*0.2)
    
    logger.debug('%s/%s', len(neighborhood), limit)
    neighborhood = neighborhood[:limit]
    
    #x_val = [x[0] for x in neighborhood[:limit]]
    #y_val = [x[1] for x in neighborhood[:limit]]
    #logger.debug(len(x_val))
    #plt.plot(x_val,y_val)
    #plt.show()
    return neighborhood


def dpw_similarity(n_a: Dict, n_b: Dict) -> float:
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
            raise Exception(f'Dict_A\n{n_a}\nDict_B\n{n_b}\nFeatures\n{features}\nA: {a}\nB: {b}')
            return 0.0

        return np.dot(a, b)/(norm_a*norm_b)


class DPW:
    def __init__(self, word, neighborhood: List):
        ps = nltk.stem.PorterStemmer()
        self.word = (ps.stem(word), word)
        # reduce neighborhood using the stem transformation
        self.neighborhood = {}
        self.names = {}
        t = max_value = 0.0
        for k,v in neighborhood:
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
    
    def __str__(self):
        return f'Profile: {self.word}\nNeighborhood: {self.neighborhood}\nNames: {self.names}'
    
    def __repr__(self):
        return self.__str__()


class DPW_Cache():
    def __init__(self, n: int, ws):
        self.n = n
        self.ws = ws
        self.cache = {}
    
    def __getitem__(self, key):
        if key not in self.cache:
            word_neighborhood = extract_neighborhood(key, self.ws, self.n)
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
    def __init__(self, word: Tuple, names: Dict, neighborhood: List):
        self.word = word
        self.names = names
        self.neighborhood = neighborhood
    
    def similarity(self, dpwc: 'DPWC') -> float:
        similarities = []
        for n_a, a_a in self.neighborhood:
            for n_b, a_b in dpwc.neighborhood:
                similarity = dpw_similarity(n_a, n_b)
                similarity_with_affinity = similarity*((a_a+a_b)/2.0)
                #logger.info('%s\n%s\nS=%s/%s\n\n', n_a, n_b, similarity, similarity_with_affinity)
                similarities.append(similarity_with_affinity)

        return max(similarities)

    def __str__(self):
        names = pprint.pformat(self.names)
        neighborhood = pprint.pformat(self.neighborhood)
        return f'Profile: {self.word}\nNames: {names}\nNeighborhood({len(self.neighborhood)}): {neighborhood}'
    
    def __repr__(self):
        return self.__str__()


def nmf_optimization(dpw: DPW, d: int, dpw_cache: DPW_Cache):
    # pre-load all neighboors
    names = dpw.get_names()
    for s, w in names:
        temp_dpw = dpw_cache[w]
        if temp_dpw is None:
            dpw.neighborhood.pop(s, None)
            dpw.names.pop(s, None)
    
    # reload names
    names = dpw.get_names()

    # Create a square matrix
    V = np.zeros(shape=(len(names), len(names)))

    idx_word = names.index(dpw.word)

    # Fill the matrix
    for i in range(len(names)):
        for j in range(len(names)):
            dpw_i = dpw_cache[names[i][1]]
            dpw_j = dpw_cache[names[j][1]]
            V[i,j] = max(dpw_i[names[j][0]], dpw_j[names[i][0]])
    logger.debug(V[idx_word, :])
    
    k = len(names)//d
    W, H = nmf.nmf_nnls(V, k)
    new_values = np.dot(W, H)[idx_word, :]

    #Vr, *_ = nmf.rwnmf(V, k)
    #new_values = Vr[idx_word, :]
    
    # update DPW
    new_neighborhood = {}
    for i in range(len(names)):
        new_neighborhood[names[i][0]] = new_values[i]
    #dpw.neighborhood = new_neighborhood

    new_dpw = copy.copy(dpw)
    new_dpw.neighborhood = new_neighborhood

    return new_dpw


def build_neighborhoods(names, values, n, labels):
    if max(labels) >= n:
        raise Exception(f'Labels {labels} should not have value bigger than n ({n})')

    neighborhoods = [[{}, 0] for i in range(n)]
    for i in range(len(labels)):
       n, _ = neighborhoods[labels[i]]
       n[names[i][0]] = values[i]
       neighborhoods[labels[i]][1] += values[i]
    
    # Rescale and normalize affinity
    sum_aff  = 0
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


def learn_dpwc(dpw: DPW, d: int, dpw_cache: DPW_Cache, r=3):
    # pre-load all neighboors
    names = dpw.get_names()
    for s, w in names:
        temp_dpw = dpw_cache[w]
        if temp_dpw is None:
            dpw.neighborhood.pop(s, None)
            dpw.names.pop(s, None)
    
    # reload names
    names = dpw.get_names()

    # Create a square matrix
    V = np.zeros(shape=(len(names), len(names)))

    idx_word = names.index(dpw.word)

    # Fill the matrix
    for i in range(len(names)):
        for j in range(len(names)):
            dpw_i = dpw_cache[names[i][1]]
            dpw_j = dpw_cache[names[j][1]]
            V[i,j] = max(dpw_i[names[j][0]], dpw_j[names[i][0]])
    
    sum_of_rows = V.sum(axis=1)
    V = V / sum_of_rows[:, np.newaxis]
    np.fill_diagonal(V, 1)

    values = V[idx_word, :]
    D = 1.0 - V
    
    
    k = len(names)//d
    W, H = nmf.nmf_nnls(V, k)
    best_cost = nmf.cost(V, W, H)
    for i in range(r):
        tW, tH = nmf.nmf_nnls(V, k)
        c = nmf.cost(V, W, H)
        if c < best_cost:
            c = best_cost
            W = tW
            H = tH
    VR = np.dot(W, H)

    sum_of_rows = VR.sum(axis=1)
    VR = VR / sum_of_rows[:, np.newaxis]
    np.fill_diagonal(VR, 1)
    D_nmf = 1.0 - VR
    values_nmf = VR[idx_word, :]

    best_score = best_score_nmf = -1.0
    best_n = best_n_nmf = 0
    labels = labels_nmf = None

    best_fpc = best_fpc_nmf = 0.0
    best_u = best_u_nmf = 0

    scores = []
    scores_nmf = []

    #votes = np.zeros(len(names)-2)
    #votes_nmf = np.zeros(len(names)-2)

    # Compute HC
    ddgm = linkage(D, method="average")
    ddgm_nmf = linkage(D_nmf, method="average")

    # K-means and Fuzzy C-Means
    for n in range(2, len(names)-1):
        '''km = KMeans(n_clusters=n, init='k-means++')
        cluster_labels = km.fit_predict(V)
        #score = silhouette_score(V, cluster_labels)
        score = davies_bouldin_score(V, cluster_labels)
        if score < best_score:
            best_n = n
            best_score = score
            labels = cluster_labels

        km_nmf = KMeans(n_clusters=n, init='k-means++')
        cluster_labels_nmf = km_nmf.fit_predict(VR)
        score_nmf = davies_bouldin_score(VR, cluster_labels_nmf)
        if score_nmf < best_score_nmf:
            best_n_nmf = n
            best_score_nmf = score
            labels_nmf = cluster_labels'''
        
        cluster_labels = scipy.cluster.hierarchy.fcluster(ddgm, n, criterion="maxclust")
        cluster_labels_nmf = scipy.cluster.hierarchy.fcluster(ddgm_nmf, n, criterion="maxclust")

        score = silhouette_score(D, cluster_labels, metric='precomputed')
        scores.append(score)
        if score > best_score:
            best_n = n
            best_score = score
            labels = cluster_labels
        
        score_nmf = silhouette_score(D_nmf, cluster_labels_nmf, metric='precomputed')
        scores_nmf.append(score_nmf)
        if score_nmf > best_score_nmf:
            best_n_nmf = n
            best_score_nmf = score
            labels_nmf = cluster_labels

        #Fuzzy
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(V, n, 2, error=0.005, maxiter=1000, init=None)

        if fpc > best_fpc:
            best_fpc = fpc
            best_u = u
        
        _, u, _, _, _, _, fpc = fuzz.cluster.cmeans(VR, n, 2, error=0.005, maxiter=1000, init=None)

        if fpc > best_fpc_nmf:
            best_fpc_nmf = fpc
            best_u_nmf = u

    #logger.debug('Labels: %s (%s; %s)', labels, best_score, best_n)
    #logger.debug('Names: %s', names)

    labels = [x-1 for x in labels]
    labels_nmf = [x-1 for x in labels_nmf]
    neighborhoods_kmeans = build_neighborhoods(names, values, best_n, labels)
    neighborhoods_kmeans_nmf = build_neighborhoods(names, values_nmf, best_n_nmf, labels_nmf)

    # Fuzzy 
    weights = build_fuzzy_weights(best_u, idx_word)
    weights_nmf = build_fuzzy_weights(best_u_nmf, idx_word)
    neighborhoods_fuzzy = build_neighborhoods_fuzzy(names, values, weights)
    neighborhoods_fuzzy_nmf = build_neighborhoods_fuzzy(names, values_nmf, weights_nmf)

    # Build DPWCs
    dpwc = (DPWC(dpw.word, dpw.names, neighborhoods_kmeans),
    DPWC(dpw.word, dpw.names, neighborhoods_kmeans_nmf),
    DPWC(dpw.word, dpw.names, neighborhoods_fuzzy),
    DPWC(dpw.word, dpw.names, neighborhoods_fuzzy_nmf))

    return dpwc