# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import copy
import nltk
import scipy
import pprint
import logging
import numpy as np

import semantic.nmf as nmf

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
import skfuzzy as fuzz


logger = logging.getLogger(__name__)


def extract_neighborhood(target_word: str, ws, n: int) -> {}:
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
    def __init__(self, word, neighborhood: list):
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
    
    def __len__(self):
        return len(self.neighborhood)
    
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
            V[i,j] = value
            V[j,i] = value
    
    # normalize the similarity matrix
    #sum_of_matrix = V.sum()
    #print(V)
    #print(sum_of_matrix)
    #V = V / sum_of_matrix
    
    #print(V)
    

    np.fill_diagonal(V, 1.0)


    # Learn the dimensions in latent space and reconstruct into token space
    k = len(names)//d
    
    seeds = [23,29,67,71,863,937,941,997]
    best_Vr = V
    best_cost = float('inf') 
    for s in seeds:
        Vr, _, _, cost = nmf.nmf_mu(V, k, seed=s)
        #Vr, _, _, cost = nmf.rwnmf(V, k, seed=s)
        if cost < best_cost:
            best_Vr = Vr
            best_cost = cost

    # Recreate the simmetric matrix
    for i in range(0, size_names-1):
        for j in range(i+1, size_names):
            value = max(best_Vr[i,j], best_Vr[j,i])
            best_Vr[i,j] = value
            best_Vr[j,i] = value

    # normalize the similarity matrix
    #sum_of_matrix = best_Vr.sum()
    #best_Vr = best_Vr / sum_of_matrix
    best_Vr = np.clip(best_Vr, 0, 1)
    np.fill_diagonal(best_Vr, 1)

    #print(f'Cost = {best_cost}')

    #print(best_Vr)
    #input('wait...')

    return V, best_Vr


def learn_dpwc(dpw: DPW, V: np.ndarray, Vr: np.ndarray, m:str='average'):
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

    best_fpc = best_fpc_nmf = 0.0
    best_u = best_u_nmf = 0

    scores = []
    scores_nmf = []

    # Compute HC
    ddgm = linkage(ssd.squareform(D), method=m)
    ddgm_nmf = linkage(ssd.squareform(D_nmf), method=m)

    # Hard and Soft Cluster
    for n in range(2, size_names-1):
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
        #_, u, _, _, _, _, fpc = fuzz.cluster.cmeans(V, n, 2, error=0.005, maxiter=1000, init=None)

        #if fpc > best_fpc:
        #    best_fpc = fpc
        #    best_u = u
        
        #_, u, _, _, _, _, fpc = fuzz.cluster.cmeans(Vr, n, 2, error=0.005, maxiter=1000, init=None)

        #if fpc > best_fpc_nmf:
        #    best_fpc_nmf = fpc
        #    best_u_nmf = u

    #logger.debug('Labels: %s (%s; %s)', labels, best_score, best_n)
    #logger.debug('Names: %s', names)

    labels = [x-1 for x in labels]
    labels_nmf = [x-1 for x in labels_nmf]
    neighborhoods_kmeans = build_neighborhoods(names, values, best_n, labels)
    neighborhoods_kmeans_nmf = build_neighborhoods(names, values_nmf, best_n_nmf, labels_nmf)

    # Fuzzy 
    #weights = build_fuzzy_weights(best_u, idx_word)
    #weights_nmf = build_fuzzy_weights(best_u_nmf, idx_word)
    #neighborhoods_fuzzy = build_neighborhoods_fuzzy(names, values, weights)
    #neighborhoods_fuzzy_nmf = build_neighborhoods_fuzzy(names, values_nmf, weights_nmf)

    # Build DPWCs
    dpwc = (DPWC(dpw.word, dpw.names, neighborhoods_kmeans),
    DPWC(dpw.word, dpw.names, neighborhoods_kmeans_nmf))
    #DPWC(dpw.word, dpw.names, neighborhoods_fuzzy),
    #DPWC(dpw.word, dpw.names, neighborhoods_fuzzy_nmf))

    return dpwc