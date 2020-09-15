# coding: utf8

import numpy as np
from collections import Counter, defaultdict
from scipy import spatial


class WSDKnnClassifier:
    """ Implements a Knn classifier to perform word sense disambiguation."""

    def __init__(self, average=False, k=1, mfs_backoff=False, mfs_files=None):

        self.average = average
        self.k = k
        self.mfs_backoff = mfs_backoff
        if mfs_backoff:
            self.id2mfs = self.read_mfs_file(*mfs_files)
        self.is_fit = False

    def fit(self, dataset):
        self.train = dataset.key2instances
        self.is_fit = True

    def compute_mfs(self, dataset):
        label2count = Counter()

        for instance_id, instance in dataset.id2instance.items():
            for label in list(instance.labels):
                label2count[label] += 1

        mfs = label2count.most_common(1)[0][1]

        return set([mfs])

    def compute_prediction(self, target_vec, candidates):

        if self.average:
            candidates = [(c.first_label, c.context_vec) for c in candidates]
            s2vec = defaultdict(list)
            for label, vec in candidates:
                s2vec[label].append(vec)

            for s in s2vec:
                s2vec[s] = np.average(s2vec[s], 0)

            candidates = list(s2vec.items())
            scores = np.array([1-spatial.distance.cosine(target_vec, vec) for vec in [c[1] for c in candidates]])
            sorted_scores = [("None", x) for x in sorted(scores, reverse=True)]
            best_score_idx = np.argmax(scores)
            pred = set([candidates[best_score_idx][0]])

        else:
            scores = np.array([(c, 1-spatial.distance.cosine(target_vec, c.context_vec)) for  c in candidates])
            sorted_scores = sorted(scores, key=lambda t:t[1],reverse=True)
            if self.k > 1:
                k_closests = [x[0].first_labels for x in sorted_scores[:self.k]]
                pred = Counter(k_closests).most_common(1)[0][0]
            else:
                k_closests = [x[0].labels for x in sorted_scores[:self.k]]
                pred = k_closests[0]
            pred = set(list(pred))

            candidates = [(x[0].id, x[0].context_vec) for x in sorted_scores]

        return pred, candidates, [x[1] for x in sorted_scores]

    def predict(self, dataset):
        assert self.is_fit, 'Model not fitted'

        id2pred = {}
        id2log = defaultdict(lambda: defaultdict(list))

        n_pred = 0
        for instance_id, instance in dataset.id2instance.items():
            key = instance.key
            lemma, pos = instance.key.split('__')
            id = instance.id
            if key not in self.train:
                if self.mfs_backoff:
                    if len(instance.id.split('.')) == 3:
                        mfs_id = instance.source + '.' + id
                    else:
                        mfs_id = instance.id
                    pred = set([self.id2mfs[mfs_id]])
                    candidates = ['MFS']
                    scores = [-1]

                else:
                    pred = None
                    id2pred[id] = None
                    candidates=None
                    scores = None
            else:
                target_vec = instance.context_vec
                #candidates = self.train[key]
                candidates = self.train[key]
                pred, candidates, scores = self.compute_prediction(target_vec, candidates)
                candidates = [c[0] for c in candidates]

            id2pred[id] = pred
            id2log[id] = (instance.key, pos, instance.source, instance.labels, instance.first_label, pred, candidates, scores)
            n_pred += 1

        return id2pred, id2log


    def read_mfs_file(self, mfs_file, fs_file):

        id2mfs = {}

        with open(fs_file) as f:
            for line in f.readlines():
                id, label = line.split()
                id2mfs[id] = label

        return id2mfs
