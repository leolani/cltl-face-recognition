import logging
import uuid
from typing import List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from cltl.vector_id.api import VectorIdentity

logger = logging.getLogger(__name__)


class ClusterIdentity(VectorIdentity):
    @classmethod
    def agglomerative(cls, ndim: int, distance_threshold):
        ac = AgglomerativeClustering(n_clusters=None,
                                affinity='cosine',
                                linkage='average',
                                distance_threshold=distance_threshold)

        return cls(ac, ndim)

    def __init__(self, clustering, ndim: int):
        self._clustering = clustering
        self._centroids = dict()
        self._representations = np.empty((0, ndim)) if ndim else None

    def add(self, representations: np.ndarray) -> List[str]:
        if self._representations is None:
            self._representations = np.atleast_2d(representations)
        else:
            self._representations = np.vstack([self._representations, representations])

        if len(self._representations) == 1:
            id = str(uuid.uuid4())
            self._centroids = {id: normalize(self._representations)[0]}

            return [id]

        cluster_ids = self._cluster_representations(self._centroids, self._representations)

        id_map = {id: idx for idx, id in enumerate(set(cluster_ids))}
        labeled_representations = np.hstack([np.atleast_2d([id_map[id] for id in cluster_ids]).T,
                                             normalize(self._representations)])
        self._centroids = {id: self._centroid(labeled_representations, label) for id, label in id_map.items()}

        return cluster_ids[-len(representations):]

    def _centroid(self, labeled_representations: np.array, label: int):
        mask = (labeled_representations[:, 0] == label)

        return labeled_representations[mask][:, 1:].mean(axis=0)

    def _cluster_representations(self, centroids, representations):
        logger.debug(f"Clustering ids")

        if len(representations) == 0:
            return []

        if len(centroids):
            ids, id_representations = zip(*centroids.items())
            all = np.vstack([id_representations, representations])
        else:
            ids, id_representations, all = (), (), representations

        clustering = self._clustering.fit(all)

        label_ids = {label: id for id, label in zip(ids, clustering.labels_[:len(ids)])}
        label_ids = {label: label_ids[label] if label in label_ids else str(uuid.uuid4())
                     for label in np.unique(clustering.labels_)}

        return [label_ids[label] for label in clustering.labels_[len(ids):]]


