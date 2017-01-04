import copy
import numpy as np


def nearest_cluster_id(vector, center_vectors):
    sub = center_vectors - vector
    distances = np.sqrt(np.sum(np.power(sub, 2), axis=1))
    return np.argmin(distances)


def do_clustering(samples, n_cluster, epoch=1000):
    """
    perform K-means.

    :param samples: samples with shape: (num_of_sample, dim_of_vector)
    :param n_cluster: e.g. 3
    :param epoch: iter
    :return: center of each cluster
    """
    n_samples = samples.shape[0]
    sample_dim = samples.shape[1]

    # randomize `cluster-id`s of samples
    cluster_ids = np.random.randint(0, n_cluster-1, size=n_samples)
    # init center of gravity with zeros
    centers = np.zeros(shape=(n_cluster, sample_dim))
    # initial value is dummy
    cluster_ids_prev = np.empty((1,))

    for _ in range(epoch):
        # update center of clusters
        for cluster_id in range(n_cluster):
            indices = np.where(cluster_ids == cluster_id)
            centers[cluster_id] += np.sum(samples[indices], axis=0)
            centers[cluster_id] /= len(indices)

        # update each sample's cluster-id to nearest one
        for i in range(n_samples):
            cluster_ids[i] = nearest_cluster_id(samples[i], centers)

        # whether iteration is suffice or not
        if all(cluster_ids_prev == cluster_ids):
            break

        cluster_ids_prev = copy.deepcopy(cluster_ids)

    return centers


def main():
    sample_dim = 2
    n_sample = 100
    n_cluster = 5

    # toy data
    samples = np.random.uniform(-1., 1., size=(n_sample, sample_dim))
    new_vector = np.random.uniform(-1., 1., size=(sample_dim,))

    # perform k-means
    centers = do_clustering(samples, n_cluster)

    # test
    print 'cluster id of {}: {} (of {})'.format(new_vector,
                                                nearest_cluster_id(new_vector, centers),
                                                n_cluster)

if __name__ == '__main__':
    main()
