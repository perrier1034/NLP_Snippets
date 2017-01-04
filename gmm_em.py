"""
Fit gmm for ./data/faithful.txt
Fitting algorithm is simple gradient descent method
"""
# dependencies
import numpy as np
import pylab as pl


# methods
def output_of_gaussian(x, mean, cov):
    """
    mean and cov is current param of our model which should be fitted.
    So don't use actual mean or cov of `x`.

    :param x: data (a row vector with arbitrary dim)
    :param mean: mean values for each probability variable
    :param cov: Variance-covariance matrix of x
    :return: scalar value

    if x.shape == (n,):
        mean.shape == (n,)
        cov.shape ==  (n, n)

    """
    x_dim = x.shape[-1]

    denominator = ((2 * np.pi) ** (x_dim/2.)) * (np.linalg.det(cov) ** 0.5)
    in_exp = - 0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean)
    return 1 / denominator * np.exp(in_exp)


def output_of_gaussian_mixture(K, n, X, mu, sigma, pi):
    a = 0.
    for k in range(K):
        a += pi[k] * output_of_gaussian(X[n], mu[k], sigma[k])
    return a


def calc_log_likelihood(X, K, mean, cov, pi):
    # X is numpy array with shape (N, vector-dim)
    dst = 0.
    for n in range(len(X)):
        dst += np.log(output_of_gaussian_mixture(K, n, X, mean, cov, pi))
    return dst


def fit_gmm(X, max_epoch=1000, n_prob_var=2, K=2):
    """
    predict mu, sigma of each gaussian and pi by EM algorithm.
    pi is mixing rate of each gaussian (=P(z|x))

    :param X: data
    :param max_epoch:
    :param n_prob_var: how many use probability variable
    :param K: how many gaussian distributions are mixed
    :return:
    """

    N = len(X)

    # init mu, pi, sigma. sigma should be diagonal.
    mu = np.random.normal(size=(K, n_prob_var))
    pi = np.random.normal(size=K)
    sigma = np.asarray([np.eye(n_prob_var, n_prob_var)]*K)

    # initial burden ratios of each gaussian
    gamma = np.zeros(shape=(N, K))

    # initial likelihood
    current_likelihood = calc_log_likelihood(X, K, mu, sigma, pi)

    for epoch in range(max_epoch):
        print epoch, current_likelihood

        # --- step1: gamma calculation using current params (pi, mu, ..)
        for n in range(N):
            for k in range(K):
                gamma[n][k] = pi[k] * output_of_gaussian(X[n], mu[k], sigma[k])
        gamma /= gamma.sum(1, keepdims=True)

        # ---- step2: optimize params for each distribution `k` by gamma
        for k in range(K):
            # Nk: total gamma for `k`
            Nk = np.sum(gamma[:, k], 0)

            # --- UPDATING TIME ---
            # Params below will maximize `expected log-likelihood`
            # we don't calc actual value of it

            # mixing rate
            pi[k] = Nk / N

            # mean
            mu[k] = np.sum(X * gamma[:, None, k], 0) / Nk

            # sigma
            new_sigma = np.zeros((n_prob_var, n_prob_var))
            for n in range(N):
                dev = X[n] - mu[k]
                new_sigma += gamma[n][k] * np.dot(dev[:, None], dev[None, :])
            sigma[k] = new_sigma / Nk

        # ---- check converged or not by calculating log-likelihood with new params
        new_likelihood = calc_log_likelihood(X, K, mu, sigma, pi)
        if new_likelihood - current_likelihood > 0.01:
            current_likelihood = new_likelihood
            continue

        return mu, sigma, pi


if __name__ == "__main__":

    def regularize(data2x2):
        col = data2x2.shape[1]
        # calc mu and sigma for each col
        mean = np.mean(data2x2, axis=0)
        std = np.std(data2x2, axis=0)
        # actual regularization
        for i in range(col):
            data2x2[:, i] = (data2x2[:, i] - mean[i]) / std[i]
        return data2x2

    K = 2
    n_prob_var = 2
    X = regularize(np.genfromtxt("./data/faithful.txt")[:, 0: n_prob_var])

    # fit model
    mu, sigma, pi = fit_gmm(X)

    # print mu
    for k in range(K):
        pl.scatter(mu[k, 0], mu[k, 1], c='r', marker='o')

    # print contour
    x_list = np.linspace(-2.5, 2.5, 50)
    y_list = np.linspace(-2.5, 2.5, 50)
    x, y = np.meshgrid(x_list, y_list)
    for k in range(K):
        z = pl.bivariate_normal(x, y, np.sqrt(sigma[k, 0, 0]), np.sqrt(sigma[k, 1, 1]), mu[k, 0], mu[k, 1], sigma[k, 0, 1])
        pl.contour(x, y, z, 3, colors='k', linewidths=1)

    # print train data
    pl.plot(X[:, 0], X[:, 1], 'gx')
    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)
    pl.show()
