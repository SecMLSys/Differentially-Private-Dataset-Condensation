import torch
from privacy_analysis import *
from utils import get_dataset

def ldpdc_sigma(dataset, data_path, group_size=50, M=50, target_epsilon=1):

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, data_path)

    indices_class = [[] for c in range(num_classes)]

    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 2000))

    qc = [group_size/len(indices_class[c]) for c in range(num_classes)]
    q = max(qc)

    sigma_low, sigma_high = 0, 10
    rdps = [M * compute_omega(q, sigma_high, alpha) for alpha in alphas]
    eps_high, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)
    epsilon_tolerance = 0.01

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        rdps = [M *  compute_omega(q, sigma, alpha) for alpha in alphas]
        eps, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def ndpdc_sigma(dataset, data_path, group_size=50, iterations=10000, target_epsilon=1):

    channel, im_size, num_classes, class_names, mean, std, \
        dst_train, dst_test, testloader = get_dataset(dataset, data_path)

    indices_class = [[] for c in range(num_classes)]

    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 2000))

    qc = [group_size/len(indices_class[c]) for c in range(num_classes)]
    q = max(qc)

    sigma_low, sigma_high = 0, 10
    rdps = [iterations * compute_omega(q, sigma_high, alpha) for alpha in alphas]
    eps_high, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)
    epsilon_tolerance = 0.01

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        rdps = [iterations * compute_omega(q, sigma, alpha) for alpha in alphas]
        eps, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high




if __name__ == '__main__':
    ## epsilon = 1
    print('Tilde Sigmas (sigma/sqrt(d)) for LDPDC with (1, 10^{-5})-DP budget')
    print('LDPDC MNIST Tilde Sigma: {}'.format(ldpdc_sigma('MNIST', './data')))
    print('LDPDC FMNIST Tilde Sigma: {}'.format(ldpdc_sigma('FashionMNIST', './data')))
    print('LDPDC CIFAR10 Tilde Sigma: {}'.format(ldpdc_sigma('CIFAR10', './data')))
    print('LDPDC CelebA Tilde Sigma: {}'.format(ldpdc_sigma('CelebA', './data')))

    print('Sigmas for NDPDC with (1, 10^{-5})-DP budget')
    print('NDPDC MNIST Sigma: {}'.format(ndpdc_sigma('MNIST', './data')))
    print('NDPDC FMNIST Sigma: {}'.format(ndpdc_sigma('FashionMNIST', './data')))
    print('NDPDC CIFAR10 Sigma: {}'.format(ndpdc_sigma('CIFAR10', './data')))
    print('NDPDC CelebA Sigma: {}'.format(ndpdc_sigma('CelebA', './data')))
