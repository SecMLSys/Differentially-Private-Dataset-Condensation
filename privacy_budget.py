import torch
from privacy_analysis import *
from utils import get_dataset


def non_linear_dpdc_budget(dataset, data_path, group_size=50, iterations=10000, tilde_sigma=1):

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, data_path)

    indices_class = [[] for c in range(num_classes)]

    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 2000))

    qc = [group_size/len(indices_class[c]) for c in range(num_classes)]
    q = max(qc)

    rdps = [iterations * compute_omega(q, tilde_sigma, alpha) for alpha in alphas]

    eps, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)

    return '({}, {})-DP'.format(eps, 1e-5)


def linear_dpdc_budget(dataset, data_path, group_size=50, M=50, tilde_sigma=1):

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, data_path)

    indices_class = [[] for c in range(num_classes)]

    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)

    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 2000))

    qc = [group_size/len(indices_class[c]) for c in range(num_classes)]
    q = max(qc)

    rdps = [M * compute_omega(q, tilde_sigma, alpha) for alpha in alphas]

    eps, order = get_privacy_spent(orders=alphas, rdp=rdps, delta=1e-5)

    return '({}, {})-DP'.format(eps, 1e-5)

if __name__ == '__main__':

    print('Linear DPDC MNIST Budget: {}'.format(linear_dpdc_budget('MNIST', './data')))

    print('Linear DPDC FMNIST Budget: {}'.format(linear_dpdc_budget('FashionMNIST', './data')))

    print('Linear DPDC CIFAR10 Budget: {}'.format(linear_dpdc_budget('CIFAR10', './data')))

    print('Linear DPDC CelebA Budget: {}'.format(linear_dpdc_budget('CelebA', './data')))
    
    print('Non-Linear DPDC MNIST Budget: {}'.format(non_linear_dpdc_budget('MNIST', './data')))

    print('Non-Linear DPDC FMNIST Budget: {}'.format(non_linear_dpdc_budget('FashionMNIST', './data')))

    print('Non-Linear DPDC CIFAR10 Budget: {}'.format(non_linear_dpdc_budget('CIFAR10', './data')))

    print('Non-Linear DPDC CelebA Budget: {}'.format(non_linear_dpdc_budget('CelebA', './data')))
