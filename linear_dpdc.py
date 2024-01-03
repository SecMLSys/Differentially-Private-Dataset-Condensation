import os
import copy
import torch
import argparse
import setGPU
import numpy as np
import torch.nn as nn
from torchvision.utils import save_image
from utils import (evaluate_synset, get_dataset,
                get_network, ParamDiffAug,
                get_daparam, get_eval_pool, poisson_sampling)

def get_parser():
    """ parser of evaluation """
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--method', type=str, default='sum', help='the method of adding noise')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode')

    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--group_size', type=int, default=50, help='group size L')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--sigma', type=float, default=1.0, help='tilde sigma')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')

    parser.add_argument('--save_path', type=str, default='ldpdc_result', help='path to save results')
    
    args = parser.parse_args()

    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.dsa_param = ParamDiffAug()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def main():
    """ Evaluation """

    args = get_parser()

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    ## hyperparameters
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    
    args.sigma = args.sigma * np.sqrt(im_size[0] * im_size[1] * channel)
    
    ## original train data
    train_loader = torch.utils.data.DataLoader(dst_train,
                batch_size=1, shuffle=True, num_workers=0)
    
    ## for evaluation
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    accs = []
    for exp in range(args.num_exp):
        ## store synthesized data
        syn_data, syn_labels = [], []
        
        for c in range(num_classes):
            ## collect data with label c
            class_data = []
            for data in train_loader:
                if data[1].item() == c:
                    ## flatten the data sample
                    class_data.append(data[0].unsqueeze(0).view(1, -1))
            class_data = torch.cat(class_data, dim=0)

            sample_rate = args.group_size / class_data.size(0)

            for i in range(args.ipc):
                sample_indice = poisson_sampling(class_data.size(0), sample_rate)
                sampled_data = class_data[sample_indice].clone().detach()
                
                if args.method == 'mean':
                    syn_sample = torch.mean(sampled_data, dim=0)
                    noise = torch.randn_like(syn_sample) * args.sigma
                    syn_sample = syn_sample + noise
                elif args.method == 'sum':
                    syn_sample = torch.sum(sampled_data, dim=0)
                    noise = torch.randn_like(syn_sample) * args.sigma
                    syn_sample = 1/float(sample_rate * class_data.size(0)) * (syn_sample + noise)

                ## reform the dimension
                syn_data.append(syn_sample.unsqueeze(0).view(1, channel,
                                                    im_size[0], im_size[1]))
                syn_labels.append(torch.tensor(c).long().unsqueeze(0))
        
        syn_data = torch.cat(syn_data, dim=0)
        syn_labels = torch.cat(syn_labels)

        save_name = os.path.join(args.save_path, 'vis_%s_%s_exp%d.png'%('ldpdc', args.dataset, exp))
        image_syn_vis = copy.deepcopy(syn_data.detach().cpu())
        for ch in range(channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0
        vis_num_per_class = min(10, args.ipc)
        vis_indices = np.array([np.arange(j * args.ipc, vis_num_per_class + j * args.ipc) for j in range(num_classes)]).flatten()
        save_image(image_syn_vis[vis_indices, :], save_name, nrow=vis_num_per_class) # Trying normalize = True/False may get better visual effects.


        for model_eval in model_eval_pool:

            print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s'%(args.model, model_eval))

            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
            _, acc_train, acc_test = evaluate_synset(0, net_eval, syn_data, syn_labels, testloader, args)

            accs_all_exps[model_eval].append(acc_test)
    
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()