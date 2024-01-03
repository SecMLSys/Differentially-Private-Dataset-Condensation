import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import (get_dataset, get_network, get_eval_pool,
                    evaluate_synset, get_time, DiffAugment,
                    ParamDiffAug, poisson_sampling)
import setGPU

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=10000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    
    parser.add_argument('--group_size', type=int, default=50, help='group size L')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='ndpdc_result', help='path to save results')

    parser.add_argument('--max_norm', type=float, default=1, help='maximum norm of the representations')
    parser.add_argument('--sigma', type=float, default=1.0, help='noise multiplier')
    args = parser.parse_args()
    args.method = 'DM'
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' or args.eval_mode == 'B' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)



        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n, indices_class): # get random n images from class c
            num_samples = len(indices_class[c])
            sample_rate = float(n)/float(num_samples)
            sample_indices = poisson_sampling(num_samples, sample_rate)
            idx_shuffle = np.random.permutation(indices_class[c])[sample_indices]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor(np.array([np.ones(args.ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_exp%d_iter%d.png'%('ndpdc', args.dataset, args.model, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                vis_num_per_class = min(10, args.ipc)
                vis_indices = np.array([np.arange(j * args.ipc, vis_num_per_class + j * args.ipc) for j in range(num_classes)]).flatten()
                save_image(image_syn_vis[vis_indices, :], save_name, nrow=vis_num_per_class) # Trying normalize = True/False may get better visual effects.


            if it < args.Iteration:
                ''' Train synthetic data '''
                net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
                net.train()
                for param in list(net.parameters()):
                    param.requires_grad = False

                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
                
                loss_avg = 0

                ''' update synthetic data '''
                if 'BN' not in args.model: # for ConvNet
                    loss = torch.tensor(0.0).to(args.device)
                    for c in range(num_classes):
                        img_real = get_images(c, args.group_size, indices_class)
                        img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

                        if args.dsa:
                            seed = int(time.time() * 1000) % 100000
                            img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                            img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                        output_real = embed(img_real).detach()
                        l2_real = torch.norm(output_real, dim=1).detach()
                        scaler = torch.min(args.max_norm * torch.ones_like(l2_real) / l2_real,
                                        torch.ones_like(l2_real))
                        # output_real_noise = scaler.unsqueeze(1) * output_real + args.sigma * torch.randn_like(output_real)
                        output_real_sum = torch.sum(scaler.unsqueeze(1) * output_real, dim=0)
                        output_real_sum_noise = output_real_sum + args.sigma * torch.randn_like(output_real_sum)
                        
                        output_syn = embed(img_syn)
                        l2_syn = torch.norm(output_syn, dim=1).detach() ## disable grad
                        syn_scaler = torch.min(args.max_norm * torch.ones_like(l2_syn) / l2_syn,
                                    torch.ones_like(l2_syn))
                        output_syn = syn_scaler.unsqueeze(1) * output_syn

                        balanced_factor = float(args.group_size)/float(args.ipc)
                        loss += torch.sum((output_real_sum_noise - balanced_factor * torch.sum(output_syn, dim=0))**2)
                        
                else: # for ConvNetBN
                    raise SystemError(" BN can not be used in DPDC ")

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()


                loss_avg /= (num_classes)

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


