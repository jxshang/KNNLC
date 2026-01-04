import torch
import numpy as np
import os
import argparse
def calibrate_unlabeled_acc_by_KCALC(train_set_ground_truth, uSet_path, forgetting_events_path, 
            start_epoch, last_epoch, episode_dirs
        ):
    acc_list = []
    for ed in episode_dirs:
        uSet = np.load(uSet_path.format(ed), allow_pickle=True).astype(np.int)
        uSet_truth = train_set_ground_truth[uSet]
        forgetting_events = None
        for epoch in range(start_epoch, last_epoch + 1):
            if forgetting_events is None:
                forgetting_events = torch.load(forgetting_events_path.format(ed, epoch)).cuda()
            else:
                forgetting_events += torch.load(forgetting_events_path.format(ed, epoch)).cuda()
        calibrated_labels = torch.argmax(forgetting_events, dim = 1).cpu().numpy()
        acc_list.append(100*(calibrated_labels == uSet_truth).sum()/uSet_truth.shape[0])
    return acc_list

def argparser():
    parser = argparse.ArgumentParser(description='KNNLC')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset name', type=str, choices = ['CIFAR10', 'CIFAR100', 'TINYIMAGENET', 'ISIC'])
    parser.add_argument('--model', default = 'resnet18', type=str)
    parser.add_argument('--start-epoch', default = 100, type=int)
    parser.add_argument('--last-epoch', default = 149, type=int)
    return parser
if __name__ == '__main__':
    args = argparser().parse_args()
    read_base_path = f'./output/{args.dataset}/{args.model}/'
    AL_methods_path = os.listdir(read_base_path)
    episode_paths =  os.listdir(read_base_path + AL_methods_path[0] + '/')
    episode_dirs = []
    for p in episode_paths:
        if os.path.isdir(read_base_path + AL_methods_path[0] + '/' + p):
            episode_dirs.append(p)
    episode_dirs = sorted(episode_dirs)
    for mp in AL_methods_path:
        train_set_ground_truth_path = f'{read_base_path}{mp}/train_set_ground_truth.npy'
        train_set_ground_truth = np.load(train_set_ground_truth_path).astype(np.int)
        uSet_path = f'{read_base_path}{mp}/' + '{}/uSet.npy'
        forgetting_events_path = f'{read_base_path}{mp}/' + '{}/forgetting_events_epoch{}.pt'
        store_path = f'{read_base_path}{mp}/calibrated_unlabeled_acc_start_epoch_{args.start_epoch}.npy'
        acc_list = calibrate_unlabeled_acc_by_KCALC(train_set_ground_truth, uSet_path, forgetting_events_path, 
            args.start_epoch, args.last_epoch, episode_dirs
        )
        print(f'{mp}:{acc_list}')
        np.save(store_path, np.array(acc_list))