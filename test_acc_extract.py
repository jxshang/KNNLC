import torch
import numpy as np
import os
import argparse
def calibrated_test_acc_by_KCALC(unlabeled_features,
        test_ground_truth,  test_features, forgetting_events_path,
        uset_path, start_epoch, last_epoch, k, test_size, 
        episode_dirs
    ):
    acc_list = []
    for ed in episode_dirs:
        # Load the representations of the unlabeled data extracted from the pretrained model
        uSet = np.load(uset_path.format(ed), allow_pickle=True).astype(np.int)
        pretrained_model_unlabeled_features = unlabeled_features[uSet.tolist()]
        # Pseudo Label Calibration
        forgetting_events = None
        for epoch in range(start_epoch, last_epoch + 1):
            if forgetting_events is None:
                forgetting_events = torch.load(forgetting_events_path.format(ed, epoch)).cuda()
            else:
                forgetting_events += torch.load(forgetting_events_path.format(ed, epoch)).cuda()
        calibrated_labels = torch.argmax(forgetting_events, dim = 1).cpu()

        # KNN-based Prediction Label Calibration
        distance = torch.cdist(test_features, pretrained_model_unlabeled_features).cpu()
        _, min_indice = torch.topk(distance, k, dim = 1, largest=False)
        prediction = torch.zeros(test_size, dtype = torch.long)
        for i in range(prediction.shape[0]):
            prediction[i] = torch.bincount(calibrated_labels[min_indice[i].tolist()]).argmax()
        acc_list.append(float(100*(prediction == test_ground_truth).sum()/test_ground_truth.shape[0]))
    return acc_list

def argparser():
    parser = argparse.ArgumentParser(description='KNNLC')
    parser.add_argument('--dataset', default='CIFAR10', help='dataset name', type=str, choices = ['CIFAR10', 'CIFAR100', 'TINYIMAGENET', 'ISIC'])
    parser.add_argument('--model', default = 'resnet18', type=str)
    parser.add_argument('--start-epoch', default = 100, type=int)
    parser.add_argument('--last-epoch', default = 149, type=int)
    parser.add_argument('--K', default = 20, help='K nearest neighbours', type=int)
    parser.add_argument('--test-size', default = 10000, type=int)
    return parser
if __name__ == '__main__':
    args = argparser().parse_args()
    dataset_name_conversion = {'CIFAR10':'cifar-10', 'CIFAR100':'cifar-100', 'TINYIMAGENET':'tiny-imagenet', 'ISIC':'isic'}
    unlabeled_features_path = f'./results/{dataset_name_conversion[args.dataset]}/pretext/features_seed1.npy'
    test_features_path = f'./results/{dataset_name_conversion[args.dataset]}/pretext/test_features_seed1.npy'
    read_base_path = f'./output/{args.dataset}/{args.model}/'
    # Load the representations of the unlabeled data and test dataset extracted from the pretrained model
    if unlabeled_features_path.endswith('.pt'):
        unlabeled_features = torch.load(unlabeled_features_path).cuda()
        test_features = torch.load(test_features_path).cuda()
    else:
        unlabeled_features = torch.tensor(np.load(unlabeled_features_path)).cuda()
        test_features = torch.tensor(np.load(test_features_path)).cuda()
    AL_methods_path = os.listdir(read_base_path)
    episode_paths =  os.listdir(read_base_path + AL_methods_path[0] + '/')
    episode_dirs = []
    for p in episode_paths:
        if os.path.isdir(read_base_path + AL_methods_path[0] + '/' + p):
            episode_dirs.append(p)
    episode_dirs = sorted(episode_dirs)
    for mp in AL_methods_path:
        test_ground_truth_path = f'{read_base_path}{mp}/test_set_ground_truth.npy'
        test_ground_truth = torch.tensor(np.load(test_ground_truth_path), dtype = torch.long)
        uSet_path = f'{read_base_path}{mp}/' + '{}/uSet.npy'
        forgetting_events_path = f'{read_base_path}{mp}/' + '{}/forgetting_events_epoch{}.pt'
        store_path = f'{read_base_path}{mp}/calibrated_test_acc_start_epoch_{args.start_epoch}_KNN_{args.K}.npy'
        acc_list = calibrated_test_acc_by_KCALC(unlabeled_features, test_ground_truth, test_features, 
                forgetting_events_path, uSet_path, args.start_epoch, args.last_epoch, args.K, args.test_size,
                episode_dirs
        )
        print(f'{mp}:{acc_list}')
        np.save(store_path, np.array(acc_list))