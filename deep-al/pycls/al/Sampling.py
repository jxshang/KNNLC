# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

import numpy as np 
import torch
from statistics import mean
import os
import math
import time
import math
from tqdm import tqdm
from scipy import stats
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, pairwise_distances
from .vaal_util import train_vae_disc
import pycls.utils.logging as lu
import pycls.core.optimizer as optim
import pycls.core.losses as losses
logger = lu.get_logger(__name__)
DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'../../results/cifar-10/pretext/features_seed{}.npy',
            'CIFAR100':'../../results/cifar-10/pretext/scan/results/cifar-100/pretext/features_seed{}.npy',
            'TINYIMAGENET': '../../results/cifar-10/pretext/features_seed{}.npy',
            'IMAGENET50': '../../dino/runs/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/trainfeat.pth',
            'IMAGENET200': '../../dino/runs/trainfeat.pth',
        },
    'test':
        {
            'CIFAR10': '../../results/cifar-10/pretext/test_features_seed{}.npy',
            'CIFAR100': '../../results/cifar-10/pretext/test_features_seed{}.npy',
            'TINYIMAGENET': '../../results/cifar-10/pretext/test_features_seed{}.npy',
            'IMAGENET50': '../../dino/runs/testfeat.pth',
            'IMAGENET100': '../../dino/runs/testfeat.pth',
            'IMAGENET200': '../../dino/runs/testfeat.pth',
        }
}
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    logger.info('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if (len(mu) + 1)%10 == 0:logger.info(str(len(mu)) + '\t' + str(sum(D2)))
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll
class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 


class CoreSetMIPSampling():
    """
    Implements coreset MIP sampling operation
    """
    def __init__(self, cfg, dataObj, isMIP = False):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.isMIP = isMIP

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []
        
        logger.info(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, indice, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        #logger.info(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1).reshape((-1,1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        #order is important
        features = np.vstack((labeled,unlabeled))
        logger.info("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        logger.info("Distance matrix computed in {} seconds".format(end-start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i!=0 and i%500==0:
                logger.info("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices),dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:],axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)
        
        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))

        return greedy_indices-n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)]
        greedy_indices_counter = 0
        #move cpu to gpu
        labeled = torch.from_numpy(labeled).cuda()
        unlabeled = torch.from_numpy(unlabeled).cuda()

        logger.info(f"[GPU] Labeled.shape: {labeled.shape}")
        logger.info(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()
        min_dist,_ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        logger.info(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)
            
            min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices [greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        
        for i in tqdm(range(amount), desc = "Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            
            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))
            
            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices [greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices,remainSet,math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def seperate_unstable_set(self, lSet, uSet, model, dataset):
        num_classes = self.cfg.MODEL.NUM_CLASSES
        clf = model.cuda()
        clf.eval()
        
        lSetLoader = self.dataObj.getSequentialDataLoader(indexes=lSet, batch_size=1, data=dataset)
        fc_optimal_parameters = clf.fc.weight.data.clone()
        optimal_fc_params_copy = fc_optimal_parameters.clone()
        optimal_fc_params_copy.requires_grad_(True)
        optimal_fc_params_bias = clf.fc.bias.clone()
        avg_class_grads = [torch.zeros_like(optimal_fc_params_copy) for i in range(num_classes)]
        train_labels = torch.zeros(lSet.shape[0], dtype = torch.int64);start = 0
        for iter, (inputs, indice, labels) in enumerate(lSetLoader):
            train_labels[start:start + labels.shape[0]] = labels;start += labels.shape[0]
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            penultimate_features, preds = clf(inputs)
            y1 = torch.matmul(penultimate_features, optimal_fc_params_copy.T) + optimal_fc_params_bias
            one_hot_labels = torch.zeros(num_classes, device = 'cuda')
            one_hot_labels[labels[0]] = 1
            y2 = -torch.sum(torch.log(torch.softmax(y1, dim = 1))*one_hot_labels, dim = 1)
            curr_grads = torch.autograd.grad(y2, optimal_fc_params_copy)[0]
            avg_class_grads[labels[0]] += curr_grads
        for lb in range(num_classes):
            avg_class_grads[lb]/=torch.where(train_labels == lb)[0].shape[0]
            avg_class_grads[lb] = torch.abs(avg_class_grads[lb])
            logger.info("avg_class_grads_min:{}, avg_class_grads_max:{}".format(avg_class_grads[lb].min(), avg_class_grads[lb].max()))
        
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        unlabeled_optimal_pseudo_labels = torch.zeros(uSet.shape[0], dtype= torch.int64)
        start = 0
        for iter, (inputs, indice, labels) in enumerate(uSetLoader):
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            _, preds = clf(inputs)
            unlabeled_optimal_pseudo_labels[start:start + labels.shape[0]] = torch.argmax(preds, dim = 1).cpu()
            start +=labels.shape[0]
        
        unlabeled_score = torch.zeros((uSet.shape[0], num_classes), dtype = torch.int64)
        logger.info("filter beta:{}".format(self.cfg.BETA))
        for lb in range(num_classes):
            single_class_uSet_indice = torch.where(unlabeled_optimal_pseudo_labels == lb)[0].numpy()
            single_class_uSet = uSet[single_class_uSet_indice]
            uSetLoader = self.dataObj.getSequentialDataLoader(indexes=single_class_uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
            beta = self.cfg.BETA/avg_class_grads[lb].max()
            for i in range(50):
                clf.fc.weight.data = torch.normal(fc_optimal_parameters, beta*avg_class_grads[lb])
                start = 0
                for iter, (inputs, indice, labels) in enumerate(uSetLoader):
                    inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    _, preds = clf(inputs)
                    pseudo_labels = torch.argmax(preds, dim = 1).cpu()
                    indice = single_class_uSet_indice[start:start + labels.shape[0]]
                    unlabeled_score[indice, pseudo_labels] += 1
                    start +=labels.shape[0]
        unlabeled_score = torch.max(unlabeled_score, dim = 1)[0].numpy()
        stable_uSet = uSet[np.where(unlabeled_score == unlabeled_score.max())[0]].astype(dtype = np.int)
        unstable_uSet = uSet[np.where(unlabeled_score < unlabeled_score.max())[0]].astype(dtype = np.int)
        logger.info('总样本数:{}, 稳定样本数:{}(稳定分数:{}), 不稳定样本数:{}'.format(unlabeled_score.shape[0], stable_uSet.shape[0], unlabeled_score.max(), unstable_uSet.shape[0]))
        return stable_uSet, unstable_uSet
    def query(self, lSet, uSet, clf_model, dataset):

        assert clf_model.training == False, "Classification model expected in training mode"
        assert clf_model.penultimate_active == True,"Classification model is expected in penultimate mode"    
        
        logger.info("Extracting Lset Representations")
        lb_repr = self.get_representation(clf_model=clf_model, idx_set=lSet, dataset=dataset)
        logger.info("Extracting Uset Representations")
        ul_repr = self.get_representation(clf_model=clf_model, idx_set=uSet, dataset=dataset)
        
        logger.info("lb_repr.shape:{}".format(lb_repr.shape))
        logger.info("ul_repr.shape:{}".format(ul_repr.shape))
        
        if self.isMIP == True:
            raise NotImplementedError
        else:
            logger.info("Solving K Center Greedy Approach")
            start = time.time()
            greedy_indexes, remainSet = self.greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            # greedy_indexes, remainSet = self.optimal_greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            end = time.time()
            logger.info("Time taken to solve K center: {} seconds".format(end-start))
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]
        return activeSet, remainSet


class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = 0 if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble") else torch.cuda.current_device()
        self.dataObj = dataObj

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        preds = []
        for i, (x, indice, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)

                temp_pred = clf_model(x)

                #To get probabilities
                temp_pred = torch.nn.functional.softmax(temp_pred,dim=1)
                preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        return preds
    def get_penultimate_embeddings(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        embeddings = torch.zeros((idx_set.shape[0], 512))
        start = 0
        for i, (x, indice, _) in enumerate(tempIdxSetLoader):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)

                embedding, temp_pred = clf_model(x)
                embeddings[start:start + x.shape[0]] = embedding.cpu()
                start += x.shape[0]

        return embeddings

    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.
        
        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet   
        """

        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(uSet, np.ndarray), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(type(uSet))
        assert isinstance(budgetSize,int), "Expected budgetSize of type int whereas provided is dtype:{}".format(type(budgetSize))
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
            .format(len(uSet), budgetSize)

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]
        return activeSet, uSet


    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        clf_model.cuda(self.cuda_id)

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        #Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        n_uPts = len(uSet)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = self.get_predictions(clf_model=clf_model, idx_set=uSet, dataset=dataset)
            
            score_All += dropout_score

            #computing F_x
            dropout_score_log = np.log2(dropout_score+1e-6)#Add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi+1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy

        U_X = G_X - F_X
        logger.info("U_X.shape:{}".format(U_X.shape))
        sorted_idx = np.argsort(U_X)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        clf_model.train()
        return activeSet, remainSet


    def dbal(self, budgetSize, uSet, clf_model, dataset):
        """
        Implements deep bayesian active learning where uncertainty is measured by 
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]
        
        In bayesian view, predictions are computed with the help of dropouts and 
        Monte Carlo approximation 
        """
        clf_model.cuda(self.cuda_id)

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            #logger.info("True")
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        logger.info("len usetLoader: {}".format(len(uSetLoader)))
        temp_i=0
        
        for k,(x_u, indice, _) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1
            x_u = x_u.type(torch.cuda.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS):
                with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    temp_op = clf_model(x_u)
                    # Till here z_op represents logits of p(y|x).
                    # So to get probabilities
                    temp_op = torch.nn.functional.softmax(temp_op,dim=1)
                    z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS
            
            z_op = torch.from_numpy(z_op).cuda(self.cuda_id)
            entropy_z_op = entropy_loss(z_op, applySoftMax=False)
            
            # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
            u_scores.append(entropy_z_op.cpu().numpy())
            ptsProcessed += x_u.shape[0]
            
        u_scores = np.concatenate(u_scores, axis=0)
        sorted_idx = np.argsort(u_scores)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet


    def ensemble_var_R(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ensemble variance_ratio measured as the number of disagreement in committee 
        with respect to the predicted class. 
        If f_m is number of members agreeing to predicted class then 
        variance ratio(var_r) is evaludated as follows:
        
            var_r = 1 - (f_m / T); where T is number of commitee members

        For more details refer equation 4 in 
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf
        """
        from scipy import stats
        T = len(clf_models)

        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        logger.info("len usetLoader: {}".format(len(uSetLoader)))

        temp_i=0
        var_r_scores = np.zeros((len(uSet),1), dtype=float)
        
        for k, (x_u, indice, _) in enumerate(tqdm(uSetLoader, desc="uSet Forward Passes through "+str(T)+" models")):
            x_u = x_u.type(torch.cuda.FloatTensor)
            ens_preds = np.zeros((x_u.shape[0], T), dtype=float)
            for i in range(len(clf_models)):
               with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    temp_op = clf_models[i](x_u)
                    _, temp_pred = torch.max(temp_op, 1)
                    temp_pred = temp_pred.cpu().numpy()
                    ens_preds[:,i] = temp_pred
            _, mode_cnt = stats.mode(ens_preds, 1)
            temp_varr = 1.0 - (mode_cnt / T*1.0)
            var_r_scores[temp_i : temp_i+x_u.shape[0]] = temp_varr

            temp_i = temp_i + x_u.shape[0]

        var_r_scores = np.squeeze(np.array(var_r_scores))
        logger.info("var_r_scores:{}".format(var_r_scores.shape))

        sorted_idx = np.argsort(var_r_scores)[::-1] #argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet

    def uncertainty(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)
        
        clf = model.cuda()
        
        u_ranks = []

        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),data=dataset)
        
        n_uLoader = len(uSetLoader)
        logger.info("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, indice, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda()

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
                temp_u_rank = 1 - temp_u_rank
                u_ranks.append(temp_u_rank.detach().cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        logger.info(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet


    def entropy(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)

        n_uLoader = len(uSetLoader)
        logger.info("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, indice, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda()

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1*torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        logger.info(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet


    def margin(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        
        n_uLoader = len(uSetLoader)
        logger.info("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, indice, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda()

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]
                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value  
                difference = -1*difference 
                u_ranks.append(difference.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        logger.info(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] #argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        return activeSet, remainSet
    
    def badge(self, budgetSize, lSet, uSet, model, dataset):
        uSet = uSet.astype(int)
        clf = model.cuda();clf.eval()
        embedding_dim = 512
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        embedding = np.zeros([len(dataset), embedding_dim * self.cfg.MODEL.NUM_CLASSES])
        with torch.no_grad():
            for iter, (datas, indice, labels) in enumerate(uSetLoader):
                datas = datas.cuda()
                penultimate_features, preds = clf(datas)
                penultimate_features = penultimate_features.cpu().numpy()
                batchProbs = torch.softmax(preds, dim=1).cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(indice)):
                    for c in range(self.cfg.MODEL.NUM_CLASSES):
                        if c == maxInds[j]:
                            embedding[indice[j]][embedding_dim * c : embedding_dim * (c+1)] = penultimate_features[j] * (1 - batchProbs[j][c])
                        else:
                            embedding[indice[j]][embedding_dim * c : embedding_dim * (c+1)] = penultimate_features[j] * (-1 * batchProbs[j][c])
        uSetEmebedding = embedding[uSet]
        chosen = init_centers(uSetEmebedding, budgetSize)
        activeSet = uSet[chosen]
        uSet[chosen] = -1
        uSet = uSet[np.where(uSet != -1)[0]]
        logger.info('activeSet size:{}, uSet size:{}'.format(activeSet.shape[0], uSet.shape[0]))
        return activeSet, uSet

    def maxherding(self, budgetSize, lSet, uSet, model, dataset):
        def gaussian_kernel(x, y, sigma = 1.0):
            x_norm = (x ** 2).sum(dim=1, keepdim=True) 
            y_norm = (y ** 2).sum(dim=1).unsqueeze(0)
            dist_sq = x_norm + y_norm - 2 * (x @ y.t())
            return torch.exp(-dist_sq / (2* sigma**2))
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        lSetLoader = self.dataObj.getSequentialDataLoader(indexes=lSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        L_all = []
        for l_data, _, _ in lSetLoader:
             L_all.append(l_data.view(l_data.size(0), -1))
        L_all = torch.cat(L_all, dim=0).cuda()
        U_all = []
        U_index_map = []
        for u_data, idx, _ in uSetLoader:
             U_index_map.extend(idx.tolist())
             U_all.append(u_data.view(u_data.size(0), -1))
        U_all = torch.cat(U_all, dim=0).cuda()  
        N_L = L_all.size(0)
        N_U = U_all.size(0)  
        batch = self.cfg.TRAIN.BATCH_SIZE
        with torch.no_grad():
            k_list = []
            for i in range(0, N_U, batch):
                U_batch = U_all[i:i+batch]
                K = gaussian_kernel(U_batch, L_all)  # B_U, N_L
                k_list.append(K.max(dim=1).values) # B_U
                del K
            k_vals = torch.cat(k_list, dim=0) # N_U 
            del k_list
        selected_indices = []    
        for b in range(budgetSize):
            # gains = torch.zeros(N_U, device=U_all.device)
            # for i in range(0, N_U, batch):
            #     U_cand = U_all[i:i+batch]      # Bc × D
            #     gain_block = torch.zeros(U_cand.size(0), device=U_all.device)
            #     for j in range(0, N_U, batch):
            #         U_ref = U_all[j:j+batch]   # Br × D
            #         K = gaussian_kernel(U_ref, U_cand)   # Br × Bc
            #         delta = torch.clamp(K - k_vals[j:j+batch].unsqueeze(1), min=0)
            #         gain_block += delta.mean(dim=0)
            #         del K, delta
            #         torch.cuda.empty_cache()
            #     gains[i:i+batch] = gain_block
            #     del gain_block
            gain_list = []
            for i in range(0, N_U, batch):
                U_cand = U_all[i:i+batch]
                K = gaussian_kernel(U_all, U_cand)  # N_U, Bu
                delta = torch.clamp(K - k_vals.unsqueeze(1), min=0) # N_U, B_U
                gain = delta.mean(dim=0)   # B_U
                gain_list.append(gain)
                del K, delta
            gains = torch.cat(gain_list, dim=0) 
            del gain_list
            best = gains.argmax().item()
            selected_indices.append(best)
            x_star = U_all[best].unsqueeze(0)     # [1,D]
            for i in range(0, N_U, batch):
                U_batch = U_all[i:i+batch]
                K_star = gaussian_kernel(U_batch, x_star).squeeze()  # [N_U]
                k_vals[i:i+batch] = torch.maximum(k_vals[i:i+batch], K_star)
                del K_star
                torch.cuda.empty_cache()
        chosen = np.array(selected_indices)
        activeSet = uSet[chosen]
        uSet[chosen] = -1
        uSet = uSet[np.where(uSet != -1)[0]]
        logger.info('activeSet size:{}, uSet size:{}'.format(activeSet.shape[0], uSet.shape[0]))
        return activeSet, uSet


    def balque(self, budgetSize, lSet, uSet, model, dataset):
        '''
        通过 unlabeled pool 在训练过程中的累计预测 one-hot 分布来计算未标注样本的信息量，
        信息量是通过 累计 one-hot 分布的信息熵计算得到
        '''
        uSet_forgetting_events = torch.load(self.cfg.EPISODE_DIR + 
            f'/cumulative_forgetting_events_start_epoch{self.cfg.CALIBRAT_EPOCH}_interval_epoch{self.cfg.EPOCH_INTERVAL}.pt'
        )
        accumulate_cnt = int(uSet_forgetting_events[0].sum())
        uSet_forgetting_events /= accumulate_cnt
        uSet_forgetting_events += 1e-9
        uSet_entropy = torch.mul(torch.log(uSet_forgetting_events), uSet_forgetting_events).sum(dim = 1)
        _, chosen = torch.topk(uSet_entropy, k = budgetSize, largest = False)
        chosen = chosen.numpy()
        activeSet = uSet[chosen]
        uSet[chosen] = -1
        uSet = uSet[np.where(uSet != -1)[0]]
        logger.info('activeSet size:{}, uSet size:{}'.format(activeSet.shape[0], uSet.shape[0]))
        return activeSet, uSet
    
class AdversarySampler:


    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.dataObj = dataObj
        self.budget = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        self.cuda_id = torch.cuda.current_device()
        if cfg.DATASET.NAME == 'TINYIMAGENET':
            cfg.VAAL.Z_DIM = 64
            cfg.VAAL.IM_SIZE = 64
        else:
            cfg.VAAL.Z_DIM = 32
            cfg.VAAL.IM_SIZE = 32


    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        return dists

    def vaal_perform_training(self, lSet, uSet, dataset, debug=False):
        oldmode = self.dataObj.eval_mode
        self.dataObj.eval_mode = True
        self.dataObj.eval_mode = oldmode

        # First train vae and disc
        vae, disc = train_vae_disc(self.cfg, lSet, uSet, dataset, self.dataObj, debug)
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS) \
            ,data=dataset)

        # Do active sampling
        vae.eval()
        disc.eval()

        return vae, disc, uSetLoader

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = []
    
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(self.compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        temp_range = 1000
        for j in range(1, labeled.shape[0], temp_range):
            if j + temp_range < labeled.shape[0]:
                dist = self.compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                # for last iteration only :)
                dist = self.compute_dists(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        for i in range(amount):
            if i!=0 and i%500 == 0:
                logger.info("{} Sampled out of {}".format(i, amount+1))
            dist = self.compute_dists(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        return greedy_indices, remainSet


    def get_vae_activations(self, vae, dataLoader):
        acts = []
        vae.eval()
        
        temp_max_iter = len(dataLoader)
        logger.info("len(dataloader): {}".format(temp_max_iter))
        temp_iter = 0
        for x,y in dataLoader:
            x = x.type(torch.cuda.FloatTensor)
            x = x.cuda(self.cuda_id)
            _, _, mu, _ = vae(x)
            acts.append(mu.cpu().numpy())
            if temp_iter%100 == 0:
                logger.info(f"Iteration [{temp_iter}/{temp_max_iter}] Done!!")

            temp_iter += 1
        
        acts = np.concatenate(acts, axis=0)
        return acts


    def get_predictions(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds = all_preds.cpu().numpy()
        return all_preds


    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists


    def efficient_compute_dists(self, labeled, unlabeled):
        """
        """
        N_L = labeled.shape[0]
        N_U = unlabeled.shape[0]
        dist_matrix = None

        temp_range = 1000

        unlabeled = torch.from_numpy(unlabeled).cuda(self.cuda_id)
        temp_dist_matrix = np.empty((N_U, temp_range))
        for i in tqdm(range(0, N_L, temp_range), desc="Computing Distance Matrix"):
            end_index = i+temp_range if i+temp_range < N_L else N_L
            temp_labeled = labeled[i:end_index, :]
            temp_labeled = torch.from_numpy(temp_labeled).cuda(self.cuda_id)
            temp_dist_matrix = self.gpu_compute_dists(unlabeled, temp_labeled)
            temp_dist_matrix = torch.min(temp_dist_matrix, dim=1)[0]
            temp_dist_matrix = torch.reshape(temp_dist_matrix,(temp_dist_matrix.shape[0],1))
            if dist_matrix is None:
                dist_matrix = temp_dist_matrix
            else:
                dist_matrix = torch.cat((dist_matrix, temp_dist_matrix), dim=1)
                dist_matrix = torch.min(dist_matrix, dim=1)[0]
                dist_matrix = torch.reshape(dist_matrix,(dist_matrix.shape[0],1))
        
        return dist_matrix.cpu().numpy()


    @torch.no_grad()
    def vae_sample_for_labeling(self, vae, uSet, lSet, unlabeled_dataloader, lSetLoader):
        
        vae.eval()
        logger.info("Computing activattions for uset....")
        u_scores = self.get_vae_activations(vae, unlabeled_dataloader)
        logger.info("Computing activattions for lset....")
        l_scores = self.get_vae_activations(vae, lSetLoader)
        
        logger.info("l_scores.shape:{}".format(l_scores.shape))
        logger.info("u_scores.shape:{}".format(u_scores.shape))
        
        dist_matrix = self.efficient_compute_dists(l_scores, u_scores)
        logger.info("Dist_matrix.shape:{}".format(dist_matrix.shape))

        min_scores = np.min(dist_matrix, axis=1)
        sorted_idx = np.argsort(min_scores)[::-1]

        activeSet = uSet[sorted_idx[0:self.budget]]
        remainSet = uSet[sorted_idx[self.budget:]]

        return activeSet, remainSet


    def sample_vaal_plus(self, vae, disc_task, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert disc_task.training == False, "Expected disc_task model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds,_ = disc_task(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices)," Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet,uSet


    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        vae.cuda(self.cuda_id)
        discriminator.cuda(self.cuda_id)

        temp_idx = 0
        for images,_ in data:
            images = images.type(torch.cuda.FloatTensor)
            images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices), " Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet


    @torch.no_grad()
    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, uSet):
        """
        Picks samples from uSet to form activeSet.

        INPUT
        ------
        vae: object of model VAE

        discriminator: object of model discriminator

        unlabeled_dataloader: Sequential dataloader iterating over uSet

        uSet: Collection of unlabelled datapoints

        NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
        results won't be appropriate.

        OUTPUT
        -------

        Returns activeSet, [remaining]uSet
        """
        activeSet, remainSet = self.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        return activeSet, remainSet

