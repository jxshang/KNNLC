import os
import sys
from datetime import datetime
import argparse
import numpy as np
from sklearn.metrics import euclidean_distances

import torch
from copy import deepcopy

# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg_file', default='../configs/{}/al/RESNET18.yaml', help='Config file', type=str)
    parser.add_argument('--exp-name', help='Experiment Name', type=str)
    parser.add_argument('--store_preffix', type=str)
    parser.add_argument('--multi', type=int, default = 1, help = 'budget size is multi times of categories',choices=[1, 5])
    parser.add_argument('--al', type=str)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--seed', help='Random seed', default=2026, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')
    parser.add_argument('--delta', type=float)
    parser.add_argument('--beta', default=0.03, type=float)
    parser.add_argument('--adversarial_eps', help='Relevant only for adversarial ProbCover', default=0.01, type=float)

    # KCALC
    parser.add_argument('--kcalc-start-epoch', help= '用于KCALC, 从 kcalc-start-epoch 开始统计每个epoch下未标注样本集的one-hot分布',
                        type = int, default = 100
    )

    # balque
    parser.add_argument('--cumulative-forgetting-events', type = list, default = None,
                        help= '用于balque, 临时存储从 kcalc-start-epoch 开始统计每个epoch下未标注样本集的累计 one-hot 分布',
    )
    parser.add_argument('--balque-epoch-interval', type = int, default = 1,
                        help= '用于balque, 每间隔 epoch-interval 累计一次 one-hot 变量',
    )
    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):

    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)


    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_dir = cfg.EXP_NAME
    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    else:
        logger.info("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    np.save(f'{cfg.EXP_DIR}/train_set_ground_truth.npy', np.array(train_data.targets, dtype = np.int))
    np.save(f'{cfg.EXP_DIR}/test_set_ground_truth.npy', np.array(test_data.targets, dtype = np.int))
    logger.info("\nDataset {} Loaded Sucessfully.Train Size: {}, Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    model = model_builder.build_model(cfg).cuda()
    if len(lSet) == 0:
        curr_active_learning_method = cfg.ACTIVE_LEARNING.SAMPLING_FN
        if cfg.ACTIVE_LEARNING.SAMPLING_FN not in ['prob_cover', 'typiclust_dc', 'typiclust_rp']:
            cfg.ACTIVE_LEARNING.SAMPLING_FN = 'random'
        al_obj = ActiveLearning(data_obj, cfg)
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
        cfg.ACTIVE_LEARNING.SAMPLING_FN = curr_active_learning_method
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

    logger.info("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TEST.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    # Initialize the model.  
    model = model_builder.build_model(cfg)
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))
    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict())
    model_init_state = deepcopy(model.state_dict().copy())

    logger.info("optimizer: {}\n".format(optimizer))

    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    uncalibrated_test_acc_list = []
    uncalibrated_uSet_acc_list = []
    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir
        cfg.EPISODE = cur_episode

        # Train model
        logger.info("======== TRAINING ========")
        best_val_acc, best_val_epoch, checkpoint_file = train_model(lSet_loader, valSet_loader, model, optimizer, cfg, uSet_loader, uSet)
        logger.info("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
        
        # Test best model checkpoint
        test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode)
        uSet_acc = test_model(uSet_loader, checkpoint_file, cfg, cur_episode)
        uncalibrated_test_acc_list.append(test_acc)
        uncalibrated_uSet_acc_list.append(uSet_acc)
        logger.info("cur_episode:{},Uncalibrated Test Accuracy:{}, Uncalibrated Unlabeled Accuracy : {}".format(
            cur_episode, test_acc, uSet_acc)
        )

        clf_model = model_builder.build_model(cfg)
        clf_model = cu.load_checkpoint(checkpoint_file, clf_model).cuda()
        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # Active Sample 
        logger.info("======== ACTIVE SAMPLING ========\n")
        al_obj = ActiveLearning(data_obj, cfg)
        activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data)

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TEST.BATCH_SIZE, data=train_data)
        
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            logger.info('Starting model from scratch - ignoring existing weights.')
            model = model_builder.build_model(cfg)
            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model)
            logger.info(model.load_state_dict(model_init_state))
            logger.info(optimizer.load_state_dict(opt_init_state))

        # os.remove(checkpoint_file)
    uncalibrated_acc_store_path = cfg.EPISODE_DIR[0:cfg.EPISODE_DIR.rfind('/')]
    np.save(uncalibrated_acc_store_path + '/uncalibrated_test_acc.npy', uncalibrated_test_acc_list)
    np.save(uncalibrated_acc_store_path + '/uncalibrated_unlabeled_acc.npy', uncalibrated_uSet_acc_list)
    logger.info("Uncalibrated Test Accuracy List:{}, Uncalibrated Unlabeled Accuracy List: {}".format(
            uncalibrated_test_acc_list, uncalibrated_uSet_acc_list)
    )

def train_model(train_loader, val_loader, model, optimizer, cfg, unlabeled_loader, uSet):

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))


    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0
    forgetting_events = torch.zeros((len(unlabeled_loader.dataset), cfg.MODEL.NUM_CLASSES))
    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
            cur_epoch, cfg, clf_iter_count, forgetting_events, unlabeled_loader, uSet
        )
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        logger.info('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_", \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)
    logger.info('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))


    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode):


    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)
    

    test_err = test_epoch(test_loader, model, test_meter, cur_episode)
    test_acc = 100. - test_err


    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, forgetting_events, unlabeled_loader, uSet):
    """Performs one epoch of training."""

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    for cur_iter, (inputs, indice, labels) in enumerate(train_loader):
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        loss = loss_fun(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        
        loss, top1_err = loss.item(), top1_err.item()
        if (cur_iter + 1)%20 == 0:
            logger.info('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    if cur_epoch >= cfg.CALIBRAT_EPOCH:
        forgetting_events.zero_()
        model.eval()
        with torch.no_grad():
            for cur_iter, (inputs, indice, labels) in enumerate(unlabeled_loader):
                inputs = inputs.cuda()
                preds = model(inputs)
                pseudo_labels = torch.argmax(preds, dim = 1).cpu()
                forgetting_events[indice, pseudo_labels] = 1
        model.train()
        torch.save(forgetting_events[uSet.tolist()], cfg.EPISODE_DIR + '/forgetting_events_epoch{}.pt'.format(cur_epoch))
        if True:#cfg.ACTIVE_LEARNING.SAMPLING_FN == 'balque':
            if cfg.CUMULATIVE_FORGETTING_EVENTS is None:
                cfg.CUMULATIVE_FORGETTING_EVENTS = forgetting_events.tolist()
            else:
                cfg.CUMULATIVE_FORGETTING_EVENTS = (torch.tensor(cfg.CUMULATIVE_FORGETTING_EVENTS) + forgetting_events).tolist()
        if cur_epoch + 1 == cfg.OPTIM.MAX_EPOCH:
            #print(type(cfg.CUMULATIVE_FORGETTING_EVENTS))
            torch.save(torch.tensor(cfg.CUMULATIVE_FORGETTING_EVENTS)[uSet.tolist()], cfg.EPISODE_DIR + 
                       f'/cumulative_forgetting_events_start_epoch{cfg.CALIBRAT_EPOCH}_interval_epoch{cfg.EPOCH_INTERVAL}.pt'
            )
            cfg.CUMULATIVE_FORGETTING_EVENTS = None
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, indice, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples



if __name__ == "__main__":
    torch.cuda.set_device(0)
    args = argparser().parse_args()
    deltas = {'cifar10':0.75, 'cifar100':0.65, 'tinyimagenet':0.45, 'isic':0.45}
    seed = args.seed
    # these files should exist after the pretraining of CIFAR10, CIFAR100 and tinyimagenet dataset
    pretrain_features = {
        'cifar10':[
            f'../../results/cifar-10/pretext/features_seed{seed}.npy',
            f'../../results/cifar-10/pretext/test_features_seed{seed}.npy'
        ],
        'cifar100':[
            f'../../results/cifar-100/pretext/features_seed{seed}.npy',
            f'../../results/cifar-100/pretext/test_features_seed{seed}.npy',
        ],
        'tinyimagenet':[
            f'../../results/tiny-imagenet/pretext/features_seed{seed}.npy',
            f'../../results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
        ],
        'isic':[
            f'../../results/isic/pretext/features_seed{seed}.npy',
            f'../../results/isic/pretext/test_features_seed{seed}.npy',
        ]
    }

    datasets_methods_repeats =  {
        'cifar10':[5, 5, 5, 5, 5, 5, 5, 5],
        'cifar100':[5, 5, 5, 5, 5, 5, 5, 5],
        'tinyimagenet':[5, 5, 5, 5, 5, 5],
        'isic':[5, 5, 5, 5, 5, 5, 5]
    }

    datasets_methods = {
        'cifar10':['prob_cover', 'random', 'entropy', 'coreset', 'badge', 'typiclust_dc','maxherding','balque'],
        'cifar100':['prob_cover', 'random', 'entropy', 'coreset', 'badge', 'typiclust_dc','maxherding','balque'],
        'tinyimagenet':['prob_cover', 'random', 'entropy', 'coreset', 'typiclust_dc', 'balque'],
        'isic':['prob_cover', 'random', 'entropy', 'coreset', 'badge', 'maxherding','balque'],
    }
    store_preffix = '_{}*categories_repeat'.format(args.multi) + '{}'

    dataset_budgets = {'cifar10':10*args.multi, 'cifar100':100*args.multi, 'tinyimagenet':200*args.multi,'isic':8*args.multi}
    cfg_file = args.cfg_file
    for key in datasets_methods.keys():
        args.cfg_file = cfg_file.format(key)
        args.delta = deltas[key]
        args.budget = dataset_budgets[key]
        
        methods = datasets_methods[key]
        repeat = datasets_methods_repeats[key]

        for method_id in range(len(methods)):
            args.al = methods[method_id]
            cfg.merge_from_file(args.cfg_file)
            cfg.CALIBRAT_EPOCH = args.kcalc_start_epoch
            cfg.CUMULATIVE_FORGETTING_EVENTS = args.cumulative_forgetting_events
            cfg.EPOCH_INTERVAL = args.balque_epoch_interval
            cfg.EPISODE = 0
            cfg.PRETRAIN_UNLABELED_FEATURES_PATH = pretrain_features[key][0]
            cfg.PRETRAIN_TEST_FEATURES_PATH = pretrain_features[key][1]
            cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
            cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
            cfg.ACTIVE_LEARNING.DELTA = args.delta
            cfg.RNG_SEED = args.seed
            cfg.ADVERSARIAL_EPS = args.adversarial_eps
            cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
            for cnt in range(repeat[method_id]):
                args.store_preffix =store_preffix.format(cnt + 1)
                cfg.EXP_NAME = args.al + args.store_preffix
                main(cfg)
