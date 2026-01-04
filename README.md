# Boosting Low-budget Active Learning with Label Calibration and Unsupervised Representations

**Authors**: Yincheng Han, Xu Li, Jiaxing Shang*, et al.

## Usage
### Installation
```bash
git clone git@github.com/jxshang/KNNLC.git
```
### representation Learning
Some AL algorithms(e.g., ProbCover) relys on unsupervised representation. We train SimCLR with a backbone of ResNet-18 on all of training data of a dataset. An example of CIFAR-10 is as follows:
```bash
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```
When the training is over, the file `./results/cifar-10/pretext/features_seed1.npy` should exist. This file is unsupervised representation applied to querying functions of some AL algorithms. 

Note that  the pretraining phase does not include any test samples.

### active learning
```bash
cd deep-al/tools
python train_al.py --cfg_file ../configs/cifar10/al/RESNET18.yaml --kcalc-start-epoch 100
cd ..
cd ..
```
Some modifiable parameters(e.g., AL algorithms, datasets) are around the main function of `train_al.py`, 
### KNNLC
When active learning is over, the intermediate parameters of the target model is saved at `./output/`. Then you can get the calibrated accuracy of unlabeled pool by runining
```bash
python unlabeled_acc_extract.py
```
and get the calibrated accuracy of test dataset by running
```bash
python test_acc_extract.py
```
The file `calibrated_test_acc_start_epoch_100_KNN_20.npy` are saved at `./output/`

## References
[1] G. Hacohen, A. Dekel, and D. Weinshall, “Active learning on a budget: Opposite strategies suit high and low budgets,” arXiv preprint arXiv:2202.02794, 2022.

## Contact

- **Jiaxing Shang** (E-mail: shangjx@cqu.edu.cn)
- **Yincheng Han** (E-mai: yc.han@foxmail.com)

## License
This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
