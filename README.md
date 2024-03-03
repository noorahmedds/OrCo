# OrCo: Towards Better Generalization via Orthogonality and Contrast for Few-Shot Class-Incremental Learning

PyTorch implementation of OrCo Framework
[Paper](...)


## Abstract
Few-Shot Class Incremental Learning (FSCIL) introduces a paradigm in which the problem space expands with limited data. FSCIL methods inherently face the challenge of catastrophic forgetting as data arrives incrementally, making models susceptible to overwriting previously acquired knowledge. Moreover, given the scarcity of labeled samples available at any given time, models may be prone to overfitting and find it challenging to strike a balance between extensive pretraining and the limited incremental data. To address these challenges, we propose the OrCo framework built on two core principles: features' orthogonality in the representation space, and contrastive learning. In particular, we improve the generalization of the embedding space by employing a combination of supervised and self-supervised contrastive losses during the pretraining phase. Additionally, we introduce OrCo loss to address challenges arising from data limitations during incremental sessions. Through feature space perturbations and orthogonality between classes, the OrCo loss maximizes margins and reserves space for the following incremental data. This, in turn, ensures the accommodation of incoming classes in the feature space without compromising previously acquired knowledge. Our experimental results showcase state-of-the-art performance across three benchmark datasets, including mini-ImageNet, CIFAR100, and CUB datasets.

<!-- <img src='' width='600' height='400'> -->

<!-- <p align="center">Continually Evolved Classifier</p> -->

## OrCo Framework

<img src='./figures/OrCo-pipeline.jpg'>

## Results

<img src='./figures/sota_fig_all_paper.png'>

## Requirements
- [PyTorch >= version 1.13.1](https://pytorch.org)
- tqdm

## Datasets and pretrained models
We follow [FSCIL](https://github.com/xyutao/fscil) setting and use the same data index_list for training splits across incremental sessions. 
For Phase 1 of the OrCo framework and the datasets CIFAR100 and mini-ImageNet, we use the [solo-learn](https://github.com/vturrisi/solo-learn) library to train our model. You can download the pretrained models [here](?). Place the downloaded models under `./params/OrCo/` and unzip it using:

    $ tar -xvf phase1_params.tar

Note that for CUB200 we do not perform phase 1 given that it is common in literature to use an ImageNet pretrained model for incremental tasks. This is automatically handled from within the code.

## Training

With the current setup, Phase 2 and Phase 3 (Base and Incremental Sessions) of our pipeline are run sequentially with a single command. The final metrics can then be found in the `./logs` under the appropriate datasets. Find the scripts with the best hyperparameters under `./scripts`. 

As an example, to run the mini-ImageNet experiment from the paper:

    $ chmod +x ./scripts/run_minet.sh
    $ ./scripts/run_minet.sh

For the above experiments find the computed metrics available under: `mini_imagenet/orco/<save_path_prefix>_<hp1_choice>-<hp2_choice>/results.txt`

## Citation
If you use the code in this repo for your work, please cite the following bib entries:

<!--
    TODO
-->

## Acknowledgment
Our project references the codes in the following repositories.

- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [solo-learn](https://github.com/vturrisi/solo-learn)
- [fscil](https://github.com/xyutao/fscil)
- [FACT](https://github.com/zhoudw-zdw/CVPR22-Fact)
- [NC-FSCIL](https://github.com/NeuralCollapseApplications/FSCIL)

<!-- Detailed numbers can be found in the [Supplementary Material](https://arxiv.org/pdf/2104.03047)

## Requirements
- [PyTorch >= version 1.1](https://pytorch.org)
- tqdm

## Datasets and pretrained models
We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the downloaded file under `data/` folder and unzip it:
    
    $ tar -xvf miniimagenet.tar 
    $ tar -xvzf CUB_200_2011.tgz

You can find pretrained models from [here](https://drive.google.com/drive/folders/1ZLQJYJ9IkXcVu7v525oCcJGE-zeJAGF_?usp=sharing). Please download the whole params folder and put under your local CEC project path

## Training scripts
cifar100

    $ python train.py -project cec -dataset cifar100 -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1,2,3 -model_dir params/cifar100/session0_max_acc7455_cos.pth

mini_imagenet

    $ python train.py -project cec -dataset mini_imagenet -epochs_base 100 -episode_way 15 -episode_shot 1 -low_way 15 -low_shot 1 -lr_base 0.0002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1,2,3 -model_dir params/miniimagenet/session0_max_acc70367_cos.pth
 
cub200

    $ python train.py -project cec -dataset cub200 -epochs_base 100 -episode_way 15 -episode_shot 1 -episode_query 10 -low_way 15 -low_shot 1 -lr_base 0.0002 -lrg 0.0002 -step 20 -gamma 0.5 -gpu 0,1,2,3 -model_dir params/cub200/session0_max_acc7552_cos.pth

## Pretrain scripts
cifar100

    $python train.py -project base -dataset cifar100  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 60 70 -gpu 0,1,2,3 -temperature 16
    
mini_imagenet

    $python train.py -project base -dataset mini_imagenet -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 40 70 -gpu 0,1,2,3 -temperature 16

cub200
    
    $python train.py -project base -dataset cub200 -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 100 -schedule Milestone -milestones 30 40 60 80 -gpu '0,1,2,3' -temperature 16
## Acknowledgment
Our project references the codes in the following repos.

- [fscil](https://github.com/xyutao/fscil)
- [DeepEMD](https://github.com/icoz69/DeepEMD)
- [FEAT](https://github.com/Sha-Lab/FEAT) -->