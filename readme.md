# Pytorch implementation of [Graph-Aware Contrasting for Multivariate Time-Series Classification](https://arxiv.org/pdf/2309.05202.pdf). 

by: Yucheng Wang, Yuecong Xu, Jianfei Yang, Min Wu, Xiaoli Li, Lihua Xie, and Zhenghua Chen

This work has been accepted for publication for AAAI-2024.

# Abstract
Contrastive learning, as a self-supervised learning paradigm, becomes popular for Multivariate Time-Series (MTS) classification. It ensures the consistency across different views of unlabeled samples and then learns effective representations for these samples. Existing contrastive learning methods mainly focus on achieving temporal consistency with temporal augmentation and contrasting techniques, aiming to preserve temporal patterns against perturbations for MTS data. However, they overlook spatial consistency that requires the stability of individual sensors and their correlations. As MTS data typically originate from multiple sensors, ensuring spatial consistency becomes essential for the overall performance of contrastive learning on MTS data. Thus, we propose Graph-Aware Contrasting for spatial consistency across MTS data. Specifically, we propose graph augmentations including node and edge augmentations to preserve the stability of sensors and their correlations, followed by graph contrasting with both node- and graph-level contrasting to extract robust sensor- and global-level features. We further introduce multi-window temporal contrasting to ensure temporal consistency in the data for each sensor. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance on various MTS classification tasks. The code is available at https://github.com/Frank-Wang-oss/TS-GAC.
![1702894943919](https://github.com/Frank-Wang-oss/TS-GAC/assets/73806631/1f3e686f-0e98-440a-af42-2e446499d6ba)

# Requirements

You will need the following to run the above:
- Pytorch 1.9.1, Torchvision 0.10.1
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run PyTorch on a GPU (cuda, etc)
  
# Dataset

We use six datasets to evaluate our method, including UCI-HAR, ISRUC-S3, three UEA datasets including ArticularyWordRecognition, FingerMovement, and SpokenArabicDigitsEq.


## UCI-HAR

You can access [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

For running the experiments on UCI-HAR, you need to first run preprocess/data_read_HAR.py to pre-process the dataset. After that, run main_GNN_batch.py

## ISRUC-S3
 
You can access [here](https://sleeptight.isr.uc.pt/), and download S3.

For running the experiments on ISRUC, you need to first run preprocess/data_read_ISRUC.py to pre-process the dataset. After that, run main_GNN_batch.py


## UEA Datasets

You can access [here](http://timeseriesclassification.com/dataset.php)


For running the experiments on these datasets, you need to first run preprocess/data_read_UEA.py to pre-process the dataset. After that, run main_GNN_batch.py
