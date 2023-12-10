Pytorch implementation of [Graph Contextual Contrasting for Multivariate Time Series Classification]. 

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
