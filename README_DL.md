# TODO
1. 可视化分析&实验


# Project
我们的实验fix on UBFC数据集，关注下面关于UBFC数据集的说明

config:configs/UBFC_TSCAN_*.yaml

运行方式 python main.py, 修改设置参考scripts内的注释

安装环境缺什么装什么就可以，我是python3.8（requirement.txt中的版本号可以无视，装最新的即可）
# Setup

STEP2: `conda activate rppg-toolbox` 

STEP3: `pip install -r requirements.txt` 


# Training on SCAMPS with DeepPhys

STEP1: Download the SCAMPS via this [link](https://github.com/danmcduff/scampsdataset) and split it into train/val/test folders.

STEP2: Modify `./configs/SYNTHETICS_DEEPPHYS_BASIC.yaml` 

STEP2: Run `python main_neural_method.py --config_file ./configs/SYNTHETICS_DEEPPHYS_BASIC.yaml` 

Note: Preprocessing requires only once, thus turn it off on the yaml file when you train the network after the first time. 

# Dataset
The toolbox supports four datasets, which are SCAMPS, UBFC, PURE and COHFACE. Cite corresponding papers when using.
For now, we only recommend training with PURE or SCAMPS due to the level of synchronization and volume of the dataset.
* [SCAMPS](https://arxiv.org/abs/2206.04197)
  
    * D. Mcudff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", Arxiv, 2022
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
            |-- P00003.mat
         |...
    -----------------

* [UBFC](https://sites.google.com/view/ybenezeth/ubfcrppg)
  
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/UBFC/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------
* [COHFACE](https://www.idiap.ch/en/dataset/cohface)
    * Guillaume Heusch, André Anjos, Sébastien Marcel, “A reproducible study on remote heart rate measurement”, arXiv, 2016.
    * In order to use this dataset in a deep model, you should organize the files as follows:
    -----------------
         data/COHFACE/
         |   |-- 1/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
         |...
         |   |-- n/
         |      |-- 0/
         |          |-- data.avi
         |          |-- data.hdf5
         |      |...
         |      |-- 3/
         |          |-- data.avi
         |          |-- data.hdf5
    -----------------
    
* [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    * In order to use this dataset in a deep model, you should organize the files as follows:
    
    -----------------
        data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------

## Add A New Dataloader

* Step1 : Create a new python file in dataset/data_loader, e.g. MyLoader.py

* Step2 : Implement the required functions, including:

  ```python
  def preprocess_dataset(self, config_preprocess)
  ```
  ```python
  @staticmethod
  def read_video(video_file)
  ```
  ```python
  @staticmethod
  def read_wave(bvp_file):
  ```

* Step3 :[Optional] Override optional functions. In principle, all functions in BaseLoader can be override, but we **do not** recommend you to override *\_\_len\_\_, \_\_get\_item\_\_,save,load*.
* Step4 :Set or add configuration parameters.  To set paramteters, create new yaml files in configs/ .  Adding parameters requires modifying config.py, adding new parameters' definition and initial values.
