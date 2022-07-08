## UCGAN

---

This is the official implementation of ["Unsupervised Cycle-consistent Generative Adversarial Networks for Pan-sharpening".](https://arxiv.org/abs/2109.09395)
The paper is accepted to TGRS2022.

## Overview of UCGAN
![image](src/overview.png)

## Architecture of UCGAN
![image](src/architecture.png)

### Requirements
This environment is mainly based on python=3.6 with CUDA=10.2

```shell
conda create -n panformer python=3.6
conda activate ucgan
conda install pytorch=1.7.1 torchvision=0.2.2 cudatoolkit=10.2
pip install mmcv==1.2.7
conda install gdal=3.1.0 -c conda-forge
conda install scikit-image=0.17.2
pip install scipy==1.5.3
pip install gpustat==0.6.0
pip install numba==0.53.1 
pip install einops==0.3.0 
pip install timm==0.3.2
pip install sewar==0.4.4
```

### Test with the pretrained Model
Due to the large size of the dataset, we only provide some samples in './data' to verify the code.

```shell
conda activate ucgan
export CUDA_VISIBLE_DEVICES='0';
python main.py -c configs/ucgan.py
```

You can modify the config file 'configs/ucgan.py' for different purposes.

# Citing UCGAN
Consider cite UCGAN in your publications if it helps your research.

```
@article{zhou2022unsupervised,
  title={Unsupervised Cycle-Consistent Generative Adversarial Networks for Pan Sharpening},
  author={Zhou, Huanyu and Liu, Qingjie and Weng, Dawei and Wang, Yunhong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--14},
  year={2022},
  publisher={IEEE}
}
```
