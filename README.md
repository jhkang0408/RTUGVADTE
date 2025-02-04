# Real-time Translation of Upper-body Gestures to Virtual Avatars in Dissimilar Telepresence Environments (IEEE TVCG, Under Review)
![Python](https://img.shields.io/badge/Python->=3.8.12-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.11.0-Red?logo=pytorch)

Official Pytorch implementation of the paper "Real-time Translation of Upper-body Gestures to Virtual Avatars in Dissimilar Telepresence Environments", IEEE TVCG, Under Review

## [Dataset](https://www.dropbox.com/scl/fi/u1z2pbewlzuy6ox3s8od6/Dataset.zip?rlkey=wbw3agb3wy37c6ph6ld23dxwq&e=1&st=nwpukgq8&dl=0)
Unzip and move to the main folder.

## Train
To train MPNet for right-handedness, run the following commands.
```bash
python train_MPNet.py --model 'MoE' --handedness 'right'
```
To train UGNet for right-handedness, run the following commands.
```bash
python train_MPNet.py --model 'MoE' --handedness 'right'
```
## Test
To train human motion manifold networks from the scratch, run the following commands.
```bash
python train.py --config configs/H3.6M.yaml
```

## Acknowledgements
This repository contains pieces of code from the following repository: \
[Interactive Character Control with Auto-Regressive Motion Diffusion Models](https://github.com/Yi-Shi94/AMDM)
