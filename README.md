# Real-time Translation of Upper-body Gestures to Virtual Avatars in Dissimilar Telepresence Environments (Official Implementation)
![Python](https://img.shields.io/badge/Python->=3.8.12-Blue?logo=python)  ![Pytorch](https://img.shields.io/badge/PyTorch->=1.11.0-Red?logo=pytorch)

> **Real-time Translation of Upper-body Gestures to Virtual Avatars in Dissimilar Telepresence Environments**<br>
> Jiho Kang, Taehei Kim, Hyeshim Kim, and [Sung-Hee Lee](https://scholar.google.com/citations?hl=en&user=AVII4wsAAAAJ)<br>
> IEEE TVCG, 2025

## [Dataset](https://www.dropbox.com/scl/fi/u1z2pbewlzuy6ox3s8od6/Dataset.zip?rlkey=wbw3agb3wy37c6ph6ld23dxwq&e=1&st=nwpukgq8&dl=0)
Unzip and move to the main folder.

## Train
To train MPNet for right-handedness, run the following commands.
```bash
python train_MPNet.py --model 'MoE' --handedness 'right' --dataAugmentation True --epochs 80 --batchSize 16
```
To train UGNet for right-handedness, run the following commands.
```bash
python train_UGNet.py --model 'MoE' --handedness 'right' --scheduledSampling True --dataAugmentation False --epochs 120 --c1 30 --c2 60 --batchSize 32
```

## Test
To test MPNet using metric *MPE* on test subject (height: 161cm), run the following commands.
```bash
python test_MPNet.py --subject '161' --model 'MoE' --metric 'MPE'
```
To test UGNet using metric *DIP* on test subject (height: 161cm), run the following commands.
```bash
python test_UGNet.py --subject '161' --model 'MoE' --metric 'DIP'
```

## Acknowledgements
This repository contains pieces of code from the following repository: \
[Interactive Character Control with Auto-Regressive Motion Diffusion Models](https://github.com/Yi-Shi94/AMDM)
