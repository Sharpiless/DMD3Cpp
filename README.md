# DMD³C: Distilling Monocular Foundation Models for Fine-grained Depth Completion

<p align="center">
Official implementation of the CVPR 2025 paper  
<b>"Distilling Monocular Foundation Models for Fine-grained Depth Completion"</b>
</p>

<p align="center">
📄 <a href="https://arxiv.org/abs/2503.16970">Paper (arXiv)</a>
</p>

---

<div align="center">
  <img width="729" src="./assets/image.png" alt="DMD3C Results"/>
</div>

---

## 🔍 Overview

Depth completion methods often suffer in regions with sparse or missing supervision, leading to inaccurate fine-grained structures and degraded depth quality.

**DMD³C** introduces a novel framework that distills rich geometric priors from **monocular foundation models** into the depth completion pipeline. By leveraging dense knowledge from foundation models, DMD³C significantly improves depth estimation quality, particularly in regions lacking ground-truth supervision.

### Key Features

- Distills geometric knowledge from monocular foundation models
- Enhances fine-grained structure recovery
- Improves depth estimation in sparse and unsupervised regions
- Achieves strong performance on benchmark datasets

---

<div align="center">
  <img src="https://github.com/user-attachments/assets/f24eef8e-5dc2-483a-bb70-67671ff5e4e9" width="100%">
</div>

---

## 🚀 Getting Started

### 0. Conda Environment

You can directly build the environment by running the following command if you use conda as the environment management tool.

```bash
conda env create -f environment.yml
```

Compile the C++ and CUDA code:

```
cd exts
python setup.py install
```

### 1. Dataset Preparation

Please follow the dataset preparation instructions from:

👉 [BP-Net](https://github.com/kakaxi314/BP-Net)

The structure of data directory should be:

```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

### 2. Training

Run the training script:

```bash
bash train.sh
```

Our models are trained on 8 GPU workstation with 4 Nvidia GTX 4090 (48G).

### 3. Pretrained Models

Download pretrained checkpoints from:

👉 [Hugging Face](https://huggingface.co/datasets/Liangyingping/DMD3Cpp-checkpoints)

Place the .pth file into "./checkpoints/PMP_Residual_Norm_ssil_KITTI/"

### 4. Submission

Generate predictions and submit results to the KITTI online benchmark:

```bash
bash submission.sh
```

The results will be save into "./results" folder.

---

<div align="center">
  <img width="729" src="./assets/0000000000.png" alt="DMD3C Results"/>
</div>

---



---

## 📊 Results

More experimental results and quantitative comparisons can be found in our paper.

---

## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{liang2025distilling,
  title={Distilling Monocular Foundation Models for Fine-grained Depth Completion},
  author={Liang, Yingping and Hu, Yutao and Shao, Wenqi and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={22254--22265},
  year={2025}
}
```

---

## 🙏 Acknowledgement

This project is built upon and inspired by:

- [BP-Net](https://github.com/kakaxi314/BP-Net)

We sincerely thank the authors for making their code publicly available.