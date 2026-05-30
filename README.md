# MedIQA: Scalable Prompt-Driven Lightweight Model for Cross-Modality and Cross-Organ Medical Image Quality Assessment

MedIQA is a universal model for medical image quality assessment (IQA), designed to overcome modality heterogeneity and data limitations in clinical settings. Leveraging a novel prompt-driven architecture and cross-modal learning.

This repository provides the PyTorch implementation for our work. The framework introduces:

- Cross-Modality: unified quality assessment for CT, MRI, and fundus
- Lightweight Design: fewer parameters than conventional vision foundation models
- Prompt-Driven Adaptation: easy transfer to unseen IQA tasks via domain-aware prompts
- Multi-Scale Saliency: automatic identification of diagnostically critical slices
- Dual Supervision: combines physical parameter learning with expert annotation fine-tuning


## Installation
1. Clone the repository
```bash
git clone https://github.com/siyi-xun/MedIQA
cd MedIQA
```
2. Create a conda environment
```bash
conda create -n MedIQA python=3.9
conda activate MedIQA
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```
## Generate prompt information
For Generate image classification prompt information, run:
```bash
python generate_info.py
```

## Training
For training and validation, run:
```bash
python MedIQA_train.py
```

## Testing
For test, run:
```bash
python MedIQA_test.py
```

## Pretrained Weights and Models

You can download the pre-trained weights and models from the [release](https://github.com/siyi-xun/MedIQA/releases/tag/V1.0.0) pages.


## Acknowledgements
This repository is built upon the [MANIQA](https://github.com/IIGROUP/MANIQA) and [timm](https://github.com/huggingface/pytorch-image-models) codebase. We thank the authors for making their work publicly available.
