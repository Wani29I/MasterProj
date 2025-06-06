﻿# MasterProj (ONGOING)
## Overview

This research seeks to develop an effective **strategy** for applying deep learning to UAV-based phenotyping of wheat, specifically the cultivar **“Natsukogane.”** Instead of focusing on a single model or trait, this study systematically explores multiple aspects of deep learning design to identify what works best for trait prediction in real-world agricultural settings.

### Key Research Directions

- **Strategic Focus**: This work is about exploring and identifying the best strategies for deep learning implementation in field-based phenotyping.
  
- **Backbone Comparison**: Multiple model architectures (CNNs, Transformers, Hybrid Models) are compared to evaluate how network design affects prediction across different traits.

- **Extra Input Evaluation**: Assesses how integrating additional non-image features (e.g., Days, SPAD, LAI) impacts model accuracy and reliability.

- **Time Evaluation**: Investigates which growth stages (measurement dates) are most informative for trait prediction, helping optimize when data should be collected.

- **Trait Difficulty Analysis**: Determines which traits can be extracted easily (e.g., Days, Ear Weight) and which are difficult or unreliable (e.g., Leaf Width, Ear Size).

- **Uncertainty Awareness**: Uses confidence-aware models to provide both predictions and uncertainty scores — aiding real decision-making by indicating when to trust model output.

---

## Input Details

- **Data Sources**: UAV-captured RGB imagery (JPG) and DSM maps using DJI Mavic 3
- **DSM**: Used as a separate channel
- **Augmentation**: 10 variants per image
- **Plots**: 115 total, 4 images per plot

---

## Model Architectures

Current models support 0, 1, or 2 extra inputs. All are confidence-aware (Laplace NLL loss) and include internal resizing.

- **EfficientNet**
- **ConvNeXt**
- **RegNet**
- **DenseNet**
- **RepVGG**
- **MobileViT**
- **MobileOne**
- **MaxViT**
- **CoAtNet**
- **FocalNet**
- **EfficientFormer**
- **PoolFormer**
- **Swin Transformer V2**
  
  All models are trained and tested with consistent datasets and evaluated under the same augmentation and loss settings.

---

## 📊 Trait Prediction Results

| Trait                 | Best R²  | MAE      | RMSE     | Result     |
|----------------------|----------|----------|----------|------------|
| Days After Sowing    | 0.9955   | 2.4708   | 3.0210   | Excellent  |
| Total Ear Weight     | 0.9752   | 30.4779  | 43.8531  | Excellent  |
| Height               | 0.8603   | 3.5178   | 4.6638   | Good       |
| Total Ear Count      | 0.8535   | 29.7559  | 38.2463  | Good       |
| SPAD                 | 0.8260   | 3.0380   | 4.1740   | Good       |
| Total Seed Weight    | 0.8220   | 23.2046  | 34.3796  | Good       |
| Total Seed Count     | 0.7650   | 673.3819 | 918.8241 | Good       |
| Straw Weight         | 0.7063   | 1.9822   | 2.5911   | Fair       |
| LAI                  | 0.6021   | 0.4984   | 0.7155   | Fair       |
| Leaf Length          | -0.1942  | 2.3034   | 2.9243   | Bad        |
| Leaf Width           | -5.1105  | 0.3633   | 0.3822   | Bad        |
| Average Ear Size     | -7.7514  | 1.6047   | 1.6903   | Bad        |

---

## Technical Details

- **Input Format**: RGB + DSM + optional extra traits
- **Image Size**: 224×224
- **Loss Function**: Laplace NLL (confidence-aware)
- **Framework**: PyTorch + TIMM
- **Hardware**: Intel i9 14th Gen, NVIDIA RTX 4080 SUPER

---

## Future Work

- Integrate multispectral data
- Improve prediction of poor-performing traits
- Trait interaction modeling
- Calibrate model uncertainty (e.g., ECE, temperature scaling)

---

## Thesis Information

- **Institution**: Tohoku University
- **Graduate School**: Graduate School of Agricultural Science  
- **Graduation**: September 2025  
- **Research Focus**: UAV-based phenotyping, deep learning, smart agriculture

