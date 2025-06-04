# Driver Drowsiness Detection using Vision Transformers (ViT)

## Project Description
This project implements a real-time driver drowsiness detection system using Vision Transformers (ViT). The system analyzes facial features (eye closure, yawning, head pose) through a camera feed to detect signs of fatigue and alerts the driver. Compared to traditional CNN-based approaches, our ViT model achieves higher accuracy (97.52%) by leveraging self-attention mechanisms to capture global facial feature relationships.

## Key Features
- **Vision Transformer Architecture**: Uses patch-based attention for robust feature extraction
- **Real-time Processing**: 12.3ms inference latency on mid-range GPUs (GTX 1650)
- **Multi-Indicator Detection**: 
  - Eye Aspect Ratio (EAR) analysis
  - Yawning detection via mouth aspect ratio
  - Head pose estimation
- **Alert System**: Visual/Auditory alerts when drowsiness is detected
- **Robust Performance**: 94.3% accuracy under occlusions (sunglasses/masks)

## Dataset
The model was trained on the **NTHU-DDD Dataset**:
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/banudeep/nthuddd2)

Dataset contains:
- 66,521 labeled images (36,030 drowsy / 30,491 non-drowsy)
- Variations in lighting, head pose, and occlusions
- Balanced class distribution

## Technical Specifications
| Component              | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| Model Architecture     | ViT-Base (12 transformer layers, 8 attention heads)                    |
| Training Framework     | PyTorch with Hugging Face Transformers                                  |
| Inference Speed        | 55.42 samples/sec (GTX 1650)                                           |
| Key Metrics            | Accuracy: 97.52%, Precision: 97.6%, Recall: 97.4%, F1-Score: 97.49%    |
| Input Resolution       | 224×224 pixels (divided into 16×16 patches)                            |
