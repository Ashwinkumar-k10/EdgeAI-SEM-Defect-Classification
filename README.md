# EdgeAI SEM Defect Classification

<div align="center">

[![IESA Hackathon Finalist](https://img.shields.io/badge/IESA_AI_Hackathon-Top_Finalist-blueviolet.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![TFLite](https://img.shields.io/badge/TFLite-Quantized_INT8-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Target Hardware](https://img.shields.io/badge/Target-NXP_eIQ_%7C_MCU_%7C_Edge-0085CA.svg)](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ)
[![Compression](https://img.shields.io/badge/Model_Compression-92.8%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An ultra-lightweight, quantized deep learning pipeline for real-time automated semiconductor wafer defect classification on edge devices and microcontrollers.**

**Top Finalist Solution in the IESA AI Hackathon**

[Overview](#overview) • [Key Features](#key-features) • [Architecture](#system-architecture) • [Defect Taxonomy](#defect-taxonomy) • [Performance Benchmarks](#performance-benchmarks) • [Quickstart](#quickstart) • [Phase-wise Workflow](#phase-wise-workflow) • [Edge Deployment](#edge-deployment--hardware-compatibility)

</div>

---

## Overview

Semiconductor manufacturing requires nanometer-scale precision. Scanning Electron Microscope (SEM) imaging is the gold standard for inspecting semiconductor wafer topography and identifying structural anomalies. However, traditional manual review or heavy server-side computer vision systems suffer from:

- **Latency Bottlenecks:** Offloading massive volumes of high-resolution SEM imagery over network infrastructure introduces severe throughput constraints.
- **Compute Constraints:** Industrial inline inspection systems operate under strict thermal, memory, and energy budgets.
- **Human Error:** Manual defect triage is subjective, non-scalable, and inconsistent across wafer runs.

This project was developed as a **Top Finalist** solution for the **IESA AI Hackathon** (organized by the India Electronics and Semiconductor Association in collaboration with I4C). It provides an **end-to-end Edge AI framework** for high-accuracy, low-latency SEM defect triage. By combining lightweight neural architectures (MobileNetV3) with post-training INT8 / dynamic range quantization, the system compresses models by up to **92.8%** (down to **725.6 KB**) while preserving high macro F1-scores, enabling deterministic on-device inference on microcontrollers (MCUs) and edge accelerators (e.g., NXP eIQ ecosystem).

---

## Key Features

- **Edge-Optimized Backbone:** Built upon lightweight convolutional architectures (MobileNetV3-Small) optimized for low Multiply-Accumulate (MAC) operations.
- **Extreme Model Compression:** 
  - ONNX INT8: **1.67 MB** (~3.5x compression over FP32)
  - Dynamic Range TFLite: **725.6 KB** (**92.8% reduction** vs. SavedModel format)
- **Multi-Class Defect Discrimination:** Accurate classification across 10-11 critical wafer defect categories (including line edge roughness, particle contamination, CMP scratches, via anomalies, and bridging).
- **Multi-Runtime Export:** Supports direct inference and conversion across **PyTorch (.pt)**, **ONNX (.onnx)**, **TensorFlow (.h5 / SavedModel)**, and **TensorFlow Lite (.tflite)**.
- **Robust Domain Shift Handling:** Evaluated under challenging zero-retraining cross-distribution benchmark datasets.
- **Production-Ready Tooling:** Complete suite of scripts for automated data augmentation, class rebalancing, INT8 quantization, test set evaluation, and headless video/image inference demonstrations.

---

## System Architecture

```text
                                 [ Raw SEM Image (128x128) ]
                                              │
                                              ▼
                                 ┌─────────────────────────┐
                                 │   Data Preprocessing   │
                                 │  - Normalization (0-1)  │
                                 │  - Channel Replication  │
                                 └────────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │                                                 │
                     ▼                                                 ▼
        [ Training & Optimization ]                       [ Edge Deployment Target ]
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │  MobileNetV3 Backbone   │                       │  TensorFlow Lite Runtime│
        │  - Depthwise Convolutions│                      │  - Dynamic INT8 Model   │
        │  - Squeeze-and-Excitation│                      │  - Footprint: 725.6 KB  │
        └────────────┬────────────┘                       └────────────▲────────────┘
                     │                                                 │
                     ▼                                                 │
        ┌─────────────────────────┐                       ┌────────────┴────────────┐
        │ Model Export & Quantize ├──────────────────────►│    NXP eIQ / MCU Target │
        │ - PyTorch -> ONNX INT8  │   (Edge Optimization) │    On-Device Inference  │
        │ - TF -> TFLite Dynamic  │                       └────────────┬────────────┘
        └─────────────────────────┘                                    │
                                                                       ▼
                                                          ┌─────────────────────────┐
                                                          │ Defect Classification   │
                                                          │ - Class Prediction      │
                                                          │ - Confidence Scores     │
                                                          │ - Metrics & Logs        │
                                                          └─────────────────────────┘
```

---

## Defect Taxonomy

The pipeline is trained to recognize critical anomalies occurring across photolithography, etching, chemical-mechanical planarization (CMP), and deposition stages:

| Defect Class | Category Code | Description & Industrial Impact |
|:---|:---:|:---|
| **Clean / Clean Crack / Clean Via** | `clean*` | Nominal wafer surface, defect-free baseline topography. |
| **Bridge** | `bridge` | Unwanted conductive interconnect between neighboring metal lines; causes electrical shorts. |
| **CMP** | `cmp` | Surface polishing defects, slurry residues, or dishing from Chemical Mechanical Planarization. |
| **Crack** | `crack` | Physical substrate fractures or dielectric cracking induced by thermal/mechanical stress. |
| **Line Edge Roughness (LER)** | `ler` | Deviations from ideal line edge profiles; affects gate length uniformity and transistor switching. |
| **Open** | `open` | Incomplete metal line continuity; results in open circuits and signal failure. |
| **Particle** | `particle` | Airborne or equipment-borne particulate contaminants on wafer surface. |
| **Scratch** | `scratch` | Mechanical surface abrasions from handling or wafer transport tools. |
| **Vias** | `vias` | Structural defects or misalignments in vertical interlayer interconnects. |
| **Other** | `other` | Rare or uncharacterized anomalous surface patterns outside primary target categories. |

---

## Performance Benchmarks

### Phase 1: Base Model & Edge Quantization (10 Classes)

Evaluated on the Phase 1 test split with balanced defect classes:

| Model Variant | Format | Model Size | Accuracy | Compression | Target Runtime |
|:---|:---:|:---:|:---:|:---:|:---|
| **MobileNetV3 FP32** | PyTorch / ONNX | 5.84 MB | **96.37%** | Baseline | ONNX Runtime / GPU / CPU |
| **MobileNetV3 INT8** | ONNX Quantized | **1.67 MB** | Hardware-Calibrated | **3.5x** | NXP eIQ / Edge NPU |

---

### Phase 2: Blind Domain Shift Evaluation (Zero-Retraining)

Rigorous evaluation on the independent `hackathon_test_dataset` with strict competition compliance (no retraining, zero model parameter modification):

| Metric | Result | Benchmark Context |
|:---|:---:|:---|
| **Evaluation Model** | Phase 1 FP32 ONNX (5.84 MB) | Zero retraining / Out-of-the-box evaluation |
| **Overall Accuracy** | **36.82%** | Real-world cross-domain test under unseen noise profiles |
| **Weighted Precision** | **39.04%** | Unmatched classes appropriately mapped to `other` |
| **Weighted Recall** | **36.82%** | Cross-domain generalization baseline |

---

### Phase 3: Final Multi-Class MCU Optimization (11 Classes)

Targeted training and dynamic range quantization on the expanded Phase 3 benchmark dataset:

| Model Architecture | Format | Size | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Size Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline Full Precision** | H5 / SavedModel | 7.46 MB (9.87 MB SM) | **84.36%** | **85.11%** | **84.36%** | **84.40%** | Baseline |
| **Dynamic TFLite (PRIMARY)** | `.tflite` | **725.6 KB** | **80.64%** | **81.75%** | **80.64%** | **80.61%** | **92.8%** |

#### Per-Class Detailed Breakdown (Phase 3 TFLite vs. Float32):

```text
========================================================================================================
 Class Name          Float32 Precision / Recall / F1          Dynamic TFLite Precision / Recall / F1
--------------------------------------------------------------------------------------------------------
 BRIDGE              0.78  /  0.72  /  0.75                   0.76  /  0.61  /  0.68
 CLEAN_CRACK         0.97  /  0.99  /  0.98                   0.95  /  0.98  /  0.97
 CLEAN_LAYER         0.72  /  0.76  /  0.74                   0.64  /  0.69  /  0.66
 CLEAN_VIA           0.86  /  0.98  /  0.92                   0.88  /  0.94  /  0.91
 CMP                 0.78  /  0.88  /  0.83                   0.73  /  0.90  /  0.81
 CRACK               0.96  /  0.94  /  0.95                   0.94  /  0.93  /  0.93
 LER                 0.67  /  0.85  /  0.75                   0.61  /  0.84  /  0.71
 OPEN                0.91  /  0.81  /  0.86                   0.86  /  0.72  /  0.78
 OTHERS              0.88  /  0.71  /  0.78                   0.79  /  0.75  /  0.77
 PARTICLE            0.89  /  0.78  /  0.83                   0.86  /  0.63  /  0.73
 VIA                 0.95  /  0.86  /  0.90                   0.97  /  0.88  /  0.92
========================================================================================================
 Macro Average       0.85  /  0.84  /  0.84                   0.82  /  0.81  /  0.81
========================================================================================================
```

---

## Repository Structure

```text
EdgeAI-SEM-Defect-Classification/
├── dataset_info/
│   └── README.md                         # Detailed dataset specifications & split schema
├── outputs/
│   ├── phase1/                           # Phase 1 evaluation metrics, CM plots, logs
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── test_accuracy.txt
│   ├── phase2/                           # Phase 2 domain shift evaluation logs & CM
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   ├── metrics.txt
│   │   └── prediction_log.txt
│   └── phase3/                           # Phase 3 submission logs, TFLite metrics & CM
│       ├── Hackathon_phase3_prediction_result.txt
│       ├── classification_report_dynamic.txt
│       ├── classification_report_float32.txt
│       ├── confusion_matrix_dynamic.png
│       ├── confusion_matrix_float32.png
│       ├── phase3_task2_final_outcome.txt
│       └── prediction_log.txt
├── src/
│   ├── augment_dataset.py                # Targeted data augmentation & balancing
│   ├── class_map.json                    # Class name to integer ID mappings
│   ├── evaluate_submission.py            # Automated multi-metric evaluator
│   ├── evaluate_test_sem_and_save.py     # Phase 1 PyTorch test set evaluation
│   ├── export_to_onnx_sem.py             # PyTorch to ONNX conversion utility
│   ├── finetune_phase3.py                # Phase 3 transfer learning & fine-tuning
│   ├── hackathon_phase3_prediction_code.py # Phase 3 submission inference pipeline
│   ├── hackathon_test_dataset_prediction.py # Phase 2 blind inference runner
│   ├── inference_onnx_video_demo.py      # Real-time visual/video inference demo
│   ├── quantize_and_eval.py              # TFLite Dynamic & INT8 quantization suite
│   ├── quantize_onnx_int8_sem.py         # ONNX Runtime INT8 quantization
│   ├── train_mcu.py                      # TensorFlow/Keras training for MCU models
│   └── train_mobilenetv3_sem.py          # PyTorch MobileNetV3 training script
├── requirements.txt                      # Project dependencies
└── README.md                             # Project documentation
```

---

## Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/Ashwinkumar-k10/EdgeAI-SEM-Defect-Classification.git
cd EdgeAI-SEM-Defect-Classification
```

### 2. Set Up Virtual Environment
```bash
# Using conda
conda create -n edgeai-sem python=3.10 -y
conda activate edgeai-sem

# Or using standard venv
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Phase-wise Workflow

### Phase 1: Base Model Training & ONNX Quantization

1. **Train MobileNetV3-Small (PyTorch):**
   ```bash
   python src/train_mobilenetv3_sem.py
   ```
2. **Evaluate Model & Generate Confusion Matrix:**
   ```bash
   python src/evaluate_test_sem_and_save.py
   ```
3. **Export Trained PyTorch Model to ONNX:**
   ```bash
   python src/export_to_onnx_sem.py
   ```
4. **Quantize ONNX Model to INT8:**
   ```bash
   python src/quantize_onnx_int8_sem.py
   ```
5. **Run Visual Inference Demonstration:**
   ```bash
   python src/inference_onnx_video_demo.py
   ```

---

### Phase 2: Cross-Domain Evaluation (Zero Retraining)

Evaluate the Phase 1 ONNX model against the domain-shifted test dataset:
```bash
python src/hackathon_test_dataset_prediction.py
```
*Outputs are saved to `outputs/phase2/`.*

---

### Phase 3: Multi-Class Optimization & TFLite Deployment

1. **Perform Class-Balanced Data Augmentation:**
   ```bash
   python src/augment_dataset.py
   ```
2. **Train Base MCU Model:**
   ```bash
   python src/train_mcu.py
   ```
3. **Fine-tune on Target Phase 3 Distribution:**
   ```bash
   python src/finetune_phase3.py
   ```
4. **Quantize to Dynamic TFLite & Validate:**
   ```bash
   python src/quantize_and_eval.py
   ```
5. **Run Comprehensive Submission Evaluation:**
   ```bash
   python src/evaluate_submission.py
   ```
6. **Generate Final Submission Prediction Logs:**
   ```bash
   python src/hackathon_phase3_prediction_code.py
   ```
*Outputs and summary reports are saved to `outputs/phase3/`.*

---

## Edge Deployment & Hardware Compatibility

The quantized artifacts generated by this pipeline are designed for heterogeneous edge platforms:

| Platform Family | Target Hardware | Preferred Format | Toolchain / Runtime |
|:---|:---|:---:|:---|
| **NXP i.MX RT Series** | i.MX RT1060 / RT1170 (Cortex-M7 / M4) | `.tflite` (Dynamic / INT8) | [NXP eIQ ML Software](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ) / TF Lite for Microcontrollers |
| **NXP i.MX 8M Series** | i.MX 8M Plus (NPU / Cortex-A53) | `.onnx` / `.tflite` | eIQ with ONNX Runtime / TIM-VX / OpenVINO |
| **Edge Gateways / IPCs** | ARM Cortex-A series, Intel x86 | `.onnx` | ONNX Runtime (CPU / OpenVINO / TensorRT) |
| **Embedded MCUs** | STM32 / ESP32 / RP2040 | `.tflite` | TensorFlow Lite for Microcontrollers (TFLM) |

### Deployment Specifications:
- **Input Dimension:** 128 x 128 x 3 (RGB) or 128 x 128 x 1 (Grayscale replicated across channels)
- **Pixel Normalization:** Scale to [0.0, 1.0] or ImageNet standard (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
- **Target Memory Budget:** Fits easily within < 1 MB flash memory budget (**725.6 KB**).

---

## Key Results Summary

<div align="center">

| Phase | Framework | Model Size | Accuracy | Primary Optimization |
|:---:|:---:|:---:|:---:|:---|
| **Phase 1** | PyTorch / ONNX | 1.67 MB | 96.37% | Post-Training ONNX INT8 Quantization |
| **Phase 2** | ONNX Runtime | 5.84 MB | 36.82% | Cross-Domain Blind Robustness Analysis |
| **Phase 3** | TensorFlow / TFLite | **725.6 KB** | **80.64%** | **Dynamic Range TFLite Quantization (92.8% reduction)** |

</div>

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## Contributors & Acknowledgments

- **Author / Lead Developer:** [Hireshkumaran G](https://github.com/Hireshkumaran-G)
- **Hackathon Recognition:** **Top Finalist** at the IESA AI Hackathon (organized by the India Electronics and Semiconductor Association & I4C).
- Special thanks to the open-source communities behind PyTorch, TensorFlow Lite, ONNX Runtime, and the NXP eIQ Development Environment.