# DFYP

Official implementation of **DFYP: A Dynamic Fusion Framework with Spectral Channel Attention and Adaptive Operator Learning for Crop Yield Prediction**  
Accepted by **IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2026**

Paper: [https://doi.org/10.1109/TGRS.2026.3684831](https://doi.org/10.1109/TGRS.2026.3684831)

## 📝 Abstract

Crop yield prediction is a fundamental task in precision agriculture, requiring effective modeling of complex spatial, temporal, and cross-modal dependencies from remote sensing and environmental observations. In this work, we propose **DFYP**, a dynamic fusion framework for crop yield prediction that integrates spectral channel attention with adaptive operator learning to improve representation quality and prediction robustness. The framework is designed to better capture informative spectral responses, exploit complementary cues across modalities, and adaptively model interactions that vary across regions and growing conditions.

## 📦 Environment Setup

```bash
conda create -n dfyp python=3.9
conda activate dfyp

git clone <YOUR_GITHUB_REPO_URL>
cd DFYP

pip install -r requirements.txt
```

## ⚡ Quick Start

### 1. Download Datasets

Download the datasets and place them under `./datasets/`.

- **MODIS branch (processed dataset)**: We provide a processed dataset that can be used directly at [https://huggingface.co/datasets/onef1shy/Crop-Yield-Prediction-MODIS](https://huggingface.co/datasets/onef1shy/Crop-Yield-Prediction-MODIS). No additional preprocessing is required. The preprocessing pipeline can be referenced from [pycrop-yield-prediction](https://github.com/gabrieltseng/pycrop-yield-prediction.git).
- **Sentinel-2 branch (Tiny-CropNet)**: [https://huggingface.co/datasets/fudong03/Tiny-CropNet/tree/main](https://huggingface.co/datasets/fudong03/Tiny-CropNet/tree/main)

After downloading the Sentinel-2 data, generate the provided json split files with:

```bash
python data/build_sentinel_json.py
```

The `datasets` directory should look like:

```text
datasets/
├── modis/
│   └── processed_data/
│       ├── histogram_all_full.npz
│       ├── 2003_17_1.npy
│       ├── 2003_17_3.npy
│       ├── ...
│       └── 2016_5_95.npy
└── sentinel2/
    ├── Sentinel-2 Imagery/
    ├── USDA/
    ├── WRF-HRRR/
    └── json/
        ├── corn/
        │   ├── corn_train.json
        │   └── corn_test.json
        ├── cotton/
        │   ├── cotton_train.json
        │   └── cotton_test.json
        ├── soybean/
        │   ├── soybean_train.json
        │   └── soybean_test.json
        └── winterwheat/
            ├── winterwheat_train.json
            └── winterwheat_test.json
```

### 2. Download Checkpoints

Download the released checkpoints from Google Drive: [Google Drive](<YOUR_GOOGLE_DRIVE_LINK>), and place them under `./checkpoints/`.

You can then evaluate the released checkpoints directly.

Evaluate a MODIS checkpoint:

```bash
python run.py eval_modis --checkpoint_path=checkpoints/modis/2015.pth --predict_year=2015
```

Evaluate a Sentinel-2 checkpoint:

```bash
python run.py eval_sentinel --checkpoint_path=checkpoints/sentinel2/corn.pth --crop=corn
```

## Retraining Or Changing Operators

If you want to retrain the model or use different operator settings, you can use the following commands.

Train the MODIS branch:

```bash
python run.py train_modis
```

Train the Sentinel-2 branch:

```bash
python run.py train_sentinel
```

Train a single Sentinel-2 crop:

```bash
python run.py train_sentinel --crop=corn
```

## Operator Selection

For the MODIS branch, the operator used for each prediction year is fixed from the stage-1 validation RMSE based selection procedure described in the paper:

- `2009`: `scharr`
- `2010`: `scharr`
- `2011`: `sobel`
- `2012`: `learnable`
- `2013`: `scharr`
- `2014`: `learnable`
- `2015`: `sobel`

If you want to change the MODIS year-specific operators:

```bash
python run.py train_modis \
  --year_operator_map="2009:scharr,2010:scharr,2011:sobel,2012:learnable,2013:scharr,2014:learnable,2015:sobel"
```

For the Sentinel-2 branch, the selected operators for the four crops are:

- `corn`: `sobel`
- `cotton`: `scharr`
- `soybean`: `sobel`
- `winter wheat`: `scharr`


## ✏️ Citation

If you find this repository useful in your research, please cite the paper:

```bibtex
@article{zhang2026dfyp,
  title={DFYP: A Dynamic Fusion Framework with Spectral Channel Attention and Adaptive Operator learning for Crop Yield Prediction},
  author={Zhang, Juli and Yan, Zeyu and Zhang, Jing and Miao, Qiguang and Wang, Quan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```
