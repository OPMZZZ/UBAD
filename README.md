# 🧠 UBAD: Unsupervised Brain Anomaly Detection

Official implementation of the paper:  
**Unsupervised Brain Anomaly Detection Using Structure-Preserving Noise Generation and Multi-Scale Dual-Expert Ensembles**  
---

## 🗂️ Project Structure

```bash
UBAD/
├── code/                 # Core training, inference, and data processing scripts
│   ├── train.py          # Main training script
│   ├── infer.py          # Main inference script
│   ├── cross_train.py    # cross_set training
│   ├── cross_infer.py    # cross_set inference
│   └── preprocessing/    # Intra-set data preprocessing code
├── cross_split/          # Cross-set data splits
├── data/                 # Intra-set data splits
├── test_saved_models/    # log
├── requirements.txt      # Python dependencies
```

---

## 🧩 Installation

It is recommended to use Python 3.10+.

```bash
pip install -r requirements.txt
```

Using a virtual environment or conda environment is also recommended.

---

## 📦 Dependencies

- `torch==2.0.1`
- `monai==1.2.0`
- `imgaug==0.4.0`
- `nibabel==5.3.2`
- `scikit_learn==1.3.2`
- `POT==0.9.4` (for Optimal Transport methods)
- `fast_slic==0.4.0` (for superpixel operations)

See [requirements.txt](./requirements.txt) for full list.

---

## 🧪 Usage

### 1. Prepare Dataset

Supported datasets include:

- BRATS2020
- BRATS2021
- MEN
- ISLE
- ATLAS
- IXI (cross_set)
- MSLUB (cross_set)


### 📁 Dataset Preparation

For **intra-set** datasets, use the scripts provided in the [`code/preprocessing/`](./code/preprocessing/) folder to perform dataset splitting and preprocessing.

For **cross-set** datasets, follow the splitting instructions provided in [pDDPM](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD) <!-- Replace `#` with actual URL or relative path -->.

> ⚠️ **Important:** Before splitting, the following datasets require **resampling** and **skull stripping**:
>
> - ISLE  
> - ATLAS  
> - IXI  
> - MSLUB  
>
> Please refer to the implementation in [pDDPM](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD) <!-- Replace `#` with actual URL or relative path --> for details on these preprocessing steps.



### 2. Train

```bash
python code/train.py 
```

### 3. Inference

```bash
python code/infer.py 
```

Cross_set training and validation are supported (`cross_train.py`, `cross_infer.py`).

---


### 🙏 Acknowledgements

This project is built upon and inspired by [DAE](https://github.com/AntanasKascenas/DenoisingAE) and [pDDPM](https://github.com/FinnBehrendt/patched-Diffusion-Models-UAD).  
We sincerely thank the original authors for their valuable contributions to the open-source community.

---
## 📄 License

This project is licensed under the MIT License.
