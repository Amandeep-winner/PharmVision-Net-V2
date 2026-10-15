# Handwritten Prescription Recognition Using OCR and Deep Learning

A deep learning mini-project that tries to read handwritten medicine names from doctors' prescriptions.
It compares two approaches on the same labelled data:

1. **An OCR pipeline:** OpenCV text-region detection, Tesseract OCR, fuzzy matching against a 78-medicine vocabulary, and rule-based extraction of drug, dosage, and frequency.
2. **CNN image classifiers:** ResNet-50 and DenseNet-121 (ImageNet transfer learning) and a small CNN trained from scratch with an ElasticNet penalty, each predicting one of 78 medicine names from a word image.

All results are shown in an interactive HTML dashboard generated from the executed notebook.

> **Educational project.**
> The models are far from accurate enough for real use.
> Predictions must never be used to dispense medicine without a pharmacist checking them.

---

## Table of Contents

- [Results at a Glance](#results-at-a-glance)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [How It Works](#how-it-works)
- [Models](#models)
- [Techniques and Algorithms](#techniques-and-algorithms)
- [How to Run the Notebook](#how-to-run-the-notebook)
- [Running the Dashboard](#running-the-dashboard)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)
- [References](#references)

---

## Results at a Glance

All numbers come from the recorded notebook run, on the held-out test set of 780 images (78 classes, 10 per class).

| Approach | Test ROC-AUC (macro, OvR) | Test Accuracy | Test Macro F1 |
|---|---|---|---|
| **ResNet-50** (pretrained, head only) | **0.72** | **8%** | **0.059** |
| DenseNet-121 (pretrained, head only) | 0.65 | 5% | 0.031 |
| ElasticNet CNN (from scratch) | 0.56 | 2% | 0.003 |
| Tesseract OCR + fuzzy match | n/a | 3.1% | n/a |
| Random guessing (1 / 78) | 0.50 | 1.3% | n/a |

On the unlabelled full prescriptions (Dataset B), the OCR pipeline matched a vocabulary medicine in **3 of 60** images (5% hit rate).

**Key findings:**

- Transfer learning clearly helps: ResNet-50 beats the from-scratch CNN even though only its final layer was trained.
- Tesseract is built for printed text and returns mostly noise on cursive handwriting.
- Training and validation curves were still improving after 3 epochs, so the models are **underfit**, not overfit.
  Longer training and unfreezing the backbone are the obvious next steps.

---

## Project Structure

```
PV2/
├── ocr-handwritten-prescription.ipynb      # Main notebook: OCR pipeline + CNN training (run on Kaggle)
├── dashboard/
│   ├── index.html                          # Interactive results dashboard (static, no server needed)
│   ├── data.js                             # Metrics extracted from the notebook (generated)
│   ├── build_data.py                       # Regenerates data.js and img/ from the executed notebook
│   └── img/                                # Figures extracted from the notebook (generated)
├── Handwritten_Prescription_Report.docx    # Project report
└── Handwritten_Prescription_Presentation.pptx  # Project presentation
```

---

## Datasets

Both datasets are on Kaggle.
The notebook finds them automatically anywhere under `/kaggle/input`.

### Dataset A: OCR Processed Handwritten Prescriptions

Kaggle input folder: `ocr-processed-handwritten-prescriptions`

- **What it is:** cropped images of single handwritten medicine names, each labelled with its medicine.
- **Classes:** 78 medicine names.
- **Splits (predefined, used unchanged):**

  | Split | Images | Per class |
  |---|---|---|
  | Train | 3,120 | 40 |
  | Validation | 780 | 10 |
  | Test | 780 | 10 |

- **Layout:**

  ```
  data/processed/
  ├── train/images/   val/images/   test/images/
  ├── train_labels.csv  val_labels.csv  test_labels.csv   # columns: IMAGE, MEDICINE_NAME (class id)
  └── mapping.json                                         # medicine name -> class id
  ```

- **Used for:** training and evaluating the CNNs, and measuring OCR-match accuracy against ground truth.

### Dataset B: Illegible Medical Prescription Images

Kaggle input folder: `illegible-medical-prescription-images-dataset`

- **What it is:** 129 photos or scans of complete handwritten prescriptions, **without labels**.
- **Used for:** running the full OCR pipeline on realistic inputs (region detection, OCR, entity extraction).
  The first 60 images are processed.
  Because there are no labels, only a hit rate can be measured, not accuracy.

---

## How It Works

```
                       ┌──────────────────────────────┐
                       │  Kaggle datasets A and B     │
                       │  (auto-detected, validated)  │
                       └──────────────┬───────────────┘
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
     ┌───────────────────────┐                 ┌───────────────────────┐
     │      OCR PATH         │                 │      CNN PATH         │
     ├───────────────────────┤                 ├───────────────────────┤
     │ Grayscale + bilateral │                 │ Resize 224×224        │
     │ Text-region detection │                 │ ImageNet normalize    │
     │ Tesseract OCR         │                 │ Augment (train only)  │
     │ Fuzzy match to vocab  │                 │ Train 3 models        │
     │ Regex entity extract  │                 │ Keep best val AUC     │
     └───────────┬───────────┘                 └───────────┬───────────┘
                 └────────────────────┬────────────────────┘
                                      ▼
                    ┌──────────────────────────────────┐
                    │ Evaluate on the same 780 test    │
                    │ images; save CSV, JSON, figures  │
                    └────────────────┬─────────────────┘
                                     ▼
                          ┌─────────────────────┐
                          │  HTML dashboard     │
                          └─────────────────────┘
```

### Part 1: OCR pipeline (notebook blocks 1 to 7)

| Step | What happens | Why |
|---|---|---|
| 1. Denoise | Convert to grayscale and apply a bilateral filter (`d=7, σ=60`). | Removes paper noise while keeping pen-stroke edges sharp. |
| 2. Find text regions | Adaptive Gaussian threshold (block 41, C 11), dilation with a 25×7 kernel (2 passes), then external contours. Boxes under 1,500 px² or covering the whole page are dropped; the 12 largest are kept. | Groups nearby characters into text blocks so each line is read on its own. Used only on full prescriptions (Dataset B); Dataset A images are already cropped. |
| 3. Upscale small crops | Crops under 60 px high or 200 px wide are enlarged 2× (bicubic). | Tesseract reads small text poorly. |
| 4. OCR | `pytesseract` with `--oem 3 --psm 6`. | Default engine (LSTM where available), treating each crop as one block of text. |
| 5. Normalize text | Lowercase, keep only letters, digits, spaces, and hyphens. | Makes matching robust to stray symbols. |
| 6. Fuzzy match | RapidFuzz `token_set_ratio` against the 78 names in `mapping.json`; accept if the score is at least 75. | OCR output is noisy, so an exact match almost never happens. |
| 7. Entity extraction | `DRUG` from the vocabulary match, `DOSAGE` and `FREQUENCY` from regular expressions (for example `500 mg`, `BD`, `TID`, `every 8 hours`). | Turns raw text into structured prescription fields, saved as JSON. |

### Part 2: CNN classification (notebook blocks after "For Model Selection")

| Step | What happens |
|---|---|
| Load and check | Reload Dataset A, plot class distributions, sample images, and image-size histograms, and check that files are readable. |
| Preprocess | Resize to 224×224, normalize with ImageNet mean and std. |
| Augment (train only) | Random rotation up to 8°, random horizontal flip, brightness and contrast jitter of 0.1. |
| Train | 3 epochs, batch size 32, AdamW, cross-entropy loss, mixed precision (AMP) on GPU. |
| Select | Save the checkpoint with the best **validation** macro ROC-AUC. |
| Test | Reload the best checkpoint and evaluate once on the test set: ROC-AUC, macro F1, classification report, and confusion matrix. |

---

## Models

| Model | Backbone | Trainable parameters | Training setup | Role in the project |
|---|---|---|---|---|
| **ResNet-50** | ImageNet-pretrained, **frozen** | 159,822 (new head only) | AdamW, LR 2e-4, weight decay 1e-4 | Main transfer-learning model; best result. |
| **DenseNet-121** | ImageNet-pretrained, **frozen** | 79,950 (new head only) | AdamW, LR 2e-4, weight decay 1e-4 | Second backbone, to see how much architecture choice matters. |
| **ElasticNet CNN** | None (trained from scratch) | 408,462 | AdamW, LR 5e-4, L1 = L2 = 1e-7 | Baseline without pretraining. |
| **Tesseract OCR** | Pretrained OCR engine | None (not trained) | `--oem 3 --psm 6` + fuzzy match | Classical OCR baseline. |

**Classifier head (ResNet-50 and DenseNet-121):**
`backbone → global average pool → Dropout(0.3) → Linear(→ 78) → softmax`

**ElasticNet CNN architecture:**
`Conv(3→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Conv(64→128) → ReLU → MaxPool → Conv(128→256) → ReLU → AdaptiveAvgPool → Dropout(0.35) → Linear(256→78)`

If the pretrained weights cannot be downloaded (internet off), the notebook falls back to random initialization and prints a warning.

---

## Techniques and Algorithms

| Technique | Where it is used | What it does | Why it is used |
|---|---|---|---|
| **Bilateral filtering** | OCR preprocessing | Smooths an image while keeping edges. | Cleans paper texture without blurring pen strokes. |
| **Adaptive thresholding** | Text-region detection | Picks a black/white threshold per neighbourhood. | Copes with uneven lighting in photographed prescriptions. |
| **Morphological dilation + contours** | Text-region detection | Thickens strokes so letters merge into blocks, then outlines each block. | Finds where the text lines are. |
| **Tesseract OCR (LSTM)** | OCR path | Converts image text into characters. | Standard open-source OCR baseline. |
| **Fuzzy string matching (RapidFuzz token-set ratio)** | OCR path | Scores how similar two strings are, ignoring word order and duplicates. | Maps noisy OCR output to the closest real medicine name. |
| **Regex-based NER** | Dataset B | Pattern-matches dosage units and dosing frequencies. | Extracts structured fields with no training data needed. |
| **Transfer learning** | ResNet-50, DenseNet-121 | Reuses features learned on ImageNet; only a new head is trained. | Only 40 training images per class, too few to train a deep network from scratch. |
| **Residual connections (ResNet)** | ResNet-50 | Each block learns `F(x) + x`. | Allows very deep networks to train without vanishing gradients. |
| **Dense connections (DenseNet)** | DenseNet-121 | Each layer receives all earlier feature maps in its block. | Reuses features with fewer parameters. |
| **ElasticNet regularization** | Scratch CNN | Adds `λ1·Σ\|w\| + λ2·Σw²` to the loss. | Discourages large weights to limit overfitting. |
| **Dropout** | All CNN heads | Randomly zeroes units during training. | Another guard against overfitting. |
| **Data augmentation** | CNN training | Small rotations, flips, brightness and contrast changes. | Makes the model less sensitive to how the image was captured. |
| **Softmax + cross-entropy** | All CNNs | Turns logits into class probabilities and penalizes wrong ones. | Standard loss for multi-class classification. |
| **AdamW optimizer** | All CNNs | Adam with decoupled weight decay. | Stable, fast convergence. |
| **Mixed precision (AMP)** | All CNNs | Runs parts of training in float16. | Faster training and less GPU memory. |
| **Checkpoint selection on validation ROC-AUC** | All CNNs | Keeps the epoch with the best validation score. | Picks the model without ever looking at the test set. |

### Evaluation metrics

- **Accuracy:** share of test images whose top prediction is correct.
- **Macro F1-score:** F1 (harmonic mean of precision and recall) averaged equally over all 78 classes.
- **Macro ROC-AUC (one-vs-rest):** how well the predicted probabilities rank the true class above the others; 0.5 is random, 1.0 is perfect.
  It can be well above chance even when accuracy is low, because it rewards putting the right answer near the top.
- **Confusion matrix:** which classes get confused with which.
- **Hit rate (Dataset B only):** share of prescriptions where any vocabulary medicine was matched; this measures coverage, not correctness.

---

## How to Run the Notebook

The notebook is written for **Kaggle** (it reads from `/kaggle/input` and writes to `/kaggle/working`).
Kaggle is the easiest way to run it.

### Option 1: Kaggle (recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and create a **New Notebook**.
2. **File → Import Notebook** and upload `ocr-handwritten-prescription.ipynb`.
3. In the right panel, **Add Input** and search for and add both datasets:
   - `ocr-processed-handwritten-prescriptions`
   - `illegible-medical-prescription-images-dataset`
4. **Settings:**
   - **Accelerator:** GPU T4 (the recorded run used a single Tesla T4).
   - **Internet:** On (needed for `pip install` and to download the ResNet-50 and DenseNet-121 weights).
5. **Run All.**
   On a T4 the whole notebook takes roughly 10 minutes, including package installs and weight downloads:
   - OCR evaluation on Dataset A: about 90 seconds.
   - OCR on 60 Dataset B prescriptions: about 60 seconds.
   - CNN training: 3 models × 3 epochs, about 19 to 37 seconds per epoch.
6. Download the outputs from `/kaggle/working` (see [Outputs](#outputs)).
7. To refresh the dashboard, **File → Download Notebook** (with outputs) and save it over `ocr-handwritten-prescription.ipynb` in this folder.

### Option 2: Run locally

Running locally works but needs a few manual steps.

1. **Install Python 3.10+** (the recorded run used Python 3.12).
2. **Install the Tesseract OCR engine** (the Python package is only a wrapper):
   - Windows: install from the [UB Mannheim Tesseract builds](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH`.
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
3. **Install the Python packages:**

   ```bash
   pip install numpy pandas matplotlib pillow opencv-python-headless pytesseract rapidfuzz scikit-learn jupyter
   pip install torch torchvision   # pick the CUDA build from pytorch.org if you have an NVIDIA GPU
   ```

4. **Download both datasets** from Kaggle and extract them.
5. **Point the notebook at your data.**
   The notebook uses fixed Kaggle paths, so either:
   - recreate them (put the datasets under `/kaggle/input/` and create `/kaggle/working/`), or
   - edit the paths in the notebook: `base = "/kaggle/input"` in the two `find_dataset_*` functions and `kaggle_input` in `find_processed_root`, plus every `/kaggle/working/...` output path.
6. **Start Jupyter and run all cells:**

   ```bash
   jupyter notebook ocr-handwritten-prescription.ipynb
   ```

   Without a GPU it still runs on CPU, just much more slowly.

**Reproducibility:** random seeds are fixed to `42` for Python, NumPy, and PyTorch.
GPU results can still differ slightly between runs and hardware.

---

## Running the Dashboard

The dashboard is a **static HTML page**.
It needs no server, no build step, and no internet connection.

### Open it

- **Easiest:** double-click `dashboard/index.html` to open it in your browser.
- **Or serve it locally** (handy if your browser restricts local files):

  ```bash
  cd dashboard
  python -m http.server 8000
  ```

  Then open <http://localhost:8000>.

### What it shows

| Section | Contents |
|---|---|
| **Overview** | Headline KPIs: best test AUC, macro F1, and accuracy, plus OCR accuracy (Dataset A) and hit rate (Dataset B). |
| **Pipeline** | The steps of both approaches. |
| **Data** | Dataset A split sizes and random labelled sample images. |
| **OCR baseline** | OCR accuracy (Dataset A) and hit rate (Dataset B) versus random guessing, and sample raw OCR output with extracted entities. |
| **Models** | Comparison table (parameters, AUC, F1, accuracy, time per epoch), test-metric chart, and validation AUC per epoch. |
| **Model detail** | Per-model loss and macro F1 curves, per-class test F1, and confusion matrix (switch models with the tabs). |
| **Findings** | What the results mean and what to try next. |
| **Figures** | Every figure from the notebook; click any to enlarge. |

Use the **Toggle theme** button for light or dark mode.

### Regenerate it after a new notebook run

`build_data.py` reads the executed notebook and rewrites `dashboard/data.js` and `dashboard/img/`.
It uses only the Python standard library.

```bash
python dashboard/build_data.py
```

Then reload `index.html`.

> The script finds figures and logs by **cell position** (cells 2, 4, 5, 11, and 15).
> If you add, remove, or reorder notebook cells, update the cell numbers in `build_data.py`.
> The notebook must be saved **with its outputs**, or there is nothing to extract.

---

## Outputs

Files written to `/kaggle/working` by the notebook:

| File | Contents |
|---|---|
| `best_resnet50.pt`, `best_densenet121.pt`, `best_elasticnet_cnn.pt` | Best checkpoint of each CNN (by validation AUC). |
| `history_<model>.csv` | Per-epoch train and validation loss, F1, and AUC. |
| `model_comparison.csv` | Final test AUC and F1 for each CNN. |
| `A_vs_B_comparison.csv` | OCR accuracy on Dataset A and hit rate on Dataset B. |
| `datasetB_demo_outputs.json` | For each prescription: file name, OCR text preview, and extracted entities. |
| `datasetB_roi_visualization.png` | First prescription with detected text regions drawn as boxes. |

Example JSON record:

```json
{
  "file": "1.jpg",
  "ocr_preview": "Dr B. Who Farmstreet 12 Kirkville tel. 3876, R/ date ...",
  "entities": []
}
```

---

## Limitations

- **Accuracy is far too low for practical use** (8% at best).
- **Small dataset:** only 40 training images per class, and only 10 per class for validation and test, so every metric is noisy.
- **Closed vocabulary:** the CNNs can only predict the 78 medicines they were trained on.
- **Short training:** 3 epochs with frozen backbones; the curves show the models had not finished learning.
- **Horizontal-flip augmentation** creates mirrored handwriting that never appears in real prescriptions, which probably hurts more than it helps.
- **Dataset B is unlabelled,** so the end-to-end OCR pipeline cannot be scored for accuracy.
- **Tesseract** is designed for printed text, not cursive handwriting.

---

## Future Work

- Replace Tesseract with a handwriting model such as **CRNN + CTC** or **TrOCR**, fine-tuned on prescriptions, so any word can be read rather than only 78 names.
- **Unfreeze deeper backbone layers** and train for many more epochs with a lower learning rate.
- Swap horizontal flips for **handwriting-friendly augmentations** (slight scaling, shearing, elastic distortion).
- **Hand-label a subset of Dataset B** so the full pipeline can be scored end to end.
- Add **Grad-CAM** heatmaps to show which pen strokes drove each prediction.
- Build a small **web app** where a pharmacist uploads a photo and confirms the top-3 suggestions.

---

## Team

Department of Artificial Intelligence and Machine Learning, Sir M. Visvesvaraya Institute of Technology, Bengaluru.

| Name | USN |
|---|---|
| Amandeep Singh | 1MV23AI005 |
| Anijesh Kumar | 1MV23AI007 |
| Harit Mehta | 1MV23AI019 |

**Guide:** Ms. Lavanya S, Assistant Professor, Dept. of AIML.

---

## References

1. R. Plamondon and S. N. Srihari, "On-Line and Off-Line Handwriting Recognition: A Comprehensive Survey," *IEEE TPAMI*, 22(1), 63-84, 2000.
2. R. Smith, "An Overview of the Tesseract OCR Engine," *ICDAR*, 629-633, 2007.
3. B. Shi, X. Bai, and C. Yao, "An End-to-End Trainable Neural Network for Image-Based Sequence Recognition and Its Application to Scene Text Recognition," *IEEE TPAMI*, 39(11), 2298-2304, 2017.
4. M. Li et al., "TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models," *AAAI*, 37(11), 13094-13102, 2023.
5. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 770-778, 2016.
6. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," *CVPR*, 2261-2269, 2017.
7. H. Zou and T. Hastie, "Regularization and Variable Selection via the Elastic Net," *JRSS Series B*, 67(2), 301-320, 2005.
