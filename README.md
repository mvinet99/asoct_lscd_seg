# ASOCT LSCD Corneal Epithelial Layer Segmentation

Segmentation of the corneal epithelial layer in AS-OCT scans for Limbal Stem Cell
Deficiency (LSCD), and estimation of corneal epithelial thickness (CET).

Everything runs from a single entry point: **`pipeline.py`**.

## Install

```bash
git clone https://github.com/mvinet99/ascot_lscd_seg.git
cd ascot_lscd_seg

conda create -n lscd python=3.10
conda activate lscd
pip install -r requirements.txt
```

Install PyTorch matching your CUDA version if the default wheel doesn't work:
https://pytorch.org/get-started/locally/

Training logs to Weights & Biases. Log in once with `wandb login`, or set
`WANDB_MODE=disabled` to skip it.

## Expected layout

`pipeline.py` resolves all paths relative to the **current working directory**, so run
it from a directory laid out like this:

```
<working dir>/
├── datasheet.xlsx        # Clinical metadata (required)
├── data/
│   ├── raw_corneal/      # Input AS-OCT scans (.png)
│   └── smoothed/         # Smoothed scans used for normalization
└── results/              # Created and populated by the pipeline
```

`datasheet.xlsx` must contain these columns:

| Column | Used for |
| --- | --- |
| `Previous File Name` | Matching rows to image files |
| `Severity` | Stratifying k-fold splits (`control` / `mild` / `moderate` / `severe`) |
| `Epithelial Thickness` | Manual CET ground truth (M-CET) |
| `Optovue Thickness` | Device-reported CET (OCT-CET) |

The remaining subfolders under `data/` and `results/` are created automatically during
preprocessing.

## Running

`pipeline.py` imports `train` from `model/`, so that directory needs to be on the
Python path:

```bash
PYTHONPATH=model python pipeline.py
```

Multi-GPU (the training stage uses DistributedDataParallel with the NCCL backend):

```bash
PYTHONPATH=model torchrun --nproc_per_node=4 pipeline.py
```

Without `torchrun`, training falls back to a single GPU.

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--mode` | `train` | Run mode |
| `--model_type` | `MSU_Net` | Architecture: `U_Net` or `MSU_Net` |
| `--img_ch` | `3` | Input channels |
| `--output_ch` | `1` | Output channels |
| `--batch_size` | `64` | Batch size |
| `--num_epochs` | `50` | Training epochs |
| `--num_epochs_test` | `50` | Epochs for the test pass |
| `--lr` | `0.001` | Learning rate |
| `--folds` | `5` | Number of cross-validation folds |
| `--num_workers` | `0` | DataLoader workers |
| `--local_rank` | `0` | Set by `torchrun`; don't pass manually |

Example:

```bash
PYTHONPATH=model python pipeline.py --model_type U_Net --folds 5 --num_epochs 100 --batch_size 32
```

## What the pipeline does

`pipeline.py` runs five stages in order:

1. **Preprocess** (`preprocessing.py`) — creates the output folder tree, resizes scans
   to 2200×820, and normalizes them by histogram-matching to a reference image.
2. **Mask creation** (`mask_creation.py`) — builds ground-truth epithelium masks from
   the annotated scans and cuts them into training patches.
3. **Train** (`model/train.py`) — patient-level stratified k-fold cross-validation.
   Each fold trains a separate model under `results/UNet/UNet<N>/`.
4. **Postprocess** (`postprocessing.py`) — reconstructs full-scan masks from patch
   predictions and computes per-scan thickness profiles.
5. **Evaluate** (`evaluation.py`) — compares predicted CET against manual and
   device measurements; reports MAE and Pearson r per severity group.

## Outputs

All results land in `results/`:

- `UNet/UNet<N>/` — model checkpoints and per-fold predictions
- `reconstructed_predictions/` — full-scan predicted masks
- `thickness_npy/` — per-scan thickness arrays
- `eval_images/` — `BoxPlot.png`, `Correlation_AI_CE.png`, `Correlation_OCT.png`

## Notes

- The histogram-matching reference image in `preprocessing.py` (`normalize`) is an
  absolute path pointing at a specific machine. Change it to a path valid on your
  system before running.
- `pandas.read_excel` requires `openpyxl`, which is included in `requirements.txt`.
