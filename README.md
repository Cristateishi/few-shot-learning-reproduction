# Investigating Contextual Calibration in Few-Shot Language Models

**Authors:** *Imana Alibašić\, Cristiano Maia*

**Course:** Understanding Language Models

Partial reproduction of [Zhao et al. (2021), *Calibrate Before Use*](https://proceedings.mlr.press/v139/zhao21c.html) with extensions for the modelo OPT-13B and Qwen3-8B-Base. We compare accuracy before/after calibration and variability across demonstration seeds. The accompanying paper has more information on the changes, methods, results, and limitations.

## Contents

- `run_classification.py`: experiment runner, calibration, evaluation.
- `utils.py`: model loading, inference, prompts, result
- `data_utils.py`: dataset loaders and templates
- `sst2_independent_gpt2xl.ipynb`: independent GPT-2 XL implementation for SST-2
- `saved_results/`: main experiment outputs when included

The code in the .py files is adapted from the [authors' code](https://github.com/tonyzhaozh/few-shot-learning). The changes made are for compatibility with, OPT/Qwen support, and removal of unused GPT-3/API code. The main sampling and calibration process is kept.

## Setup

- CUDA-capable NVIDIA GPU required. GPT-2 XL ran locally on a GTX 1080 Ti (11 GB); OPT and Qwen ran separately on a cloud based service on A40 GPUs (48 GB each)
- Tested with Python 3.13.5 locally / 3.12.3 on RunPod, PyTorch 2.8.0, and Transformers 5.20.2
- Models download automatically from Hugging Face and are cached. No OpenAI API needed, since we didn't use GPT3 like the original paper

For the tested local CUDA 12.6 environment:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==5.20.2 numpy pandas tqdm matplotlib ipykernel
```

On RunPod we kept the preinstalled CUDA-enabled PyTorch instead of reinstalling it

## Data

Use the authors SST-2 and AG News files, run this Python snippet from the project root:

```python
from pathlib import Path
from urllib.request import urlretrieve

base = "https://raw.githubusercontent.com/tonyzhaozh/few-shot-learning/main/data/"
for name in ("sst2/stsa.binary.train", "sst2/stsa.binary.test", "agnews/train.csv", "agnews/test.csv"):
    path = Path("data") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(base + name, path)
```

SST2 uses the authors labeled binary train/test files not GLUE's unlabeled test split

## Main experiments

To run the .py adapted code run from the project root:

```bash
python run_classification.py --models gpt2-xl --datasets "sst2,agnews" --num_seeds 5 --all_shots "0,1,4,8" --subsample_test_set 300 --bs 1
```

- Run OPT and Qwen with the same command just replacing the model id with `facebook/opt-13b` or `Qwen/Qwen3-8B-Base`
- One model per Python process, because the cache retains the first model
- Keep `--bs 1` and omit `--approx`, missing label probabilities are recovered rather than set to zero, our experiment has approx=false by default
- Test sampling uses seed 0 and demonstrations use seeds 0–4
- Outputs: 20 `.pkl` files per dataset run in `saved_results/` while existing matching filenames are overwritten 
- Add `--use_saved_results` to the same command to print existing summaries without inference

## Independent notebook

- Open the notebook in VS Code/Jupyter, select your Python kernel, and run cells in order
- Settings: GPT-2 XL, float32, `APPROX=False`, 300 reviews, 0/1/4/8 shots, five seeds for nonzero shots
- Zero-shot runs once - 16 configurations total
- This outputs prediction/configuration CSVs, design record, summary, plot, and ZIP under `independent_sst2_results/`
