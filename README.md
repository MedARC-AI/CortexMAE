# CortexMAE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/clane9/fmri-fm/blob/clean_release/notebooks/quickstart.ipynb)
[![Preprint](https://img.shields.io/badge/arXiv-2510.13768-green?logo=bookstack&logoColor=white)](https://arxiv.org/abs/2510.13768)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

CortexMAE is an fMRI foundation model trained on 2.1K hours of fMRI data from the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/overview) using the [masked autoencoder](https://arxiv.org/abs/2205.09113) framework. We release a family of models trained with different fMRI input representations:
- **CortexMAE-P**: a computationally efficient model based on the Schaefer-400 parcellation.
- **CortexMAE-F**: our flagship model based on fMRI flat maps.
- **CortexMAE-V**: a dense volume model based on an efficient cortex-only representation.

<p align="center">
  <img src=".github/fmri_spaces.png" width="600">
</p>

The strength of CortexMAE is its learned representations of dynamic fMRI activity patterns. On cross-subject decoding tasks, CortexMAE-P and CortexMAE-V are best in their respective classes, while CortexMAE-F is SotA overall.

<p align="center">
  <img src=".github/state_decoding.png" width="450">
</p>

## Installation

```bash
uv pip install "cortexmae @ git+https://github.com/MedARC-AI/cortexmae.git"
```

Or clone the repo and install locally

```bash
git clone https://github.com/MedARC-AI/cortexmae
cd cortexmae
uv sync --python 3.11
```

## Quickstart

Load a pretrained model and compute embeddings on a preprocessed fMRI time series from OpenNeuro:

```python
import cortex_mae
from cortex_mae import CortexMAE

model = CortexMAE.from_pretrained("cortex_mae_flat")

path = cortex_mae.resolve_file(
  "s3://openneuro.org/ds006072/NON_BIDS/ciftis/sub-1_Drug2_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii",
  anon=True,
)
embeds = model.run_embedding(path)
print(embeds.patch_embeds.shape)  # (batch, clips, tokens, dim)
```

See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) for the full demo.

## Pretrained models

Pretrained checkpoints and training logs are available [HuggingFace](https://huggingface.co/medarc/CortexMAE). We release default models for each input space:

| name                  | input space        | shape       | size  |
| --------------------- | ------------------ | ----------- | ----- |
| `cortex_mae_flat`     | flat map           | 224×560     | ViT-B |
| `cortex_mae_parcel`   | Schaefer-400       | 400×1       | ViT-B |
| `cortex_mae_volume`   | MNI cortex         | 465×512     | ViT-B |

as well as >50 ablation variants covering data scale, model scale, alternative parcellations, etc. List all the available models with `cortex_mae.list_models()`.

```python
model = CortexMAE.from_pretrained("cortex_mae_flat")     # default
model = CortexMAE.from_pretrained("cortex_mae_flat_d6")  # depth-6 model
```

## Datasets

Benchmark datasets are distributed in HuggingFace Arrow format on the MedARC R2
bucket, maintained by [Brainmarks](https://github.com/MedARC-AI/brainmarks). To
request access, fill out [this form](https://forms.gle/VGnakBFCBoNnUt2C7), then
configure credentials:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_ENDPOINT_URL_S3=...   # Cloudflare R2 endpoint
```

The HCP-YA pretraining data are also available as [webdataset](https://github.com/webdataset/webdataset) shards. The data can be streamed from R2 during pretraining or downloaded locally.

## Pretraining

```bash
uv run python src/cortex_mae/main_pretrain.py --overrides input_space=flat
```

See the default config [src/cortex_mae/config/default_pretrain.yaml](src/cortex_mae/config/default_pretrain.yaml) for all available options.

## Downstream evaluation

Probe evaluation uses [Brainmarks](https://github.com/MedARC-AI/Brainmarks). The CortexMAE encoder is registered as
`cortex_mae`:

```bash
uv run python -m brainmarks.main_probe cortex_mae patch attn nsd_cococlip \
  --overrides \
  model_kwargs.model_name=cortex_mae_flat
```

## License

Code is released under the Apache License 2.0 (see [LICENSE](LICENSE)). Released
model weights on HuggingFace are licensed CC-BY-NC (see [LICENSE.models](LICENSE.models)).

## Citation

```bibtex
@article{cortexmae,
  title   = {Scaling Vision Transformers for Functional {MRI} with Flat Maps},
  author  = {Lane, Connor and Tripathy, Mihir and Murali, Leema Krishna and
             Grandhi, Ratna Sagari and Yang, Shamus Sim Zi and Gijsen, Sam and
             Das, Debojyoti and Ram, Manish and Singh, Utkarsh Kumar and
             Villanueva, Cesar Kadir Torrico and Wei, Yuxiang and Beddow, Will and
             Cort\'{e}s, Gianfranco and Cho, Suin and Kaplan, Daniel Z. and
             Warner, Benjamin and Abraham, Tanishq Mathew and Scotti, Paul S.},
  journal = {arXiv preprint arXiv:2510.13768},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.13768}
}
```
