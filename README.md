# CortexMAE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MedARC-AI/CortexMAE/blob/main/notebooks/quickstart.ipynb)
[![Preprint](https://img.shields.io/badge/arXiv-2510.13768-green?logo=bookstack&logoColor=white)](https://arxiv.org/abs/2510.13768)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/tVR4TWnRM9?style=flat)](https://discord.gg/tVR4TWnRM9)
[![Code License](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](LICENSE)
[![Model License](https://img.shields.io/badge/Model_License-CC_BY--NC_4.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/deed.en)

CortexMAE is an fMRI foundation model trained on 2.1K hours of fMRI data from the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/overview) using the [masked autoencoder](https://arxiv.org/abs/2205.09113) framework. We release a family of models trained with different fMRI input representations:
- **CortexMAE-P**: a computationally efficient model based on the Schaefer-400 parcellation.
- **CortexMAE-F**: our flagship model based on fMRI flat maps.
- **CortexMAE-V**: a dense volume model based on an efficient cortex-only representation.

<p align="center">
  <img src=".github/fmri_spaces.png" width="600">
</p>

## Installation

```bash
uv pip install cortex_mae
```

Or install the latest version from github

```bash
uv pip install "cortex_mae @ git+https://github.com/MedARC-AI/CortexMAE.git"
```

Or clone the repo and install locally

```bash
git clone https://github.com/MedARC-AI/CortexMAE.git
cd CortexMAE
uv sync --python 3.11
```

## Quickstart

Load a pretrained model and compute embeddings on a preprocessed fMRI time series from OpenNeuro:

```python
from cortex_mae import CortexMAE, resolve_file

model = CortexMAE.from_pretrained("cortex_mae_flat")

path = resolve_file(
  "s3://openneuro.org/ds006072/NON_BIDS/ciftis/sub-1_Drug2_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii",
  anon=True,
)
embeds = model.run_embedding(path)
print(embeds.patch_embeds.shape)  # (clips, tokens, dim)
```

See [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) for the full demo.

## Pretrained models

Pretrained checkpoints and training logs are available on [HuggingFace](https://huggingface.co/medarc/CortexMAE). We release default models for each input space:

| name                  | input space        | shape       | size  |
| --------------------- | ------------------ | ----------- | ----- |
| `cortex_mae_flat`     | flat map           | 224×560     | ViT-B |
| `cortex_mae_parcel`   | Schaefer-400       | 400×1       | ViT-B |
| `cortex_mae_volume`   | MNI cortex         | 465×512     | ViT-B |

as well as >50 ablation variants covering data scale, model scale, alternative parcellations, etc. List all the available models with `cortex_mae.list_models()`.

```python
model = CortexMAE.from_pretrained("cortex_mae_flat")     # default
model = CortexMAE.from_pretrained("cortex_mae_flat_r2")  # repeat with new seed
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

To reproduce pretraining of the default CortexMAE-F model, run

```bash
uv run python src/cortex_mae/main_pretrain.py
```

You can also override defaults

```bash
uv run python src/cortex_mae/main_pretrain.py \
  --config config.yaml \
  --overrides \
  input_space=schaefer400 \
  base_lr=3e-4
```

See the default config [src/cortex_mae/config/default_pretrain.yaml](src/cortex_mae/config/default_pretrain.yaml) for all available options. To reproduce specific model variants, use the original configs on [HuggingFace](https://huggingface.co/medarc/CortexMAE).

## Downstream evaluation

Probe evaluation uses [Brainmarks](https://github.com/MedARC-AI/Brainmarks). The CortexMAE encoders are registered as `cortex_mae_{parcel,flat,volume}`:

```bash
uv run python -m brainmarks.main_probe cortex_mae_flat patch attn nsd_cococlip
```

To evaluate a different model variant:

```bash
uv run python -m brainmarks.main_probe cortex_mae_flat patch attn nsd_cococlip \
    --overrides model_kwargs.variant=d6
```

To see a list of all variants:

```python
from brainmarks.models.cortex_mae_wrapper import list_variants

print(list_variants("cortex_mae_flat"))
```

## Support

For help with any issues, reach out to us on [MedARC Discord](https://discord.gg/tVR4TWnRM9) in the `#neuro-fm` channel.

## License

Code is released under the Apache License 2.0 ([LICENSE](LICENSE)). Model weights are relased under CC-BY-NC 4.0 ([LICENSE.models](LICENSE.models)).

## Citation

```bibtex
@inproceedings{lane2026scaling,
    title={Scaling Vision Transformers for Functional MRI with Flat Maps},
    author={Connor Lane and Mihir Tripathy and Leema Krishna Murali and
            Ratna Sagari Grandhi and Shamus Sim Zi Yang and Sam Gijsen and
            Debojyoti Das and Manish Ram and Utkarsh Kumar Singh and
            Cesar Kadir Torrico Villanueva and Yuxiang Wei and Will Beddow and
            Gianfranco Cortés and Suin Cho and Daniel Z. Kaplan
            and Benjamin Warner and Tanishq Mathew Abraham and Paul S. Scotti},
    booktitle={ICML},
    year={2026},
}
```
