"""
uv run python split_hcpya_wds.py rest
uv run python split_hcpya_wds.py task
"""

from argparse import ArgumentParser
from pathlib import Path
import webdataset as wds

ROOT = "/data/fmri-datasets/pretrain"
SOURCE_URL = f"{ROOT}/hcpya-all.flat.wds/hcpya-all-flat-{{00000..01799}}.tar"


def main():
    parser = ArgumentParser()
    parser.add_argument("subset", choices=["rest", "task"])
    args = parser.parse_args()

    print(f"generating subset: {args.subset}")
    dataset = wds.WebDataset(SOURCE_URL, shardshuffle=False)

    keep_mod = {"rest": "rfMRI", "task": "tfMRI"}[args.subset]

    outdir = Path(f"{ROOT}/hcpya-{args.subset}.flat.wds")
    outdir.mkdir(exist_ok=True, parents=True)

    with wds.ShardWriter(f"{outdir}/hcpya-{args.subset}-flat-%05d.tar", maxsize=6e8) as sink:
        for sample in dataset:
            key = sample["__key__"]
            mod = key.split("_")[1].split("-")[1]
            if mod == keep_mod:
                print(key)
                sink.write(sample)


if __name__ == "__main__":
    main()
