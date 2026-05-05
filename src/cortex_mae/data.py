# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import inspect
import os
import subprocess
from glob import glob
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Iterable, Literal

import braceexpand
import numpy as np
import torch
import datasets as hfds
import webdataset as wds
from huggingface_hub.utils import disable_progress_bars

DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "/tmp/datasets")

disable_progress_bars()


def make_fmri_wds_dataset(
    url: str | list[str],
    num_frames: int = 16,
    clipping: Literal["random", "sequential"] = "random",
    clipping_kwargs: dict[str, Any] | None = None,
    shuffle: bool = True,
    buffer_size: int = 1000,
) -> wds.WebDataset:
    """Make fMRI webdataset for pretraining."""

    # resampling creates an infinite stream of shards sampled with replacement,
    # guaranteeing that no process runs out of data early in distributed training.
    # see webdataset FAQ: https://github.com/webdataset/webdataset/blob/main/FAQ.md
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    # when streaming from s3, we can sometimes get KeyError due to incomplete shards (I guess?)
    # in any case, just ignore with the warn and continue handler
    dataset = dataset.decode().map(extract_fmri_sample, handler=warn_and_continue)

    # generate clips before shuffling for slightly better mixing.
    clipping_kwargs = clipping_kwargs or {}
    clip_fn = make_clipping(clipping, num_frames=num_frames, **clipping_kwargs)
    dataset = dataset.compose(clip_fn)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset


def expand_urls(urls: str | list[str]) -> list[str]:
    """
    Expand wds urls:

    - expand glob patterns
    - expand brace expressions
    - filter files that don't exist

    Adapted from `webdataset.shardlists.expand_urls`.
    """
    if isinstance(urls, str):
        urls = [urls]
    results = []
    for url in urls:
        chars = set(url)
        if chars.intersection("[*?"):
            result = sorted(glob(url))
        elif "{" in chars:
            result = braceexpand.braceexpand(url)
        else:
            result = [url]
        results.extend(result)
    return results


def warn_and_continue(exn):
    # modified wds warn and continue handler to send warning to stdout log.
    # but note, this won't propagate to the wandb console log since it will
    # originate in a child data loader worker process.
    print(f"WARNING {repr(exn)}")
    return True


def maybe_download(url: str, cache_dir: str | Path | None = None) -> str:
    cache_dir = Path(cache_dir or DATA_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)

    parsed = urlparse(url)
    # NOTE: previously we also downloaded hf urls, but we can delegate this to hf
    # load_dataset itself.
    if parsed.scheme == "s3":
        local_path = str(Path(cache_dir) / parsed.path.removeprefix("/"))
        print(f"downloading {url} -> {local_path}")
        # TODO: is this the best way to download from s3?
        # it's faster at least than letting hf download
        subprocess.run(["aws", "s3", "sync", "--quiet", str(url), str(local_path)], check=True)
    else:
        local_path = url
    return local_path


def extract_fmri_sample(sample: dict[str, Any]):
    # sample metadata
    meta = sample["meta.json"]

    # task trial events in BIDS events format.
    events = sample.get("events.json", [])

    # fMRI bold data, shape (T, D), dtype float16
    # z-scored per dimension
    bold = sample["bold.npy"]

    # for loading hf data that don't include the mean and std
    if "mean.npy" not in sample:
        mean = np.zeros(bold.shape[1], dtype=np.float32)
        std = np.ones(bold.shape[1], dtype=np.float32)
    else:
        mean = sample["mean.npy"]
        std = sample["std.npy"]
    return {"meta": meta, "events": events, "bold": bold, "mean": mean, "std": std}


def random_clips(num_frames: int = 16, oversample: float = 1.0):
    """Webdataset filter to generate random clips.

    The number of clips is `oversample * T / num_frames`.
    """

    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["bold"]
            n_clips = int(oversample * len(image) / num_frames)
            indices = np.sort(np.random.randint(0, len(image) - num_frames + 1, size=n_clips))
            for start in indices:
                # copy to avoid a memory leak when used with a shuffle buffer.
                clip = image[start : start + num_frames].copy()

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "bold": clip,
                    "mean": sample["mean"],
                    "std": sample["std"],
                    "start": start,
                }

    return _filter


def sequential_clips(num_frames: int = 16, stride: int | None = None):
    """Webdataset filter to generate sequential clips.

    By default, stride = num_frames.
    """
    stride = stride or num_frames

    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["bold"]
            for start in range(0, len(image) - num_frames + 1, stride):
                clip = image[start : start + num_frames].copy()

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "bold": clip,
                    "mean": sample["mean"],
                    "std": sample["std"],
                    "start": start,
                }

    return _filter


def event_clips(num_frames: int = 16, tr: float = 1.0, hrf_delay: float = 0.0):
    """Webdataset filter to generate event-locked clips.

    tr and hrf_delay are in seconds. A 1s tr is the default for flat datasets. Setting
    hrf_delay > 0, e.g. to 3 or 4 seconds can concentrate the clip more on the
    activation peak.
    """

    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["bold"]
            events = sample["events"]
            for event in events:
                start = int((event["onset"] + hrf_delay) / tr)
                if start + num_frames > len(image):
                    continue
                clip = image[start : start + num_frames].copy()

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "bold": clip,
                    "mean": sample["mean"],
                    "std": sample["std"],
                    "start": start,
                    **event,
                }

    return _filter


CLIPPING_DICT = {
    "random": random_clips,
    "sequential": sequential_clips,
    "event": event_clips,
}


def make_clipping(clipping: str, **kwargs) -> Callable:
    clip_fn = CLIPPING_DICT[clipping]
    kwargs = filter_kwargs(clip_fn, kwargs)
    return clip_fn(**kwargs)


def filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs


class HFDataset(torch.utils.data.Dataset):
    # we have this light wrapper to be able to apply on the fly transforms without undoing the torch formatting
    # https://github.com/huggingface/datasets/issues/6012

    def __init__(
        self,
        dataset: hfds.Dataset,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.dataset = dataset
        self.dataset.set_format("torch")
        self.transform = transform

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        s = f"    dataset={self.dataset},\n    transform={self.transform}"
        s = f"HFDataset(\n{s}\n)"
        return s
