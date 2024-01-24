import os
import pickle as pkl
import urllib.request as request
from collections import defaultdict

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict


def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if not isinstance(offline_batch, dict):
        offline_batch = offline_batch.unfreeze()

    if not isinstance(online_batch, dict):
        online_batch = online_batch.unfreeze()

    for k, v in offline_batch.items():
        if type(v) is dict:
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            batch[k] = jnp.concatenate((offline_batch[k], online_batch[k]), axis=axis)

    return frozen_dict.freeze(batch)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)


def _unpack(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            obs = batch["observations"].copy(add_or_replace={pixel_key: obs_pixels})
            next_obs = batch["next_observations"].copy(
                add_or_replace={pixel_key: next_obs_pixels}
            )
            batch = batch.copy(
                add_or_replace={"observations": obs, "next_observations": next_obs}
            )

    return batch


def load_resnet10_params(agent, image_keys=("image",), public=False):
    if not public:  # if github repo is not public, load from local file
        with open("resnet10_params.pkl", "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        file_name = "resnet10_params.pkl"
        # Construct the full path to the file
        file_path = os.path.expanduser(f"~/.serl/{file_name}")

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exists at '{file_path}'.")
        else:
            url = "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
            print(f"Downloading file from {url}")
            try:
                request.urlretrieve(url, file_path)
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = agent.state.params.unfreeze()
    for image_key in image_keys:
        new_encoder_params = new_params["modules_actor"]["encoder"][
            f"encoder_{image_key}"
        ]
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for k in new_encoder_params:
            if k in encoder_params:
                new_encoder_params[k] = encoder_params[k]
                print(f"replaced {k} in pretrained_encoder")

    from flax.core.frozen_dict import freeze

    new_params = freeze(new_params)
    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent
