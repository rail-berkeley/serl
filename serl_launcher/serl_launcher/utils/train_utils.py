import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
import matplotlib.pyplot as plt
from flax.core import frozen_dict
from jaxlib.xla_extension import ArrayImpl


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


def load_resnet10_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception("Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count / 1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = agent.state.params

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

    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent


def print_agent_params(agent, image_keys=("image",)):
    """
    helper function to print the parameter count of the actor and critic networks
    """

    def get_size(params):
        return sum(x.size for x in jax.tree.leaves(params))

    total_param_count = get_size(agent.state.params)
    actor, critic = agent.state.params["modules_actor"], agent.state.params["modules_critic"]

    # calculate encoder params
    try:
        pretrained_encoder_count = get_size(actor["encoder"][f"encoder_{image_keys[0]}"]["pretrained_encoder"])
    except Exception as e:
        pretrained_encoder_count = 0
    encoder_count = get_size(actor["encoder"])

    actor_count = get_size(actor)
    critic_count = get_size(critic)

    print(f"\ntotal params: {total_param_count / 1e6:.3f}M")
    print(
        f"encoder params: {(encoder_count - pretrained_encoder_count) / 1e6:.3f}M    pretrained encoder params: {pretrained_encoder_count / 1e6:.3f}M")
    print(f"actor params: {(actor_count - encoder_count) / 1e6:.3f}M       critic_params: {critic_count / 1e6:.3f}M")
    print(f"total parameters to train: {(total_param_count - pretrained_encoder_count) / 1e6:.3f}M\n")


def parameter_overview(agent):
    from clu import parameter_overview
    print(parameter_overview.get_parameter_overview(agent.state.params))


def plot_feature_kernel_histogram(agent):
    import matplotlib.pyplot as plt

    feature_kernel = agent.state.params["modules_actor"]["network"]["Dense_0"]["kernel"]
    feature_bias = agent.state.params["modules_actor"]["network"]["Dense_0"]["bias"]
    print(f"kernel has shape: {feature_kernel.shape}")

    plots = feature_kernel.shape[0] // 256 + 1
    fig, axes = plt.subplots(nrows=plots, ncols=1, sharex=True, sharey=True, figsize=(5, 3 * plots))

    feature_kernel = feature_kernel.mean(axis=1)

    for p in range(plots):
        legend = f"image {p}" if p < plots - 1 else "observations"
        print(f"{legend} mean and std:   ",
              [feature_kernel[p * 256:(p + 1) * 256].mean(), feature_kernel[p * 256:(p + 1) * 256].std()])
        axes[p].hist(feature_kernel[p * 256:(p + 1) * 256], bins=50)
        axes[p].set_title(legend)
    plt.show()


def find_zero_weights(params, print_str="", print_all=False):
    if isinstance(params, dict):
        for key in params.keys():
            find_zero_weights(params[key], print_str=print_str + " " + key, print_all=print_all)
    else:
        if print_all:
            print(f"{print_str}  m:{params.mean()}  s:{params.std()}")
            return
        if abs(params.mean()) < 1e-5:
            print(f" zero weights--> {print_str}    std: {params.std()}")
        else:
            print(".", end='')


def plot_3d_kernel(kernel, nr):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_len, y_len, z_len = kernel.shape

    x = np.arange(x_len)
    y = np.arange(y_len)
    z = np.arange(z_len)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Normalize the values for color mapping
    norm = plt.Normalize(kernel.min(), kernel.max())
    colors = plt.cm.viridis(norm(kernel))
    alpha = np.clip(np.abs(kernel) * 9, 0., 1.)
    alpha = np.clip(np.where(alpha < 0.17, 0., alpha), 0., 1.)

    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len):
                ax.bar3d(i, j, k, .9, .9, .9, color=colors[i, j, k], alpha=alpha[i, j, k])

    ax.set_title(str(nr))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.show()


def plot_conv3d_kernels(params):
    kernels = params["modules_actor"]["encoder"]["encoder_wrist_pointcloud"]["conv_5x5x5"]["kernel"]
    # shape (5, 5, 5, 1, 32)
    for i in range(kernels.shape[-1]):
        print(kernels[..., 0, i].mean(), kernels[..., 0, i].std())
        plot_3d_kernel(kernels[..., 0, i], i)


def get_pretrained_VoxNet_Conv3d_kernels(features: int = 32):
    ckpt = jnp.load("/home/nico/Downloads/c-11/voxnet/conv1/conv3d/kernel:0.npy")
    return ckpt[..., :features]
