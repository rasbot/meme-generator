"""This module creates GAN images for a specified category.
"""
# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was adopted from an example notebook:
# https://github.com/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb

# ==============================================================================
import json
import argparse
from random import randint
import PIL.Image
import numpy as np
from scipy.stats import truncnorm

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

tf.disable_v2_behavior()


def load_model(model_path: str) -> hub.Module:
    """Load a hub model.

    Args:
        model_path (str): Path to model.

    Returns:
        hub.Module: hub model.
    """
    tf.reset_default_graph()
    print("Loading BigGAN module from:", model_path)
    model = hub.Module(model_path)
    return model


def truncated_z_sample(
        batch_size: int, truncation_coeff: float = 1.0, seed: float = None
) -> np.ndarray:
    """Truncate normal distribution to prevent sampling on tails.

    Lower truncation gives more diversity but image fidelity
    is lowered.

    Args:
        batch_size (int): Number of samples to generate.
        truncation_coeff (float, optional): Truncation coefficient.
            Defaults to 1.
        seed (float, optional): Seed used to determine latent features.
            Defaults to None.

    Returns:
        np.ndarray: Truncated z sample.
    """
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return truncation_coeff * values


def one_hot(index: np.ndarray, vocab_size_: int = 1000) -> np.ndarray:
    """One hot encoder for label array.

    Creates an array of zeros with size `vocab_size` and replaces the nth
    element with 1, where n is an index value.

    Args:
        index (np.ndarray): Index array to one hot encode.
        vocab_size (int, optional): Size of resulting array.
            Defaults to vocab_size.

    Returns:
        np.ndarray: One hot encoded array.
    """
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output_ = np.zeros((num, vocab_size_), dtype=np.float32)
    output_[np.arange(num), index] = 1
    return output_


def one_hot_if_needed(
        label: np.ndarray,
        vocab_size_: int = 1000
        ) -> np.ndarray:
    """One hot encode a label and ensure the shape of the array is equal to 2.

    Calls `one_hot` and one hot encodes a label array.

    Args:
        label (np.ndarray): Label array of categories.
        vocab_size (int, optional): Size of resulting array.
            Defaults to vocab_size.

    Returns:
        np.ndarray: One hot encoded array.
    """
    label = np.asarray(label)
    if len(label.shape) <= 1:
        label = one_hot(label, vocab_size_)
    assert len(label.shape) == 2
    return label


def sample(
        sess_: object,  # tf.python.client.session.Session,
        noise_: np.ndarray,
        label_: int,
        truncation_: float = 1.0,
        batch_size_: int = 10,
        vocab_size_: int = 1000,
) -> np.ndarray:
    """Generate samples with GAN.

    Main function used to generate images using bigGAN.

    Args:
        sess_ (tf.python.client.session.Session): Tensorflow session.
        noise_ (np.ndarray): Truncated z sample.
        label_ (int): Label of sample type.
        truncation_ (float, optional): Truncation coefficient.
            Defaults to 1.
        batch_size_ (int, optional): Number of samples to generate.
            Defaults to 10.
        vocab_size_ (int, optional): Used in one hot encoding.
            Defaults to 1000.

    Raises:
        ValueError: Raises exception if the number of noise samples is not
            equal to the number of label samples.

    Returns:
        np.ndarray: Numpy array of generated images.
    """
    noise = np.asarray(noise_)
    label = np.asarray(label_)
    num = noise.shape[0]
    if len(label.shape) == 0:
        label = np.asarray([label] * num)
    if label.shape[0] != num:
        raise ValueError(
            "Got # noise samples ({}) != # label samples ({})".format(
                noise.shape[0], label.shape[0]
            )
        )
    label = one_hot_if_needed(label, vocab_size_=vocab_size_)
    ims_ = []
    for batch_start in range(0, num, batch_size_):
        slice_ = slice(batch_start, min(num, batch_start + batch_size_))
        feed_dict = {
            input_z: noise[slice_],
            input_y: label[slice_],
            input_trunc: truncation_,
        }
        ims_.append(sess_.run(output, feed_dict=feed_dict))
    ims_ = np.concatenate(ims_, axis=0)
    assert ims_.shape[0] == num
    ims_ = np.clip(((ims_ + 1) / 2.0) * 256, 0, 255)
    ims_ = np.uint8(ims_)
    return ims_


def imgrid(imarray: np.ndarray, cols: int = 5, pad: int = 1) -> np.ndarray:
    """Create a grid of images.

    Args:
        imarray (np.ndarray): Numpy array of images.
        cols (int, optional): Number of columns in the grid.
            Defaults to 5.
        pad (int, optional): Optional padding if number of generated
            images do not fit evenly into the grid. Defaults to 1.

    Raises:
        ValueError: Raise exception if input array is not the right type.

    Returns:
        np.ndarray: A numpy array of images.
    """
    if imarray.dtype != np.uint8:
        raise ValueError("imgrid input imarray must be uint8")
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    n_samples, height, width, channels = imarray.shape
    rows = n_samples // cols + int(n_samples % cols != 0)
    batch_pad = rows * cols - n_samples
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, "constant", constant_values=255)
    height += pad
    width += pad
    grid = (
        imarray.reshape(rows, cols, height, width, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * height, cols * width, channels)
    )
    if pad:
        grid = grid[:-pad, :-pad]
    return grid


if __name__ == "__main__":

    with open("_data/categories.json", "r") as infile:
        CATEGORIES = json.load(infile)

    parser = argparse.ArgumentParser(
        description="Pick parameters for GAN generator."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="number of samples generated by GAN. int range from 1 to 20.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="230",
        help="category to generate."
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=0,
        help="int with range from 0 to 100."
    )
    parser.add_argument(
        "--truncation",
        type=float,
        default=0.4,
        help="float with range 0.02 to 1."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="gan_images_{}.jpeg".format(randint(0, 999999999)),
        help="filename of image.",
    )
    args = parser.parse_args()

    num_samples = args.num_samples
    truncation = args.truncation
    noise_seed = args.noise
    filename = args.filename

    category = CATEGORIES[args.category]

    MODULE_PATH = "https://tfhub.dev/deepmind/biggan-deep-512/1"

    module = load_model(MODULE_PATH)

    inputs = {
        k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
        for k, v in module.get_input_info_dict().items()
    }
    output = module(inputs)

    print(
        "Inputs:\n", "\n".join(
            "  {}: {}".format(*kv) for kv in inputs.items()
        )
    )
    print("-" * 30)
    print("Output:", output)

    input_z = inputs["z"]
    input_y = inputs["y"]
    input_trunc = inputs["truncation"]

    dim_z = input_z.shape.as_list()[1]
    vocab_size = input_y.shape.as_list()[1]

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)

    z = truncated_z_sample(num_samples, truncation, noise_seed)
    y = int(category.split(")")[0])

    ims = sample(sess, z, y, truncation_=truncation, vocab_size_=vocab_size)
    img_grid = imgrid(ims, cols=min(num_samples, 5))
    im = PIL.Image.fromarray(img_grid)
    im.save(filename)
