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
# ==============================================================================
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import io
import json
import IPython.display
import numpy as np
import PIL.Image
import argparse
from random import randint
from scipy.stats import truncnorm
import tensorflow_hub as hub


def truncated_z_sample(batch_size, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
    return truncation * values

def one_hot(index, vocab_size=vocab_size):
    index = np.asarray(index)
    if len(index.shape) == 0:
        index = np.asarray([index])
    assert len(index.shape) == 1
    num = index.shape[0]
    output = np.zeros((num, vocab_size), dtype=np.float32)
    output[np.arange(num), index] = 1
    return output

def one_hot_if_needed(label, vocab_size=vocab_size):
    label = np.asarray(label)
    if len(label.shape) <= 1:
        label = one_hot(label, vocab_size)
    assert len(label.shape) == 2
    return label

def sample(sess, noise, label, truncation=1., batch_size=8,
           vocab_size=vocab_size):
    noise = np.asarray(noise)
    label = np.asarray(label)
    num = noise.shape[0]
    if len(label.shape) == 0:
        label = np.asarray([label] * num)
    if label.shape[0] != num:
        raise ValueError('Got # noise samples ({}) != # label samples ({})'
                     .format(noise.shape[0], label.shape[0]))
    label = one_hot_if_needed(label, vocab_size)
    ims = []
    for batch_start in range(0, num, batch_size):
        s = slice(batch_start, min(num, batch_start + batch_size))
        feed_dict = {input_z: noise[s], input_y: label[s], input_trunc: truncation}
        ims.append(sess.run(output, feed_dict=feed_dict))
    ims = np.concatenate(ims, axis=0)
    assert ims.shape[0] == num
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    return ims

def interpolate(A, B, num_interps):
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape to interpolate.')
    alphas = np.linspace(0, 1, num_interps)
    return np.array([(1-a)*A + a*B for a in alphas])

def imgrid(imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
        raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def imshow(a, format='png', jpeg_fallback=True):
    a = np.asarray(a, dtype=np.uint8)
    data = io.BytesIO()
    PIL.Image.fromarray(a).save(data, format)
    im_data = data.getvalue()
    im_data.save("test.png")
    try:
        disp = IPython.display.display(IPython.display.Image(im_data))
    except IOError:
        if jpeg_fallback and format != 'jpeg':
            print(('Warning: image was too large to display in format "{}"; '
                'trying jpeg instead.').format(format))
            return imshow(a, format='jpeg')
        else:
            raise Exception("Error in displaying image.")
    return disp

if __name__=='__main__':

    with open('_data/categories.json', 'r') as infile:
        categories = json.load(infile)

    parser = argparse.ArgumentParser(description='Pick parameters for GAN generator.')

    parser.add_argument('--category', type=str, default='230', help='category to generate.')
    parser.add_argument('--noise', type=int, default=0, help='int with range from 0 to 100.')
    parser.add_argument('--truncation', type=float, default=0.4, help='float with range 0.02 to 1.')
    parser.add_argument('--filename', type=str, default='gan_images_{}.jpeg'.format(randint(0,999999999)), help='filename of image.')
    args = parser.parse_args()

    module_path = 'https://tfhub.dev/deepmind/biggan-deep-512/1'

    tf.reset_default_graph()
    print('Loading BigGAN module from:', module_path)
    module = hub.Module(module_path)
    inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
            for k, v in module.get_input_info_dict().items()}
    output = module(inputs)

    print()
    print('Inputs:\n', '\n'.join(
        '  {}: {}'.format(*kv) for kv in inputs.items()))
    print()
    print('Output:', output)

    input_z = inputs['z']
    input_y = inputs['y']
    input_trunc = inputs['truncation']

    dim_z = input_z.shape.as_list()[1]
    vocab_size = input_y.shape.as_list()[1]

    initializer = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(initializer)

    num_samples = 10
    truncation = args.truncation
    noise_seed = args.noise
    filename = args.filename

    category = categories[args.category]

    z = truncated_z_sample(num_samples, truncation, noise_seed)
    y = int(category.split(')')[0])

    ims = sample(sess, z, y, truncation=truncation)
    img_grid = imgrid(ims, cols=min(num_samples, 5))
    im = PIL.Image.fromarray(img_grid)
    im.save(filename)
    # imshow(imgrid(ims, cols=min(num_samples, 5)))