# Meme Generator

## Table of Contents

* [Introduction](#introduction)
* [Image Generation](#image-generation)
  * [GANs](#gans)
* [Text Generation](#text-generation)
  * [GPT-2](#gpt-2)

# Introduction

Using a Flask app, this meme generator will generate pet related memes that have an image of a pet, a quote written on the image, and an associated quote author. There are two types of pets supported - cats and dogs. There are two main modes to generate memes with this project, as well as sub-modes, as described below.

__Local generation__
> User will get a randomized image and quote/author for a specified animal type. 

> The user can also provide their own quote and author which will be used on a randomized image.

__Flask GUI generation__
> User will get a randomized image and quote/author for a specified animal type.

> User can provide an image link, as well as their own quote and author, and a meme will be generated using those parameters.

# Image Generation

Instead of using real animal images, a machine learning model was used to generate dog and cat images to use as memes. The model used to generate the animal images is BigGAN, which is a version of a Generative Aversarial Network (GAN). A GAN is a neural network trained on real images. Training a GAN takes a lot of training data, and to get good results, it can take a long time unless the model is trained on a high end system (specifically the graphics card, or GPU). Luckily there are pre-trained models which can be used to generate images.

## GANs

A GAN has two neural networks under the hood. There is a generator, and a discriminator. The generator will take an input image which is randomized noise, and pass it to the descriminator. The descriminator will give feedback to the generator, telling it how close it is to generating a realistic image of the target class (such as a dog or a cat). The generator will keep trying and attempting to improve the image generation until the metrics for how good the resulting image is pass a certain threshold.

The images used for this project were created with a higher diversity in the generation, which has a tradeoff of looking a bit...weird. This can be entertaining, and a handful of dog and cat images were used from this model.

The code to generate the images is provided, but requires a specific version of Tensorflow to be installed. Within the script has a link to the Tensorflow project code that can be ran via Google colab, which provides an online environment that anyone can use to generate GAN images. The categories are not limited to dogs and cats.