"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.io
import scipy.misc
import numpy as np
import copy
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for cyclegan


def load_svhn(image_dir, split='train', fine_size = 256):
    print ('loading svhn image dataset..')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) / 127.5 - 1
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    print ('finished loading svhn image dataset..!')
    return images, labels

def load_unnormalized_svhn(image_dir, split='train', fine_size = 256):
    print ('loading svhn image dataset..')

    image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

    image_dir = os.path.join(image_dir, image_file)
    svhn = scipy.io.loadmat(image_dir)
    images = np.transpose(svhn['X'], [3, 0, 1, 2]) #/ 127.5 - 1
    labels = svhn['y'].reshape(-1)
    labels[np.where(labels == 10)] = 0
    print ('finished loading svhn image dataset..!')
    return images, labels

def load_mnist(image_dir, split='train', fine_size = 256):
    print ('loading mnist image dataset..')
    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        print ('bbbbbbbbbbbbbbbbbbbbbbbbbb',f)
        mnist = pickle.load(f, encoding='latin1') #because of pickle load issue encoding arg was added, not there previously
    images = mnist['X'] / 127.5 - 1
    labels = mnist['y']
    print ('finished loading mnist image dataset..!')
    return images, labels

def load_unnormalized_mnist(image_dir, split='train', fine_size = 256):
    print ('loading mnist image dataset..')
    image_file = 'train.pkl' if split == 'train' else 'test.pkl'
    image_dir = os.path.join(image_dir, image_file)
    with open(image_dir, 'rb') as f:
        print ('bbbbbbbbbbbbbbbbbbbbbbbbbb',f)
        mnist = pickle.load(f, encoding='latin1') #because of pickle load issue encoding arg was added, not there previously
    images = mnist['X'] #/ 127.5 - 1
    labels = mnist['y']
    print ('finished loading mnist image dataset..!')
    return images, labels

def load_mnist_m(image_dir, split='train', fine_size = 256):
    print ('loading mnist_m image dataset..')
    label_dir = image_dir
    image_file = 'mnist_m_train_images.npy' if split == 'train' else 'mnist_m_test_images.npy'
    label_file = 'mnist_m_train_labels.npy' if split == 'train' else 'mnist_m_test_labels.npy'
    image_dir = os.path.join(image_dir, image_file)
    label_dir = os.path.join(label_dir, label_file)
    #with open(image_dir, 'rb') as f:
     #   mnist = pickle.load(f)
    images = np.load(image_dir)
    #mnist_m_train_images = np.empty((60000,32,32,3))
    #for i in range(images.shape[0]):
     #   temp = images
      #  mnist_m_train_images[i,:,:,:] = scipy.misc.imresize(images[i,:,:,:],[32,32])
    
    #np.save('mnist_m_train_images.npy',mnist_m_train_images)
    #images = mnist_m_train_images
    images = images / 127.5 - 1
    labels = np.load(label_dir)
    print ('finished loading mnist_m image dataset..!')
    return images, labels


def load_usps(image_dir, split='train', fine_size=256):
    print ('loading usps image dataset..')
    image_file = 'usps_all.mat'
    image_dir = os.path.join (image_dir, image_file)
    
    # with open(image_dir, 'rb') as f:
    #   mnist = pickle.load(f)

    mat_contents = scipy.io.loadmat(image_dir)
    images =  mat_contents['data']
    final_images = []
    labels = []#np.zeros([11000], dtype=np.int32)
    for ii in range(10):
        for jj in range(1100):
            image = images[:,jj,ii]
            image = np.reshape(image,[16,16])
            image = np.transpose(image)
            image = np.pad(image, ((6, 6), (6, 6)), 'constant', constant_values=0)
            image = scipy.misc.imresize (
                image, [32, 32])
            image = np.reshape(image,[32,32,1])
            #image = np.tile(image,(1,1,3))
            final_images.append(image)
            labels.append(ii+1)
    # images shape = (256,1100,10)
    final_images = np.asanyarray(final_images,np.float32)
    final_images = final_images / 127.5 - 1
    #plt.plot(final_images[0,:,:,0])
    implot = plt.imshow(final_images[6000,:,:,0])
    plt.savefig('usps1.png')
    implot = plt.imshow (final_images[7000, :, :, 0])
    plt.savefig ('usps2.png')
    implot = plt.imshow (final_images[8000, :, :, 0])
    plt.savefig ('usps3.png')
    implot = plt.imshow (final_images[9000, :, :, 0])
    plt.savefig ('usps4.png')
    implot = plt.imshow (final_images[10000, :, :, 0])
    plt.savefig ('usps5.png')
    labels = np.asanyarray(labels,np.int32)
    labels[np.where (labels == 10)] = 0
    #print ('size of usps', final_images.shape)
    #print ('size of usps', labels.shape)
    np.save('final_images.npy',final_images)
    np.save('labels.npy',labels)
    
    
    # mnist_m_train_images = np.empty((60000,32,32,3))
    # for i in range(images.shape[0]):
    #   temp = images
    #  mnist_m_train_images[i,:,:,:] = scipy.misc.imresize(images[i,:,:,:],[32,32])

    # np.save('mnist_m_train_images.npy',mnist_m_train_images)
    # images = mnist_m_train_images
    #images = images / 127.5 - 1
    #labels = np.load (label_dir)
    print ('finished loading usps image dataset..!')
    return final_images, labels


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []
    def __call__(self, image):
        if self.maxsize == 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            return image
        if np.random.rand > 0.5:
            idx = int(np.random.rand*self.maxsize)
            tmp = copy.copy(self.images[idx])
            self.images[idx] = image
            return tmp
        else:
            return image

def load_test_data(image_path, fine_size=256):
    img = imread(image_path)
    img = scipy.misc.imresize(img, [fine_size, fine_size])
    img = img/127.5 - 1
    return img

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=37, fine_size=32, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.



def inverse_transform(images):
    return (images+1.0)*127.5

