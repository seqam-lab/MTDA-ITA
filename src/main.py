import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import MDA

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='svhn2mnist', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=32, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=32, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=16, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=8, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--num_domains', dest='num_domains', type=int, default=4, help='# domains')
parser.add_argument('--num_source_domains', dest='num_source_domains', type=int, default=2, help='# source domains')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--is_pretrain', dest='is_pretrain', default=False, help='pretraining')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=10, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--num_class', dest='num_class', type=int, default=10, help='Number of classes in source and target domains')
parser.add_argument('--base_chnl', dest='base_chnl', type=int, default=16, help='Basic number of channels for the network')


args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:

        args.is_pretrain = False
        model = MDA(sess, args)
        #model.generate_reconstructed_images(args)
        #model.pretrain_private_encoder_shared_encoder_decoder(args)

        model.train(args) #if args.phase == 'train' \
        #model.train_multiple_source(args)  # if args.phase == 'train' \
        #model.compute_accuracy(args)
        #model.visualization_2(args)
        #model.visualization(args)
    #model.new_sample_model(args)
    #with tf.Session () as sess:

     #   args.is_pretrain = False
      #  model = MDA (sess, args)

        #model.pretrain_private_encoder_shared_encoder_decoder (args)
        # args.is_pretrain = False
        #model.train(args) #if args.phase == 'train' \
if __name__ == '__main__':
    tf.app.run()
