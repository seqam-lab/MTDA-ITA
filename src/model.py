from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



from sklearn.manifold import TSNE


class MDA(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.is_pretrain = args.is_pretrain
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        self.image_reconstruction_loss_lambda = 1
        self.domain_label_reconstruction_loss_lambda = 1
        self.source_classification_loss_lambda = 1
        #self.private_encoder_lambda = 1
        #self.decoder_lambda = 1
        #self.classifier_lambda = 1
        #self.discriminator_lambda = 1
        self.dataset_dir = args.dataset_dir
        self.num_class = args.num_class
        self.base_chnl = args.base_chnl

        self.shared_encoder = encoder
        self.private_encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.discriminator = discriminator

        self.classification_loss = softmax_criterion
        self.entropy_loss = entropy_criterion
        self.entropy_loss2 = vat_loss
        self.reconstruction_loss  = mae_criterion


        self.discrimination_loss  = softmax_criterion

        self.generator_discrimination_loss = hellinger_loss
        #self.generator_discrimination_loss =  softmax_criterion


        self.reverse_label_discrimination_loss = softmax_criterion2
        self.criterionGAN = mae_criterion
        self.sce_criterion = mae_criterion
        self.cls_criterion =softmax_criterion
        self.entropy_criterion = entropy_criterion
        self.maxim_criterion = softmaxce_criterion_maximization
        self.maxim_criterion2 = softmaxce_criterion_maximization2
        self.num_domains = args.num_domains
        self.num_source_domains = args.num_source_domains

        self.svhn_dir = 'svhn'
        self.mnist_dir = 'mnist'
        self.mnist_m_dir = 'mnist_m'
        self.usps_dir = 'usps'
        self.image_size = args.fine_size
        self.config = tf.ConfigProto()



        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim num_domains')   # wat does the fox say
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc, args.num_domains))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)


    def grayscale2rgb(self, grayscale_images):
        num_sam, w, h, _ = grayscale_images.shape
        grayscale_images = grayscale_images[:, :, :, 0]
        rgb_images = np.empty ((num_sam, w, h, self.input_c_dim), dtype=np.float32)
        # print ('ssssssssssssssssss',temp.shape, all_target1_images.shape)
        rgb_images[:, :, :, 0] = grayscale_images
        rgb_images[:, :, :, 1] = rgb_images[:, :, :, 2] = rgb_images[:, :, :, 0]
        return rgb_images

    def rgb2grayscale(self, rgb_images):

        grayscale_images = np.mean(rgb_images,axis=3)
        print ("oyoyoyoyoyoyoyoyoyoyoyoyoyoyoyo",grayscale_images.shape)

        return grayscale_images

    def _build_model(self):


        self.source_image = tf.placeholder(tf.float32, [None,self.image_size,self.image_size,self.input_c_dim],'source_image')
        self.source_domain_label = tf.placeholder(tf.int32, [None],'source_domain_label')
        self.source_class_label = tf.placeholder(tf.int32, [None],'source_class_label')

        self.target_image = tf.placeholder(tf.float32, [None,self.image_size,self.image_size,self.input_c_dim],'target_image')
        self.target_domain_label = tf.placeholder(tf.int32, [None],'target_domain_label')
        self.target_class_label = tf.placeholder(tf.int32, [None],'target_class_label')

        #self.real_A = tf.placeholder(tf.float32, [None, 32, 32, 3], 'real_A')
        #self.real_B = tf.placeholder(tf.float32, [None, 32, 32, 1], 'real_B')

        #self.real_A = tf.image.resize_images (self.real_A_,[256,256])
        #self.real_B = tf.image.resize_images (self.real_B_,[256,256])
        #print self.real_A.get_shape().as_list()


        #self.A_labels = tf.placeholder(tf.uint8, [None], 'A_labels')
        #self.B_labels = tf.placeholder(tf.uint8, shape=[None])#, 'B_labels')
            #self.fake_label = tf.placeholder(tf.int64, [None], 'fake_label')
            #self.Laplacian_matrix = tf.placeholder(tf.float32, [self.batch_size,self.batch_size],'laplacian')
        #self.fake_label = (self.num_class)*tf.ones_like(self.B_labels) # = k+1




        self.source_shared_feature = self.shared_encoder(self.source_image,
            self.options,reuse=False,name="shared_encoder")
        self.source_private_feature = self.private_encoder(self.source_image,
            self.options,reuse=False,name="private_encoder")

        self.target_shared_feature = self.shared_encoder(self.target_image,
            self.options,reuse=True,name="shared_encoder")
        self.target_private_feature = self.private_encoder(self.target_image,
            self.options,reuse=True,name="private_encoder")


        #self.source_reconstructed_image = self.decoder(tf.concat([self.source_shared_feature,self.source_private_feature],axis=3),
        #        self.options,reuse=False,name="decoder")
        #self.target_reconstructed_image = self.decoder(tf.concat([self.target_shared_feature,self.target_private_feature],axis=3),
         #       self.options,reuse=True,name="decoder")

        self.source_reconstructed_image = self.decoder(self.source_shared_feature + self.source_private_feature,
                self.options,reuse=False,name="decoder")
        self.target_reconstructed_image = self.decoder(self.target_shared_feature + self.target_private_feature,
                self.options,reuse=True,name="decoder")

        self.source_classifier_logits = self.classifier(self.source_shared_feature,self.options,
                                                        reuse=False,num_class=self.num_class, name="classifier")


        self.source_discriminator_logits_shared = self.discriminator (self.source_shared_feature, self.options,
                                                                      reuse=False, num_domains=self.num_domains,
                                                         name="discriminator")
        self.source_discriminator_logits_private = self.discriminator (self.source_private_feature, self.options,
                                                                      reuse=True,num_domains=self.num_domains,
                                                                      name="discriminator")

        self.var_class = tf.get_collection ('trainable_variables',
                                            'classifier')  # Get list of the classifier's trainable variables
        self.ema = tf.train.ExponentialMovingAverage (decay=0.998)
        self.ema_op = self.ema.apply (self.var_class)

        self.target_classifier_logits = self.classifier (self.target_shared_feature, self.options,
                                                         reuse=True, num_class=self.num_class,name="classifier")
        self.target_ema_classifier_logits = self.classifier (self.target_shared_feature, self.options,
                                                             reuse=True, num_class=self.num_class,name="classifier",
                                                             getter=get_getter (self.ema))

        self.target_discriminator_logits_shared = self.discriminator (self.target_shared_feature, self.options,
                                                                      reuse=True,num_domains=self.num_domains,
                                                                      name="discriminator")
        self.target_discriminator_logits_private = self.discriminator (self.target_private_feature, self.options,
                                                                       reuse=True,num_domains=self.num_domains,
                                                                       name="discriminator")

        #self.g_loss_a2b = self.sce_criterion(logits=self.binary_B_fake, labels=tf.ones_like(self.binary_B_fake))\

        print ('hhhhhhhhhhhhhhh',self.source_reconstructed_image)
        print ('hhhhhhhhhhhhhhh', self.target_reconstructed_image)
        #print ('hhhhhhhhhhhhhhh', self.source_reconstructed_image)
        #print ('hhhhhhhhhhhhhhh', self.source_reconstructed_image)

        self.reconstruction_loss_term = self.image_reconstruction_loss_lambda * self.reconstruction_loss \
            (tf.concat([self.source_image,self.target_image],axis=0),tf.concat([self.source_reconstructed_image,
                                                                                self.target_reconstructed_image],axis=0))

        #self.shared_encoder_discriminator_loss =  self.generator_discrimination_loss (logits=tf.concat([self.source_discriminator_logits_shared,
         #                                            self.target_discriminator_logits_shared],axis=0),power=self.num_domains)

        self.shared_encoder_discriminator_loss = self.discrimination_loss (
            labels=tf.concat ([self.source_domain_label, self.target_domain_label], axis=0)
            , logits=tf.concat ([self.source_discriminator_logits_shared,
                                 self.target_discriminator_logits_shared], axis=0), N=self.num_domains)

        self.shared_encoder_classification_loss = self.classification_loss(labels=self.source_class_label,
                                                    logits=self.source_classifier_logits,N=self.num_class)

        self.shared_encoder_entropy_loss =   (self.entropy_loss(self.target_classifier_logits) +
             0.0 * self.entropy_loss2(self.target_image,self.target_classifier_logits,
                                self.shared_encoder, self.classifier, self.options, self.num_class))\
            + 0.0 * self.entropy_loss2(self.source_image,self.source_classifier_logits,self.shared_encoder,
                                 self.classifier, self.options, self.num_class)

        self.shared_encoder_reconstruction_loss = self.image_reconstruction_loss_lambda * self.reconstruction_loss \
            (tf.concat([self.source_image,self.target_image],axis=0),tf.concat([self.source_reconstructed_image,
                                                                                self.target_reconstructed_image],axis=0))


        self.private_encoder_reconstruction_loss = self.image_reconstruction_loss_lambda * self.reconstruction_loss \
            (tf.concat([self.source_image,self.target_image],axis=0),tf.concat([self.source_reconstructed_image,
                                                                                self.target_reconstructed_image],axis=0))


        #self.private_encoder_reconstruction_loss = self.image_reconstruction_loss_lambda * (self.reconstruction_loss(self.source_image,
         #           self.source_reconstructed_image)+
          #              self.reconstruction_loss(self.target_image, self.target_reconstructed_image))

        self.private_encoder_discriminator_loss = self.domain_label_reconstruction_loss_lambda * \
                                          (self.discrimination_loss (labels=tf.concat ([self.source_domain_label,
                                                                                        self.target_domain_label],
                                                                                       axis=0), logits=tf.concat (
                                              [self.source_discriminator_logits_private,
                                               self.target_discriminator_logits_private], axis=0), N=self.num_domains))
        #self.private_encoder_discriminator_loss = self.domain_label_reconstruction_loss_lambda * \
         #                                         (self.discrimination_loss(labels=self.source_domain_label,
          #          logits=self.source_discriminator_logits_private,N=self.num_domains)+
           #             self.discrimination_loss(labels=self.target_domain_label, logits=self.target_discriminator_logits_private
            #                                     , N=self.num_domains))

        if self.is_pretrain:
            self.shared_encoder_loss = self.shared_encoder_reconstruction_loss
        else:
            print ("This is not pretrain!!!!!!!!!!!!!")

            # standard multi-class discrimnator loss
            self.shared_encoder_loss =   .1 * self.shared_encoder_reconstruction_loss +   -.20 * self.shared_encoder_discriminator_loss \
                                       +  1 * self.shared_encoder_classification_loss +   .01 * self.shared_encoder_entropy_loss
            #---------------------------------------
            #
            # Hellinger discriminator loss
            #self.shared_encoder_loss = .1 * self.shared_encoder_reconstruction_loss + -2.0 * self.shared_encoder_discriminator_loss \
             #                          + 1 * self.shared_encoder_classification_loss + .01 * self.shared_encoder_entropy_loss


        if self.is_pretrain:
            self.private_encoder_loss = self.private_encoder_reconstruction_loss
        else:
            self.private_encoder_loss =  self.private_encoder_reconstruction_loss +  .1 * self.private_encoder_discriminator_loss

        self.decoder_loss =  .01 * self.reconstruction_loss \
            (tf.concat([self.source_image,self.target_image],axis=0),tf.concat([self.source_reconstructed_image,
                                                                                self.target_reconstructed_image],axis=0))


        self.classifier_label_loss =  1 * self.classification_loss(labels=self.source_class_label,
                                                    logits=self.source_classifier_logits,N=self.num_class)

                               #0.1 * self.entropy_loss(self.target_classifier_logits) + \
                               #0.0 * self.entropy_loss2 (self.target_classifier_logits,self.target_ema_classifier_logits)
        self.classifier_entropy_loss = self.shared_encoder_entropy_loss

        self.classifier_loss = .05 * self.classifier_label_loss +  0.01 * self.classifier_entropy_loss

        self.discriminator_loss_shared = self.domain_label_reconstruction_loss_lambda * \
                                     (self.discrimination_loss(labels=tf.concat([self.source_domain_label,
                                                                                 self.target_domain_label],axis=0),
                                                               logits=tf.concat([self.source_discriminator_logits_shared,
                                    self.target_discriminator_logits_shared],axis=0),N=self.num_domains))

        #self.discriminator_loss_private = self.domain_label_reconstruction_loss_lambda * \
         #                                (self.reverse_label_discrimination_loss (labels=tf.concat
          #                               ([self.source_domain_label,self.target_domain_label],
           #                                axis=0), logits=tf.concat (
            #                                 [self.source_discriminator_logits_private,
             #                                 self.target_discriminator_logits_private], axis=0), N=self.num_domains))

        self.discriminator_loss_private =  self.domain_label_reconstruction_loss_lambda * \
                                          (self.discrimination_loss (labels=tf.concat ([self.source_domain_label,
                                                                                        self.target_domain_label],
                                                                                       axis=0), logits=tf.concat (
                                              [self.source_discriminator_logits_private,
                                               self.target_discriminator_logits_private], axis=0), N=self.num_domains))
        self.discriminator_loss =  self.discriminator_loss_shared #+ 0.001 * self.discriminator_loss_private

        #self.discriminator_loss_private = self.domain_label_reconstruction_loss_lambda * \
         #                                (self.discrimination_loss (labels=self.source_domain_label,
          #                                                          logits=self.source_discriminator_logits_private,
           #                                                         N=self.num_domains) +
            #                              self.discrimination_loss (labels=self.target_domain_label,
             #                                                       logits=self.target_discriminator_logits_private,
              #                                                      N=self.num_domains))

        #self.discriminator_loss_private = self.domain_label_reconstruction_loss_lambda * \
         #                        (self.discrimination_loss (labels=self.source_domain_label,
          #                                                  logits=self.source_discriminator_logits_private,
           #                                                 N=self.num_domains) + \
            #                      self.discrimination_loss (labels=self.target_domain_label,
             #                                               logits=self.target_discriminator_logits_private,
              #                                              N=self.num_domains))


    #self.discriminator_loss = self.domain_label_reconstruction_loss_lambda * \
         #                             (self.discrimination_loss(labels=self.source_domain_label,
          #          logits=self.source_discriminator_logits_shared,N=self.num_domains)+ \
           #             self.discrimination_loss(labels=self.target_domain_label,
            #                                     logits=self.target_discriminator_logits_shared,N=self.num_domains)) + \
             #       self.domain_label_reconstruction_loss_lambda * (self.discrimination_loss(labels=self.source_domain_label,
              #      logits=self.source_discriminator_logits_private,N=self.num_domains)+ \
               #         self.discrimination_loss(labels=self.target_domain_label,
                #                                 logits=self.target_discriminator_logits_private,N=self.num_domains))

        self.test_data = tf.placeholder (tf.float32,
                                         [None, self.image_size, self.image_size,
                                          self.input_c_dim], name='test_data')
        self.test_domain_label = tf.placeholder (tf.int32,
                                                 [None], name='test_domain_label')
        self.test_class_label = tf.placeholder (tf.int32,
                                                [None], name='test_class_label')

        self.test_shared_feature = self.shared_encoder (self.test_data,
                                                        self.options, reuse=True, name="shared_encoder")
        self.test_classifier_logits = self.classifier (self.test_shared_feature, self.options, reuse=True,
                                                       num_class=self.num_class,name="classifier")
        self.test_correct_prediction = tf.equal(tf.argmax (self.test_classifier_logits, axis=1), tf.cast(self.test_class_label,tf.int64))
        self.test_acc = tf.reduce_mean (tf.cast (self.test_correct_prediction, tf.float32))
        self.test_loss = self.classification_loss (logits=self.test_classifier_logits, labels=self.test_class_label,
                                                   N=self.num_class)

        ####################################################################################################
        # ----------ema classifier--------------------------------------------------------------------
        self.test_ema_classifier_logits = self.classifier (self.test_shared_feature, self.options, reuse=True,
                                                           num_class=self.num_class,name="classifier", getter=get_getter (self.ema))
        self.test_ema_correct_prediction = tf.equal (tf.argmax (self.test_ema_classifier_logits, axis=1), tf.cast(self.test_class_label,tf.int64))
        self.test_ema_acc = tf.reduce_mean (tf.cast (self.test_ema_correct_prediction, tf.float32))

        ####################################################################################################





        self.shared_encoder_loss_sum = tf.summary.scalar("shared_encoder_loss", self.shared_encoder_loss)
        self.shared_encoder_classification_loss_sum = tf.summary.scalar("shared_encoder_classification_loss",
                                                                    self.shared_encoder_classification_loss)
        self.shared_encoder_discrimination_loss_sum = tf.summary.scalar("shared_encoder_discrimination_loss",
                                                                    self.shared_encoder_discriminator_loss/4)
        self.shared_encoder_reconstruction_loss_sum = tf.summary.scalar("shared_encoder_reconstruction_loss",
                                                                    self.shared_encoder_reconstruction_loss)
        self.shared_encoder_entropy_loss_sum = tf.summary.scalar("shared_encoder_entropy_loss",
                                                                    self.shared_encoder_entropy_loss)

        self.private_encoder_loss_sum = tf.summary.scalar("private_encoder_loss", self.private_encoder_loss)
        self.decoder_loss_sum = tf.summary.scalar("decoder_loss", 100 * self.decoder_loss)
        self.classifier_label_loss_sum = tf.summary.scalar ("classifier_label_loss", self.classifier_label_loss)
        self.classifier_entropy_loss_sum = tf.summary.scalar ("classifier_entropy_loss", self.classifier_entropy_loss)
        self.classifier_loss_sum = tf.summary.scalar("classifier_loss", self.classifier_loss)
        self.discriminator_loss_sum = tf.summary.scalar("discriminator_loss", self.discriminator_loss)
        self.discriminator_loss_shared_sum = tf.summary.scalar ("discriminator_shared_loss", self.discriminator_loss_shared)
        self.discriminator_loss_private_sum = tf.summary.scalar ("discriminator_private_loss", self.discriminator_loss_private)

        self.test_loss_sum = tf.summary.scalar("test_loss", self.test_loss/4)
        self.test_acc_sum = tf.summary.scalar("Classification Accuracy", self.test_acc)
        self.test_ema_acc_sum = tf.summary.scalar ("ema Classification Accuracy", self.test_ema_acc)


        self.test_image = tf.placeholder(tf.float32,
                                     [self.batch_size, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_image')
        self.test_image_domain_label = tf.placeholder(tf.int32,
                                     [self.batch_size], name='test_image_domain_label')
        self.test_image_shared_feature = self.shared_encoder(self.test_image, self.options, True, name="shared_encoder")
        self.test_image_private_feature = self.private_encoder(self.test_image, self.options, True, name="private_encoder")
        self.test_reconstructed_image = self.decoder(tf.concat([self.test_image_shared_feature+self.test_image_private_feature],axis=3), self.options, True, name="decoder")

        self.test_private_reconstructed_image = self.decoder (
             self.test_image_private_feature, self.options, True,
            name="decoder")
        self.test_shared_reconstructed_image = self.decoder (self.test_image_shared_feature, self.options, True,
            name="decoder")
        t_vars = tf.trainable_variables()
        self.shared_encoder_vars = [var for var in t_vars if 'shared_encoder' in var.name]
        self.private_encoder_vars = [var for var in t_vars if 'private_encoder' in var.name]
        self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]
        self.classifier_vars = [var for var in t_vars if 'classifier' in var.name]
        self.discriminator_vars = [var for var in t_vars if 'discriminator' in var.name]
        #---------------------------------------------------------------------------------
            #self.dT_vars = [var for var in t_vars if 'discriminatorB' in var.name] #target discriminator parameters

        for var in t_vars: print(var.name)

    def train(self, args):

        all_source_images, all_source_labels = load_svhn (self.svhn_dir, split='train')
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        #all_source_images, all_source_labels = load_mnist (self.mnist_dir, split='train')
        #all_target1_images, all_target1_labels = load_svhn (self.svhn_dir, split='train')
        #all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        #all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        #all_source_images, all_source_labels = load_mnist_m (self.mnist_m_dir, split='train')
        #all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        #all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        #all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        #all_source_images, all_source_labels = load_usps (self.usps_dir, split='train')
        #all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        #all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        #all_target3_images, all_target3_labels = load_mnist_m (self.mnist_m_dir, split='train')
        print ("Running the whole model")
        if all_source_images.shape[3] == 1:
            all_source_images = self.grayscale2rgb (all_source_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        if all_target3_images.shape[3] == 1:
            all_target3_images = self.grayscale2rgb (all_target3_images)


        all_images_list = [all_source_images, all_target1_images, all_target2_images, all_target3_images]
        all_class_label_list = [all_source_labels.astype (np.int32), all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32), all_target3_labels.astype (np.int32)]

        all_source_images = None
        all_target1_images = None
        all_target2_images = None
        all_target3_images = None
        all_source_labels = None
        all_target1_labels = None
        all_target2_labels = None
        all_target3_labels = None
        #all_images_list = [all_source_images, all_target3_images]
        #all_class_label_list = [all_source_labels.astype(np.int32),all_target3_labels]


        #print('source  size', all_source_images.shape)
        #print('target1 size', all_target1_images.shape)
        #print('target2  size', all_target2_images.shape)
        #print('target3  size', all_target3_images.shape)
        self.optim_shared_encoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.shared_encoder_loss, var_list=self.shared_encoder_vars)

        self.optim_private_encoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.private_encoder_loss, var_list=self.private_encoder_vars)

        self.optim_decoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.decoder_loss, var_list=self.decoder_vars)

        self.optim_classifier = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                       .minimize(self.classifier_loss, var_list=self.classifier_vars)
        self.optim_classifier = tf.group(self.optim_classifier,self.ema_op)

        self.optim_discriminator = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.discriminator_loss, var_list=self.discriminator_vars)

        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)
        self.writer = tf.summary.FileWriter ("./logs", self.sess.graph)

        if self.load(args.checkpoint_dir):
           print(" [*] Load SUCCESS")
        else:
           print(" [!] Load failed...")

        orders = []
        for i in range (self.num_domains):
            orders.append (np.arange (0, all_images_list[i].shape[0]))

        # order_source = np.arange(0, all_source_images.shape[0])
        # order_target1 = np.arange(0, all_target1_images.shape[0])
        num_epochs = 100
        #num_update = 3
        for epoch in xrange (num_epochs):
            # initialize G and D :
            #if np.mod (epoch, 5) == 0 and num_update <10:
             #   num_update += 1
            self.writer = tf.summary.FileWriter (logdir="./logs", graph=tf.get_default_graph ())
            min_len = 100000000
            for i in range (self.num_domains):
                if min_len > len (all_class_label_list[i]):
                    min_len = len (all_class_label_list[i])
            self.train_iter = min (min_len, args.train_size) // self.batch_size

            # self.train_iter= min(min(len(all_source_labels), len(all_target1_labels)), args.train_size) // self.batch_size

            print ('start training..!')
            for i in range (self.num_domains):
                np.random.shuffle (orders[i])
                all_images_list[i] = all_images_list[i][orders[i], ...]
                all_class_label_list[i] = all_class_label_list[i][orders[i], ...]

            # np.random.shuffle(order_source)
            # np.random.shuffle(order_target1)
            # all_source_images = all_source_images[order_source, ...]
            # all_target1_images = all_target1_images[order_target1, ...]
            # all_source_labels = all_source_labels[order_source, ...]
            # all_target1_labels = all_target1_labels[order_target1, ...]
            # B_images = np.array(B_images).astype(np.float32)
            # B_labels = np.array(B_labels).astype(np.float32)

            for step in range (self.train_iter):

                i = step % int (all_images_list[0].shape[0] / self.batch_size)
                # i = step % int(all_source_images.shape[0] / self.batch_size)

                shape = all_images_list[0].shape
                # print("ggggggggggggggggggggggggggggggggg",shape[1])
                mini_batch_image_list = []
                mini_batch_class_label_list = []
                mini_batch_domain_label_list = []
                for jj in range (self.num_domains):
                    mini_batch_image_list.append (all_images_list[jj][i * self.batch_size:(i + 1) * self.batch_size])
                    mini_batch_image_list[jj] = mini_batch_image_list[jj].astype (np.float32)

                    mini_batch_class_label_list.append (
                        all_class_label_list[jj][i * self.batch_size:(i + 1) * self.batch_size])
                    mini_batch_class_label_list[jj] = mini_batch_class_label_list[jj].astype (np.int32)

                    domain_label = jj * np.ones(self.batch_size,dtype=np.int32)
                    mini_batch_domain_label_list.append (domain_label)

                # print ("fake label: ", fake_label)

                src_images = mini_batch_image_list[0]
                src_class_labels = mini_batch_class_label_list[0]
                src_domain_labels = mini_batch_domain_label_list[0]

                trg_images = mini_batch_image_list[1]
                trg_class_labels = mini_batch_class_label_list[1]
                trg_domain_labels = mini_batch_domain_label_list[1]
                for jj in range (2, self.num_domains):
                    trg_images = np.concatenate ([trg_images, mini_batch_image_list[jj]], axis=0)
                    trg_class_labels = np.concatenate ([trg_class_labels, mini_batch_class_label_list[jj]], axis=0)
                    trg_domain_labels = np.concatenate ([trg_domain_labels, mini_batch_domain_label_list[jj]], axis=0)


                #print (";;;;;;;;;;;;;;;;;;;;;;;;;",trg_images.shape)
                test_classifier_logits, test_ema_classifier_logits, _, summary_shared_encoder, summary_shared_encoder_entropy,\
                summary_shared_encoder_classification, \
                summary_shared_encoder_discrimination, summary_shared_encoder_reconstruction, \
                _, summary_private_encoder, _, summary_decoder, \
                _,summary_classifier,summary_classifier_entropy, summary_classifier_label, _, summary_discriminator, summary_discriminator_shared, \
                summary_discriminator_private, \
                summary_test_classifier, summary_test_acc, summary_ema_test_acc = \
                    self.sess.run ([self.test_classifier_logits,
                                    self.test_ema_classifier_logits, self.optim_shared_encoder,
                                    self.shared_encoder_loss_sum,
                                    self.shared_encoder_entropy_loss_sum,
                                    self.shared_encoder_classification_loss_sum,
                                    self.shared_encoder_discrimination_loss_sum,
                                    self.shared_encoder_reconstruction_loss_sum,
                                    self.optim_private_encoder, self.private_encoder_loss_sum,
                                    self.optim_decoder, self.decoder_loss_sum,
                                    self.optim_classifier, self.classifier_loss_sum,
                                    self.classifier_entropy_loss_sum,self.classifier_label_loss_sum,
                                    self.optim_discriminator, self.discriminator_loss_sum,
                                    self.discriminator_loss_shared_sum, self.discriminator_loss_private_sum,
                                    self.test_loss_sum, self.test_acc_sum, self.test_ema_acc_sum],
                                   feed_dict={self.source_image: src_images,
                                              self.source_domain_label: src_domain_labels,
                                              self.source_class_label: src_class_labels,
                                              self.target_image: trg_images,
                                              self.target_domain_label: trg_domain_labels,
                                              self.target_class_label:trg_class_labels,
                                              self.test_data: trg_images,
                                              self.test_domain_label: trg_domain_labels,
                                              self.test_class_label: trg_class_labels})

                # _, summary_shared_encoder, summary_shared_encoder_classification , \
                #    summary_shared_encoder_discrimination, summary_shared_encoder_reconstruction,\
                #    _, summary_private_encoder, _, summary_decoder, \
                # _, summary_classifier, _, summary_discriminator, summary_discriminator_shared, \
                # summary_discriminator_private, \
                # summary_test_classifier, summary_test_acc = \
                #     self.sess.run ([self.optim_shared_encoder, self.shared_encoder_loss_sum,
                #      self.shared_encoder_classification_loss_sum,
                #                         self.shared_encoder_discrimination_loss_sum,
                #                         self.shared_encoder_reconstruction_loss_sum,
                #                     self.optim_private_encoder, self.private_encoder_loss_sum,
                #                     self.optim_decoder, self.decoder_loss_sum,
                #                     self.optim_classifier, self.classifier_loss_sum,
                #                     self.optim_discriminator, self.discriminator_loss_sum,
                #                     self.discriminator_loss_shared_sum, self.discriminator_loss_private_sum,
                #                     self.test_loss_sum, self.test_acc_sum],
                #                    feed_dict={self.source_image: src_images,
                #                               self.source_domain_label: src_domain_labels,
                #                               self.source_class_label: src_class_labels,
                #                               self.target_image: trg_images,
                #                               self.target_domain_label: trg_domain_labels,
                #                               self.test_data: trg_images,
                #                               self.test_domain_label: trg_domain_labels,
                #                               self.test_class_label: trg_class_labels})




                #----------------------------------------------------------------------------------------
                if np.mod (step, 500) == 2:
                    self.save (args.checkpoint_dir, step + epoch * self.train_iter)

                if np.mod (step, 50) == 2:
                    self.writer.add_summary (summary_shared_encoder, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_classification, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_discrimination, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_reconstruction, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_entropy, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_private_encoder, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_decoder, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_classifier, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_classifier_entropy, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_classifier_label, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator_shared, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator_private, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_test_classifier, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_test_acc, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_ema_test_acc, step + epoch * self.train_iter)

                    #print(" Step = ", step)
                    # trg_labels = B_labels[i * self.batch_size:(i + 1) * self.batch_size]

                    # Get the loss of target (just for checking)
                    #summary_test = self.sess.run (self.test_loss_sum,
                     #                             feed_dict={self.test_data: trg_images,
                      #                                       self.test_domain_label: trg_domain_labels,
                       #                                      self.test_class_label: trg_class_labels})
                    #self.writer.add_summary (summary_test, step + epoch * self.train_iter)

                    # if np.mod(epoch, 1) == 0 and step == 0 and epoch > num_epochs_for_source_classifier:
                    #   summary_strvalidation = self.sess.run(self.validation_accuracy_sum,
                    #                                     feed_dict={self.validation_data: B_images[1:1000,...],
                    #                                                       self.validation_label: B_labels[1:1000]})
                    # self.writer.add_summary(summary_strvalidation, step + epoch*self.train_iter)

            if epoch > 500:
                #if epoch > 5:
                print(" epoch = ", epoch)
                svhn_acc,svhn_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[1],all_class_label_list[1])
                #mnist_acc, mnist_ema_acc = self.compute_accuracy_for_each_dataset (all_images_list[3],
                 #                                                                  all_class_label_list[3])
                mnist_m_acc, mnist_m_ema_acc = self.compute_accuracy_for_each_dataset (all_images_list[2],
                                                                                       all_class_label_list[2])
                usps_acc,usps_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[3],all_class_label_list[3])

                print ("svhn_acc: ", svhn_acc)
                print ("svhn_ema_acc: ", svhn_ema_acc)

                #print ("mnist_acc: ", mnist_acc)
                #print ("mnist_ema_acc: ", mnist_ema_acc)

                print ("mnist_m_acc: ", mnist_m_acc)
                print ("mnist_m_ema_acc: ", mnist_m_ema_acc)

                print ("usps_acc: ", usps_acc)
                print ("usps_ema_acc: ", usps_ema_acc)

    def train_multiple_source(self, args):

        all_source1_images, all_source1_labels = load_svhn (self.svhn_dir, split='train')
        all_source2_images, all_source2_labels = load_usps (self.usps_dir, split='train')
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')

        # all_source_images, all_source_labels = load_mnist (self.mnist_dir, split='train')
        # all_target1_images, all_target1_labels = load_svhn (self.svhn_dir, split='train')
        # all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_usps (self.usps_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_mnist_m (self.mnist_m_dir, split='train')
        print ("Running the whole model")
        if all_source1_images.shape[3] == 1:
            all_source1_images = self.grayscale2rgb (all_source1_images)
        if all_source2_images.shape[3] == 1:
            all_source2_images = self.grayscale2rgb (all_source2_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        #if all_target3_images.shape[3] == 1:
         #   all_target3_images = self.grayscale2rgb (all_target3_images)

        all_images_list = [all_source1_images, all_source2_images, all_target1_images, all_target2_images]
        all_class_label_list = [all_source1_labels.astype (np.int32), all_source2_labels.astype (np.int32), all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32)]

        all_source_images = None
        all_target1_images = None
        all_target2_images = None
        all_target3_images = None
        all_source_labels = None
        all_target1_labels = None
        all_target2_labels = None
        all_target3_labels = None
        # all_images_list = [all_source_images, all_target3_images]
        # all_class_label_list = [all_source_labels.astype(np.int32),all_target3_labels]


        # print('source  size', all_source_images.shape)
        # print('target1 size', all_target1_images.shape)
        # print('target2  size', all_target2_images.shape)
        # print('target3  size', all_target3_images.shape)
        self.optim_shared_encoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.shared_encoder_loss, var_list=self.shared_encoder_vars)

        self.optim_private_encoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.private_encoder_loss, var_list=self.private_encoder_vars)

        self.optim_decoder = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.decoder_loss, var_list=self.decoder_vars)

        self.optim_classifier = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.classifier_loss, var_list=self.classifier_vars)
        self.optim_classifier = tf.group (self.optim_classifier, self.ema_op)

        self.optim_discriminator = tf.train.AdamOptimizer (args.lr, beta1=args.beta1) \
            .minimize (self.discriminator_loss, var_list=self.discriminator_vars)

        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)
        self.writer = tf.summary.FileWriter ("./logs", self.sess.graph)

        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        orders = []
        for i in range (self.num_domains):
            orders.append (np.arange (0, all_images_list[i].shape[0]))

        # order_source = np.arange(0, all_source_images.shape[0])
        # order_target1 = np.arange(0, all_target1_images.shape[0])
        num_epochs = 200

        svhn_acc = 0.0
        mnist_acc = 0.0
        mnist_m_acc = 0.0
        usps_acc = 0.0

        svhn_ema_acc = 0.0
        mnist_ema_acc = 0.0
        mnist_m_ema_acc = 0.0
        usps_ema_acc = 0.0

        # num_update = 3
        for epoch in xrange (num_epochs):
            # initialize G and D :
            # if np.mod (epoch, 5) == 0 and num_update <10:
            #   num_update += 1
            self.writer = tf.summary.FileWriter (logdir="./logs", graph=tf.get_default_graph ())
            min_len = 100000000
            for i in range (self.num_domains):
                if min_len > len (all_class_label_list[i]):
                    min_len = len (all_class_label_list[i])
            self.train_iter = min (min_len, args.train_size) // self.batch_size

            # self.train_iter= min(min(len(all_source_labels), len(all_target1_labels)), args.train_size) // self.batch_size

            print ('start training..!')
            for i in range (self.num_domains):
                np.random.shuffle (orders[i])
                all_images_list[i] = all_images_list[i][orders[i], ...]
                all_class_label_list[i] = all_class_label_list[i][orders[i], ...]

            # np.random.shuffle(order_source)
            # np.random.shuffle(order_target1)
            # all_source_images = all_source_images[order_source, ...]
            # all_target1_images = all_target1_images[order_target1, ...]
            # all_source_labels = all_source_labels[order_source, ...]
            # all_target1_labels = all_target1_labels[order_target1, ...]
            # B_images = np.array(B_images).astype(np.float32)
            # B_labels = np.array(B_labels).astype(np.float32)

            for step in range (self.train_iter):

                i = step % int (all_images_list[0].shape[0] / self.batch_size)
                # i = step % int(all_source_images.shape[0] / self.batch_size)

                shape = all_images_list[0].shape
                # print("ggggggggggggggggggggggggggggggggg",shape[1])
                mini_batch_image_list = []
                mini_batch_class_label_list = []
                mini_batch_domain_label_list = []
                for jj in range (self.num_domains):
                    mini_batch_image_list.append (all_images_list[jj][i * self.batch_size:(i + 1) * self.batch_size])
                    mini_batch_image_list[jj] = mini_batch_image_list[jj].astype (np.float32)

                    mini_batch_class_label_list.append (
                        all_class_label_list[jj][i * self.batch_size:(i + 1) * self.batch_size])
                    mini_batch_class_label_list[jj] = mini_batch_class_label_list[jj].astype (np.int32)

                    domain_label = jj * np.ones (self.batch_size, dtype=np.int32)
                    mini_batch_domain_label_list.append (domain_label)

                # print ("fake label: ", fake_label)

                src_images = mini_batch_image_list[0]
                src_class_labels = mini_batch_class_label_list[0]
                src_domain_labels = mini_batch_domain_label_list[0]
                for jj in range (1, self.num_source_domains):
                    src_images = np.concatenate ([src_images, mini_batch_image_list[jj]], axis=0)
                    src_class_labels = np.concatenate ([src_class_labels, mini_batch_class_label_list[jj]], axis=0)
                    src_domain_labels = np.concatenate ([src_domain_labels, mini_batch_domain_label_list[jj]], axis=0)

                trg_images = mini_batch_image_list[int(self.num_source_domains)]
                trg_class_labels = mini_batch_class_label_list[int(self.num_source_domains)]
                trg_domain_labels = mini_batch_domain_label_list[int(self.num_source_domains)]
                for jj in range (int(self.num_source_domains)+1, self.num_domains):
                    trg_images = np.concatenate ([trg_images, mini_batch_image_list[jj]], axis=0)
                    trg_class_labels = np.concatenate ([trg_class_labels, mini_batch_class_label_list[jj]], axis=0)
                    trg_domain_labels = np.concatenate ([trg_domain_labels, mini_batch_domain_label_list[jj]], axis=0)

                #print (";;;;;;;;;;;;;;;;;;;;;;;;;", trg_images.shape)
                test_classifier_logits, test_ema_classifier_logits, _, summary_shared_encoder, summary_shared_encoder_entropy, \
                summary_shared_encoder_classification, \
                summary_shared_encoder_discrimination, summary_shared_encoder_reconstruction, \
                _, summary_private_encoder, _, summary_decoder, \
                _, summary_classifier, summary_classifier_entropy, summary_classifier_label, _, summary_discriminator, summary_discriminator_shared, \
                summary_discriminator_private, \
                summary_test_classifier, summary_test_acc, summary_ema_test_acc = \
                    self.sess.run ([self.test_classifier_logits,
                                    self.test_ema_classifier_logits, self.optim_shared_encoder,
                                    self.shared_encoder_loss_sum,
                                    self.shared_encoder_entropy_loss_sum,
                                    self.shared_encoder_classification_loss_sum,
                                    self.shared_encoder_discrimination_loss_sum,
                                    self.shared_encoder_reconstruction_loss_sum,
                                    self.optim_private_encoder, self.private_encoder_loss_sum,
                                    self.optim_decoder, self.decoder_loss_sum,
                                    self.optim_classifier, self.classifier_loss_sum,
                                    self.classifier_entropy_loss_sum, self.classifier_label_loss_sum,
                                    self.optim_discriminator, self.discriminator_loss_sum,
                                    self.discriminator_loss_shared_sum, self.discriminator_loss_private_sum,
                                    self.test_loss_sum, self.test_acc_sum, self.test_ema_acc_sum],
                                   feed_dict={self.source_image: src_images,
                                              self.source_domain_label: src_domain_labels,
                                              self.source_class_label: src_class_labels,
                                              self.target_image: trg_images,
                                              self.target_domain_label: trg_domain_labels,
                                              self.target_class_label: trg_class_labels,
                                              self.test_data: trg_images,
                                              self.test_domain_label: trg_domain_labels,
                                              self.test_class_label: trg_class_labels})

                # _, summary_shared_encoder, summary_shared_encoder_classification , \
                #    summary_shared_encoder_discrimination, summary_shared_encoder_reconstruction,\
                #    _, summary_private_encoder, _, summary_decoder, \
                # _, summary_classifier, _, summary_discriminator, summary_discriminator_shared, \
                # summary_discriminator_private, \
                # summary_test_classifier, summary_test_acc = \
                #     self.sess.run ([self.optim_shared_encoder, self.shared_encoder_loss_sum,
                #      self.shared_encoder_classification_loss_sum,
                #                         self.shared_encoder_discrimination_loss_sum,
                #                         self.shared_encoder_reconstruction_loss_sum,
                #                     self.optim_private_encoder, self.private_encoder_loss_sum,
                #                     self.optim_decoder, self.decoder_loss_sum,
                #                     self.optim_classifier, self.classifier_loss_sum,
                #                     self.optim_discriminator, self.discriminator_loss_sum,
                #                     self.discriminator_loss_shared_sum, self.discriminator_loss_private_sum,
                #                     self.test_loss_sum, self.test_acc_sum],
                #                    feed_dict={self.source_image: src_images,
                #                               self.source_domain_label: src_domain_labels,
                #                               self.source_class_label: src_class_labels,
                #                               self.target_image: trg_images,
                #                               self.target_domain_label: trg_domain_labels,
                #                               self.test_data: trg_images,
                #                               self.test_domain_label: trg_domain_labels,
                #                               self.test_class_label: trg_class_labels})




                # ----------------------------------------------------------------------------------------
                if np.mod (step, 500) == 2:
                    self.save (args.checkpoint_dir, step + epoch * self.train_iter)

                if np.mod (step, 50) == 2:
                    self.writer.add_summary (summary_shared_encoder, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_classification, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_discrimination, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_reconstruction, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_shared_encoder_entropy, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_private_encoder, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_decoder, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_classifier, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_classifier_entropy, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_classifier_label, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator_shared, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_discriminator_private, step + epoch * self.train_iter)

                    self.writer.add_summary (summary_test_classifier, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_test_acc, step + epoch * self.train_iter)
                    self.writer.add_summary (summary_ema_test_acc, step + epoch * self.train_iter)

                    # self.writer.add_summary (svhn_acc, step + epoch * self.train_iter)
                    # self.writer.add_summary (svhn_ema_acc, step + epoch * self.train_iter)
                    #
                    # self.writer.add_summary (mnist_acc, step + epoch * self.train_iter)
                    # self.writer.add_summary (mnist_ema_acc, step + epoch * self.train_iter)
                    #
                    # self.writer.add_summary (mnist_m_acc, step + epoch * self.train_iter)
                    # self.writer.add_summary (mnist_m_ema_acc, step + epoch * self.train_iter)
                    #
                    # self.writer.add_summary (usps_acc, step + epoch * self.train_iter)
                    # self.writer.add_summary (usps_ema_acc, step + epoch * self.train_iter)

                    #print(" Step = ", step)


            if epoch > 10:
                        #svhn_acc,svhn_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[0],all_class_label_list[0])
                        mnist_acc,mnist_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[2],all_class_label_list[2])
                        mnist_m_acc,mnist_m_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[3],all_class_label_list[3])
                        #usps_acc,usps_ema_acc = self.compute_accuracy_for_each_dataset(all_images_list[3],all_class_label_list[3])

                        #print ("svhn_acc: ", svhn_acc)
                        #print ("svhn_ema_acc: ", svhn_ema_acc)

                        print ("mnist_acc: ", mnist_acc)
                        print ("mnist_ema_acc: ", mnist_ema_acc)

                        print ("mnist_m_acc: ", mnist_m_acc)
                        print ("mnist_m_ema_acc: ", mnist_m_ema_acc)

                        #print ("usps_acc: ", usps_acc)
                        #print ("usps_ema_acc: ", usps_ema_acc)




                        # trg_labels = B_labels[i * self.batch_size:(i + 1) * self.batch_size]

                    # Get the loss of target (just for checking)
                    # summary_test = self.sess.run (self.test_loss_sum,
                    #                             feed_dict={self.test_data: trg_images,
                    #                                       self.test_domain_label: trg_domain_labels,
                    #                                      self.test_class_label: trg_class_labels})
                    # self.writer.add_summary (summary_test, step + epoch * self.train_iter)

                    # if np.mod(epoch, 1) == 0 and step == 0 and epoch > num_epochs_for_source_classifier:
                    #   summary_strvalidation = self.sess.run(self.validation_accuracy_sum,
                    #                                     feed_dict={self.validation_data: B_images[1:1000,...],
                    #                                                       self.validation_label: B_labels[1:1000]})
                    # self.writer.add_summary(summary_strvalidation, step + epoch*self.train_iter)

    def visualization(self, args):
        all_source_images, all_source_labels = load_svhn (self.svhn_dir, split='train')
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist (self.mnist_dir, split='train')
        # all_target1_images, all_target1_labels = load_svhn (self.svhn_dir, split='train')
        # all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_usps (self.usps_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_mnist_m (self.mnist_m_dir, split='train')
        print ("Running the whole model")
        if all_source_images.shape[3] == 1:
            all_source_images = self.grayscale2rgb (all_source_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        if all_target3_images.shape[3] == 1:
            all_target3_images = self.grayscale2rgb (all_target3_images)
        # all_target1_images = self.grayscale2rgb (all_target1_images)

        # print ("lllllllllllllllllllllllllllllll",all_source_images.shape)
        # print ("claas range for svhn: [", np.min (all_source_labels), ",", np.max (all_source_labels))
        # print ("claas range for mnist: [", np.min (all_target1_labels), ",", np.max (all_target1_labels))
        # print ("claas range for mnist_m: [", np.min (all_target2_labels), ",", np.max (all_target2_labels))
        # print ("claas range for usps: [", np.min (all_target3_labels), ",", np.max (all_target3_labels))

        # print ("image range for svhn: [", np.min (all_source_images), ",", np.max (all_source_images))
        # print ("image range for mnist: [", np.min (all_target1_images), ",", np.max (all_target1_images))
        # print ("image range for mnist_m: [", np.min (all_target2_images), ",", np.max (all_target2_images))

        all_images_list = [all_source_images, all_target1_images, all_target2_images, all_target3_images]
        all_class_label_list = [all_source_labels.astype (np.int32), all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32), all_target3_labels]

        # all_images_list = [all_source_images, all_target3_images]
        # all_class_label_list = [all_source_labels.astype(np.int32),all_target3_labels]


        print('source label size', all_source_labels.shape)
        print('target3 label size', all_target3_labels.shape)


        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)


        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        orders = []
        for i in range (self.num_domains):
            orders.append (np.arange (0, all_images_list[i].shape[0]))

        # order_source = np.arange(0, all_source_images.shape[0])
        # order_target1 = np.arange(0, all_target1_images.shape[0])
        # num_update = 3

            # initialize G and D :
            # if np.mod (epoch, 5) == 0 and num_update <10:
            #   num_update += 1
        num_samples = 100


        for i in range (self.num_domains):
                np.random.shuffle (orders[i])
                all_images_list[i] = all_images_list[i][orders[i], ...]
                all_class_label_list[i] = all_class_label_list[i][orders[i], ...]

            # np.random.shuffle(order_source)
            # np.random.shuffle(order_target1)
            # all_source_images = all_source_images[order_source, ...]
            # all_target1_images = all_target1_images[order_target1, ...]
            # all_source_labels = all_source_labels[order_source, ...]
            # all_target1_labels = all_target1_labels[order_target1, ...]
            # B_images = np.array(B_images).astype(np.float32)
            # B_labels = np.array(B_labels).astype(np.float32)


        i = 0
                # i = step % int(all_source_images.shape[0] / self.batch_size)

        shape = all_images_list[0].shape
                # print("ggggggggggggggggggggggggggggggggg",shape[1])
        mini_batch_image_list = []
        mini_batch_class_label_list = []
        mini_batch_domain_label_list = []
        for jj in range (self.num_domains):
                    mini_batch_image_list.append (all_images_list[jj][i * num_samples:(i + 1) * num_samples])
                    mini_batch_image_list[jj] = mini_batch_image_list[jj].astype (np.float32)

                    mini_batch_class_label_list.append (
                        all_class_label_list[jj][i * num_samples:(i + 1) * num_samples])
                    mini_batch_class_label_list[jj] = mini_batch_class_label_list[jj].astype (np.int32)

                    domain_label = jj * np.ones (self.batch_size, dtype=np.int32)
                    mini_batch_domain_label_list.append (domain_label)

                # print ("fake label: ", fake_label)

        images = mini_batch_image_list[0]
        class_labels = mini_batch_class_label_list[0]
        domain_labels = mini_batch_domain_label_list[0]


        for jj in range (1, self.num_domains):
                    images = np.concatenate ([images, mini_batch_image_list[jj]], axis=0)
                    class_labels = np.concatenate ([class_labels, mini_batch_class_label_list[jj]], axis=0)
                    domain_labels = np.concatenate ([domain_labels, mini_batch_domain_label_list[jj]], axis=0)

        shared_features, private_features = \
                    self.sess.run ([self.source_shared_feature,self.source_private_feature],
                                   feed_dict={self.source_image: images,
                                              self.source_domain_label: domain_labels,
                                              self.source_class_label: class_labels})

        shape = images.shape
        images = images.reshape([-1,shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE(n_components=2).fit_transform (images)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1,1,0]]
        red_patch = mpatches.Patch (color='red', label='svhn')
        green_patch = mpatches.Patch (color='green', label='mnist')
        blue_patch = mpatches.Patch (color='blue', label='mnist_m')
        black_patch = mpatches.Patch (color='yellow', label='usps')



        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        plt.figure ()
        plt.legend (handles=[red_patch, green_patch, blue_patch,black_patch])
        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=0.5)
        plt.savefig ('original.png')

        colorr = [[1, 0, 0],[0, 1, 0],[0, 0, 1],[1, 1, 0],[1, 0, 1],[0, 1, 1],[1, 1, 1],[.5, .5, .5],[0, .5, .5],[.5, 0, .5]]


        plt.figure ()

        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=class_labels, alpha=0.5)
        plt.savefig ('original_class.png')
        #--------------------------------------------------------------------------
        features = np.concatenate((shared_features,private_features),axis=0)
        shape = features.shape
        features = features.reshape ([-1, shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE (n_components=2).fit_transform (features)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        red_patch = mpatches.Patch (color='red', label='source(svhn)')
        green_patch = mpatches.Patch (color='green', label='target 1(mnist)')
        blue_patch = mpatches.Patch (color='blue', label='target 2(mnist_m)')
        black_patch = mpatches.Patch (color='yellow', label='target 3(usps)')

        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        colors = np.concatenate((colors,colors),axis=0)
        plt.figure ()
        plt.legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=0.5)
        plt.savefig ('feature.png')


        #-----------------------------------------------------------------------------------

        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        a1 = [[1,0,1]]
        a2 = [[0,1,1]]
        a3 = [[1,1,1]]
        a4 = [[.5,.5,.5]]
        a5 = [[0,.5,.5]]
        a6 = [[.5,0,.5]]
        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0),
                                  np.repeat (a1, num_samples, axis=0),
                                  np.repeat (a2, num_samples, axis=0),
                                  np.repeat (a3, num_samples, axis=0),
                                  np.repeat (a4, num_samples, axis=0),
                                  np.repeat (a5, num_samples, axis=0),
                                  np.repeat (a6, num_samples, axis=0)), axis=0)
        colors = np.concatenate ((colors, colors), axis=0)
        print("fffffffffffffffffff",class_labels.shape)
        plt.figure ()

        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=np.concatenate((class_labels,class_labels),axis=0), alpha=0.5)
        plt.savefig ('feature_class.png')

        #----------------------------------------------------------------------------------------

    def visualization_2(self, args):
        all_source_images, all_source_labels = load_svhn (self.svhn_dir, split='train')
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist (self.mnist_dir, split='train')
        # all_target1_images, all_target1_labels = load_svhn (self.svhn_dir, split='train')
        # all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_usps (self.usps_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_mnist_m (self.mnist_m_dir, split='train')
        print ("Running the whole model")
        if all_source_images.shape[3] == 1:
            all_source_images = self.grayscale2rgb (all_source_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        if all_target3_images.shape[3] == 1:
            all_target3_images = self.grayscale2rgb (all_target3_images)
        # all_target1_images = self.grayscale2rgb (all_target1_images)

        # print ("lllllllllllllllllllllllllllllll",all_source_images.shape)
        # print ("claas range for svhn: [", np.min (all_source_labels), ",", np.max (all_source_labels))
        # print ("claas range for mnist: [", np.min (all_target1_labels), ",", np.max (all_target1_labels))
        # print ("claas range for mnist_m: [", np.min (all_target2_labels), ",", np.max (all_target2_labels))
        # print ("claas range for usps: [", np.min (all_target3_labels), ",", np.max (all_target3_labels))

        # print ("image range for svhn: [", np.min (all_source_images), ",", np.max (all_source_images))
        # print ("image range for mnist: [", np.min (all_target1_images), ",", np.max (all_target1_images))
        # print ("image range for mnist_m: [", np.min (all_target2_images), ",", np.max (all_target2_images))

        all_images_list = [all_source_images, all_target1_images, all_target2_images, all_target3_images]
        all_class_label_list = [all_source_labels.astype (np.int32), all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32), all_target3_labels]

        # all_images_list = [all_source_images, all_target3_images]
        # all_class_label_list = [all_source_labels.astype(np.int32),all_target3_labels]


        print('source label size', all_source_labels.shape)
        print('target3 label size', all_target3_labels.shape)

        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)

        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        orders = []
        for i in range (self.num_domains):
            orders.append (np.arange (0, all_images_list[i].shape[0]))

            # order_source = np.arange(0, all_source_images.shape[0])
            # order_target1 = np.arange(0, all_target1_images.shape[0])
            # num_update = 3

            # initialize G and D :
            # if np.mod (epoch, 5) == 0 and num_update <10:
            #   num_update += 1
        num_samples = 100

        for i in range (self.num_domains):
            np.random.shuffle (orders[i])
            all_images_list[i] = all_images_list[i][orders[i], ...]
            all_class_label_list[i] = all_class_label_list[i][orders[i], ...]

            # np.random.shuffle(order_source)
            # np.random.shuffle(order_target1)
            # all_source_images = all_source_images[order_source, ...]
            # all_target1_images = all_target1_images[order_target1, ...]
            # all_source_labels = all_source_labels[order_source, ...]
            # all_target1_labels = all_target1_labels[order_target1, ...]
            # B_images = np.array(B_images).astype(np.float32)
            # B_labels = np.array(B_labels).astype(np.float32)

        i = 0
        # i = step % int(all_source_images.shape[0] / self.batch_size)

        shape = all_images_list[0].shape
        # print("ggggggggggggggggggggggggggggggggg",shape[1])
        mini_batch_image_list = []
        mini_batch_class_label_list = []
        mini_batch_domain_label_list = []
        for jj in range (self.num_domains):
            mini_batch_image_list.append (all_images_list[jj][i * num_samples:(i + 1) * num_samples])
            mini_batch_image_list[jj] = mini_batch_image_list[jj].astype (np.float32)

            mini_batch_class_label_list.append (
                all_class_label_list[jj][i * num_samples:(i + 1) * num_samples])
            mini_batch_class_label_list[jj] = mini_batch_class_label_list[jj].astype (np.int32)

            domain_label = jj * np.ones (self.batch_size, dtype=np.int32)
            mini_batch_domain_label_list.append (domain_label)

            # print ("fake label: ", fake_label)

        images = mini_batch_image_list[0]
        class_labels = mini_batch_class_label_list[0]
        domain_labels = mini_batch_domain_label_list[0]

        for jj in range (1, self.num_domains):
            images = np.concatenate ([images, mini_batch_image_list[jj]], axis=0)
            class_labels = np.concatenate ([class_labels, mini_batch_class_label_list[jj]], axis=0)
            domain_labels = np.concatenate ([domain_labels, mini_batch_domain_label_list[jj]], axis=0)

        shared_features, private_features = \
            self.sess.run ([self.source_shared_feature, self.source_private_feature],
                           feed_dict={self.source_image: images,
                                      self.source_domain_label: domain_labels,
                                      self.source_class_label: class_labels})

        shape = images.shape
        images = images.reshape ([-1, shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE (n_components=2).fit_transform (images)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        red_patch = mpatches.Patch (color='red', label='svhn')
        green_patch = mpatches.Patch (color='green', label='mnist')
        blue_patch = mpatches.Patch (color='blue', label='mnist_m')
        black_patch = mpatches.Patch (color='yellow', label='usps')

        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        plt.figure ()
        plt.legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=1.0)
        plt.legend (loc='best')
        plt.axis ('off')
        plt.savefig ('original.png')

        colorr = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [.5, .5, .5],
                  [0, .5, .5], [.5, 0, .5]]

        plt.figure ()

        plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=class_labels, alpha=1.0, cmap='tab10')
        plt.legend (loc='best')
        plt.axis ('off')
        plt.savefig ('original_class.png')
        # --------------------------------------------------------------------------

        p1 = private_features[0:100]
        p2 = private_features[100:200]
        p3 = private_features[200:300]
        p4 = private_features[300:400]
        features = np.concatenate ((shared_features, private_features), axis=0)
        #features = np.concatenate ((shared_features, np.concatenate ((shared_features[:,0:1], private_features[:,0:7]), axis=1)), axis=0)
        #features = shared_features
        print ("shape of latent feature ", features.shape)

        #features = np.concatenate ((shared_features[:,0:4], shared_features[:,4:8]), axis=0)
        shape = features.shape
        features = features.reshape ([-1, shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE (n_components=2).fit_transform (features)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        red_patch = mpatches.Patch (color='red', label='source(svhn)')
        green_patch = mpatches.Patch (color='green', label='target 1(mnist)')
        blue_patch = mpatches.Patch (color='blue', label='target 2(mnist_m)')
        black_patch = mpatches.Patch (color='yellow', label='target 3(usps)')

        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        #colors = np.concatenate ((colors, colors), axis=0)
        plt.figure ()
        plt.legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        plt.scatter (features_embedded[0:400, 0], features_embedded[0:400, 1], c=colors, alpha=0.5)
        plt.scatter (features_embedded[400:800, 0]-3, features_embedded[400:800, 1]-3, marker='^', c=colors, alpha=0.5)
        plt.axis ('off')
        plt.savefig ('feature.png')

        # -----------------------------------------------------------------------------------

        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        a1 = [[1, 0, 1]]
        a2 = [[0, 1, 1]]
        a3 = [[1, 1, 1]]
        a4 = [[.5, .5, .5]]
        a5 = [[0, .5, .5]]
        a6 = [[.5, 0, .5]]
        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0),
                                  np.repeat (a1, num_samples, axis=0),
                                  np.repeat (a2, num_samples, axis=0),
                                  np.repeat (a3, num_samples, axis=0),
                                  np.repeat (a4, num_samples, axis=0),
                                  np.repeat (a5, num_samples, axis=0),
                                  np.repeat (a6, num_samples, axis=0)), axis=0)
        colors = np.concatenate ((colors, colors), axis=0)
        print("fffffffffffffffffff", class_labels.shape)
        plt.figure ()

        plt.scatter (features_embedded[0:400, 0], features_embedded[0:400, 1], c=class_labels, alpha=1.0, cmap='tab10')
        plt.scatter (features_embedded[400:800, 0]-3, features_embedded[400:800, 1]-3, marker='^', c=class_labels, alpha=1.0, cmap='tab10')

        #plt.scatter (features_embedded[:, 0], features_embedded[:, 1],
         #            c=np.concatenate ((class_labels, class_labels), axis=0), alpha=0.5)
        plt.legend(loc='best')
        plt.axis ('off')
        plt.savefig ('feature_class.png')

        # ----------------------------------------------------------------------------------------

    def visualization_3(self, args):
        all_source_images, all_source_labels = load_svhn (self.svhn_dir, split='train')
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist (self.mnist_dir, split='train')
        # all_target1_images, all_target1_labels = load_svhn (self.svhn_dir, split='train')
        # all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_mnist_m (self.mnist_m_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        # all_source_images, all_source_labels = load_usps (self.usps_dir, split='train')
        # all_target1_images, all_target1_labels = load_mnist (self.mnist_dir, split='train')
        # all_target2_images, all_target2_labels = load_svhn (self.svhn_dir, split='train')
        # all_target3_images, all_target3_labels = load_mnist_m (self.mnist_m_dir, split='train')
        print ("Running the whole model")
        if all_source_images.shape[3] == 1:
            all_source_images = self.grayscale2rgb (all_source_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        if all_target3_images.shape[3] == 1:
            all_target3_images = self.grayscale2rgb (all_target3_images)
        # all_target1_images = self.grayscale2rgb (all_target1_images)

        # print ("lllllllllllllllllllllllllllllll",all_source_images.shape)
        # print ("claas range for svhn: [", np.min (all_source_labels), ",", np.max (all_source_labels))
        # print ("claas range for mnist: [", np.min (all_target1_labels), ",", np.max (all_target1_labels))
        # print ("claas range for mnist_m: [", np.min (all_target2_labels), ",", np.max (all_target2_labels))
        # print ("claas range for usps: [", np.min (all_target3_labels), ",", np.max (all_target3_labels))

        # print ("image range for svhn: [", np.min (all_source_images), ",", np.max (all_source_images))
        # print ("image range for mnist: [", np.min (all_target1_images), ",", np.max (all_target1_images))
        # print ("image range for mnist_m: [", np.min (all_target2_images), ",", np.max (all_target2_images))

        all_images_list = [all_source_images, all_target1_images, all_target2_images, all_target3_images]
        all_class_label_list = [all_source_labels.astype (np.int32), all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32), all_target3_labels]

        # all_images_list = [all_source_images, all_target3_images]
        # all_class_label_list = [all_source_labels.astype(np.int32),all_target3_labels]


        print('source label size', all_source_labels.shape)
        print('target3 label size', all_target3_labels.shape)

        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)

        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        orders = []
        for i in range (self.num_domains):
            orders.append (np.arange (0, all_images_list[i].shape[0]))

            # order_source = np.arange(0, all_source_images.shape[0])
            # order_target1 = np.arange(0, all_target1_images.shape[0])
            # num_update = 3

            # initialize G and D :
            # if np.mod (epoch, 5) == 0 and num_update <10:
            #   num_update += 1
        num_samples = 100

        for i in range (self.num_domains):
            np.random.shuffle (orders[i])
            all_images_list[i] = all_images_list[i][orders[i], ...]
            all_class_label_list[i] = all_class_label_list[i][orders[i], ...]

            # np.random.shuffle(order_source)
            # np.random.shuffle(order_target1)
            # all_source_images = all_source_images[order_source, ...]
            # all_target1_images = all_target1_images[order_target1, ...]
            # all_source_labels = all_source_labels[order_source, ...]
            # all_target1_labels = all_target1_labels[order_target1, ...]
            # B_images = np.array(B_images).astype(np.float32)
            # B_labels = np.array(B_labels).astype(np.float32)

        i = 0
        # i = step % int(all_source_images.shape[0] / self.batch_size)

        shape = all_images_list[0].shape
        # print("ggggggggggggggggggggggggggggggggg",shape[1])
        mini_batch_image_list = []
        mini_batch_class_label_list = []
        mini_batch_domain_label_list = []
        for jj in range (self.num_domains):
            mini_batch_image_list.append (all_images_list[jj][i * num_samples:(i + 1) * num_samples])
            mini_batch_image_list[jj] = mini_batch_image_list[jj].astype (np.float32)

            mini_batch_class_label_list.append (
                all_class_label_list[jj][i * num_samples:(i + 1) * num_samples])
            mini_batch_class_label_list[jj] = mini_batch_class_label_list[jj].astype (np.int32)

            domain_label = jj * np.ones (self.batch_size, dtype=np.int32)
            mini_batch_domain_label_list.append (domain_label)

            # print ("fake label: ", fake_label)

        images = mini_batch_image_list[0]
        class_labels = mini_batch_class_label_list[0]
        domain_labels = mini_batch_domain_label_list[0]

        for jj in range (1, self.num_domains):
            images = np.concatenate ([images, mini_batch_image_list[jj]], axis=0)
            class_labels = np.concatenate ([class_labels, mini_batch_class_label_list[jj]], axis=0)
            domain_labels = np.concatenate ([domain_labels, mini_batch_domain_label_list[jj]], axis=0)

        shared_features, private_features = \
            self.sess.run ([self.source_shared_feature, self.source_private_feature],
                           feed_dict={self.source_image: images,
                                      self.source_domain_label: domain_labels,
                                      self.source_class_label: class_labels})

        shape = images.shape
        images = images.reshape ([-1, shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE (n_components=2).fit_transform (images)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        red_patch = mpatches.Patch (color='red', label='svhn')
        green_patch = mpatches.Patch (color='green', label='mnist')
        blue_patch = mpatches.Patch (color='blue', label='mnist_m')
        black_patch = mpatches.Patch (color='yellow', label='usps')

        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        fig, axs = plt.subplots (2, 2, figsize=(64, 64))
        axs = axs.ravel ()
        axs[0].legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        axs[0].scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=1.0)
        # plt.figure ()
        # plt.legend (handles=[red_patch, green_patch, blue_patch,black_patch])
        # plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=0.5)
        # original_plot = PdfPages('original.pdf')
        # plt.savefig(original_plot, format='pdf')
        # plt.show()
        colorr = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [.5, .5, .5],
                  [0, .5, .5], [.5, 0, .5]]

        # plt.figure ()
        axs[2].scatter (features_embedded[:, 0], features_embedded[:, 1], c=class_labels, alpha=1.0, cmap='tab10')
        axs[2].legend ()
        # plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=class_labels, alpha=0.5)
        # plt.savefig ('original_class.png')
        # --------------------------------------------------------------------------
        features = np.concatenate ((shared_features, private_features), axis=0)
        shape = features.shape
        features = features.reshape ([-1, shape[1] * shape[2] * shape[3]])

        features_embedded = TSNE (n_components=2).fit_transform (features)
        print ("shape of embeded feature ", features_embedded.shape)
        # TSNE Visualization##############################
        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        red_patch = mpatches.Patch (color='red', label='svhn')
        green_patch = mpatches.Patch (color='green', label='mnist')
        blue_patch = mpatches.Patch (color='blue', label='mnist_m')
        black_patch = mpatches.Patch (color='yellow', label='usps')

        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0)), axis=0)
        colors = np.concatenate ((colors, colors), axis=0)
        # plt.figure ()
        # plt.legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        # plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=colors, alpha=0.5)
        # plt.savefig ('feature.png')

        axs[1].legend (handles=[red_patch, green_patch, blue_patch, black_patch])
        axs[1].scatter (features_embedded[:400, 0], features_embedded[:400, 1], c=colors, alpha=1.0, marker="o")
        axs[1].scatter (features_embedded[400:, 0], features_embedded[400:, 1], c=colors, alpha=1.0, marker="^")
        # -----------------------------------------------------------------------------------

        red = [[1, 0, 0]]
        green = [[0, 1, 0]]
        blue = [[0, 0, 1]]
        black = [[1, 1, 0]]
        a1 = [[1, 0, 1]]
        a2 = [[0, 1, 1]]
        a3 = [[1, 1, 1]]
        a4 = [[.5, .5, .5]]
        a5 = [[0, .5, .5]]
        a6 = [[.5, 0, .5]]
        colors = np.concatenate ((np.repeat (red, num_samples, axis=0),
                                  np.repeat (green, num_samples, axis=0),
                                  np.repeat (blue, num_samples, axis=0),
                                  np.repeat (black, num_samples, axis=0),
                                  np.repeat (a1, num_samples, axis=0),
                                  np.repeat (a2, num_samples, axis=0),
                                  np.repeat (a3, num_samples, axis=0),
                                  np.repeat (a4, num_samples, axis=0),
                                  np.repeat (a5, num_samples, axis=0),
                                  np.repeat (a6, num_samples, axis=0)), axis=0)
        colors = np.concatenate ((colors, colors), axis=0)
        print("fffffffffffffffffff", class_labels.shape)
        # plt.figure ()

        # plt.scatter (features_embedded[:, 0], features_embedded[:, 1], c=np.concatenate((class_labels,class_labels),axis=0), alpha=0.5)
        # plt.savefig ('feature_class.png')
        # axs[3].scatter (features_embedded[:400, 0], features_embedded[:400, 1],
        #                c=np.concatenate ((class_labels, class_labels), axis=0), alpha=0.5, cmap='tab10', marker='o')
        axs[3].scatter (features_embedded[:400, 0], features_embedded[:400, 1], c=class_labels, alpha=1.0, cmap='tab10',
                        marker='o')
        axs[3].scatter (features_embedded[400:, 0], features_embedded[400:, 1], c=class_labels, alpha=1.0, cmap='tab10',
                        marker='^')
        axs[3].legend ()
        plt.show ()
        # ----------------------------------------------------------------------------------------
    def compute_accuracy(self, args):
        # load svhn dataset
        all_source_images, all_source_labels = load_svhn (self.svhn_dir,
                                                          split='train')  # for now source is svhn
        all_target1_images, all_target1_labels = load_mnist (self.mnist_dir,
                                                             split='train')  # ant target is mnist
        all_target2_images, all_target2_labels = load_mnist_m (self.mnist_m_dir, split='train')
        all_target3_images, all_target3_labels = load_usps (self.usps_dir, split='train')

        if all_source_images.shape[3] == 1:
            all_source_images = self.grayscale2rgb (all_source_images)
        if all_target1_images.shape[3] == 1:
            all_target1_images = self.grayscale2rgb (all_target1_images)
        if all_target2_images.shape[3] == 1:
            all_target2_images = self.grayscale2rgb (all_target2_images)
        if all_target3_images.shape[3] == 1:
            all_target3_images = self.grayscale2rgb (all_target3_images)

        all_images_list = [all_source_images, all_target1_images, all_target2_images,
                           all_target3_images]
        all_class_label_list = [all_source_labels.astype (np.int32),
                                all_target1_labels.astype (np.int32)
            , all_target2_labels.astype (np.int32), all_target3_labels.astype (np.int32)]

        print ("Running the whole model")
        # all_target1_images = self.grayscale2rgb (all_target1_images)
        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)
        # self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        order_source = np.arange (0, all_source_images.shape[0])
        order_target1 = np.arange (0, all_target1_images.shape[0])
        order_target2 = np.arange (0, all_target2_images.shape[0])
        order_target3 = np.arange (0, all_target3_images.shape[0])


        # initialize G and D :

        # self.writer = tf.summary.FileWriter(logdir="./logs", graph=tf.get_default_graph())


        print ('start testing..!')

        np.random.shuffle (order_source)
        np.random.shuffle (order_target1)
        np.random.shuffle (order_target2)
        np.random.shuffle (order_target3)

        all_source_images = all_source_images[order_source, ...].astype (np.float32)
        all_target1_images = all_target1_images[order_target1, ...].astype (np.float32)
        all_target2_images = all_target2_images[order_target2, ...].astype (np.float32)
        all_target3_images = all_target3_images[order_target3, ...].astype (np.float32)

        all_source_labels = all_source_labels[order_source, ...].astype (np.int32)
        all_target1_labels = all_target1_labels[order_target1, ...].astype (np.int32)
        all_target2_labels = all_target2_labels[order_target2, ...].astype (np.int32)
        all_target3_labels = all_target3_labels[order_target3, ...].astype (np.int32)

        source_acc = 0.0
        source_ema_acc = 0.0
        target1_acc = 0.0
        target2_acc = 0.0
        self.train_iter = len (all_source_labels) // self.batch_size
        for step in range (self.train_iter):
            # print ("step: ", step)
            i = step % int (all_source_images.shape[0] / self.batch_size)

            # train the model for source domain S

            src_images = all_source_images[i * self.batch_size:(i + 1) * self.batch_size]

            src_labels = all_source_labels[i * self.batch_size:(i + 1) * self.batch_size]

            [acc, ema_acc] = self.sess.run ([self.test_acc, self.test_ema_acc],
                                            feed_dict={self.test_data: src_images,
                                                       self.test_class_label: src_labels})
            source_acc += acc
            source_ema_acc += ema_acc
        source_acc = source_acc / self.train_iter
        source_ema_acc = source_ema_acc / self.train_iter
        print ("classification accuracy of source samples:   ", source_acc)
        print ("ema classification accuracy of source samples:   ", source_ema_acc)

        target1_acc = 0.0
        target1_ema_acc = 0.0
        self.train_iter = len (all_target1_labels) // self.batch_size
        for step in range (self.train_iter):
            # print ("step: ", step)
            i = step % int (all_target1_images.shape[0] / self.batch_size)

            # train the model for source domain S

            trg_images = all_target1_images[i * self.batch_size:(i + 1) * self.batch_size]

            trg_labels = all_target1_labels[i * self.batch_size:(i + 1) * self.batch_size]

            [acc, ema_acc] = self.sess.run ([self.test_acc, self.test_ema_acc],
                                            feed_dict={self.test_data: trg_images,
                                                       self.test_class_label: trg_labels})
            target1_acc += acc
            target1_ema_acc += ema_acc
        target1_acc = target1_acc / self.train_iter
        target1_ema_acc = target1_ema_acc / self.train_iter
        print ("classification accuracy of target 1 samples:   ", target1_acc)
        print ("ema classification accuracy of target 1 samples:   ", target1_ema_acc)

        target2_acc = 0.0
        target2_ema_acc = 0.0
        self.train_iter = len (all_target2_labels) // self.batch_size
        for step in range (self.train_iter):
            # print ("step: ", step)
            i = step % int (all_target2_images.shape[0] / self.batch_size)

            # train the model for source domain S

            trg_images = all_target2_images[i * self.batch_size:(i + 1) * self.batch_size]

            trg_labels = all_target2_labels[i * self.batch_size:(i + 1) * self.batch_size]

            [acc, ema_acc] = self.sess.run ([self.test_acc, self.test_ema_acc],
                                            feed_dict={self.test_data: trg_images,
                                                       self.test_class_label: trg_labels})
            target2_acc += acc
            target2_ema_acc += ema_acc
        target2_acc = target2_acc / self.train_iter
        target2_ema_acc = target2_ema_acc / self.train_iter
        print ("classification accuracy of target 2 samples:   ", target2_acc)
        print ("ema classification accuracy of target 2 samples:   ", target2_ema_acc)

        target3_acc = 0.0
        target3_ema_acc = 0.0
        self.train_iter = len (all_target3_labels) // self.batch_size
        for step in range (self.train_iter):
            # print ("step: ", step)
            i = step % int (all_target3_images.shape[0] / self.batch_size)

            # train the model for source domain S

            trg_images = all_target3_images[i * self.batch_size:(i + 1) * self.batch_size]

            trg_labels = all_target3_labels[i * self.batch_size:(i + 1) * self.batch_size]

            [acc, ema_acc] = self.sess.run ([self.test_acc, self.test_ema_acc],
                                            feed_dict={self.test_data: trg_images,
                                                       self.test_class_label: trg_labels})
            target3_acc += acc
            target3_ema_acc += ema_acc
        target3_acc = target3_acc / self.train_iter
        target3_ema_acc = target3_ema_acc / self.train_iter
        print ("classification accuracy of target 3 samples:   ", target3_acc)
        print ("ema classification accuracy of target 3 samples:   ", target3_ema_acc)





    def compute_accuracy_for_each_dataset(self, images, labels):
        # load svhn dataset

        source_acc = 0.0
        source_ema_acc = 0.0
        target1_acc = 0.0
        target2_acc = 0.0
        self.train_iter = len (labels) // self.batch_size
        for step in range (self.train_iter):
            # print ("step: ", step)
            i = step % int (images.shape[0] / self.batch_size)

            # train the model for source domain S

            src_images = images[i * self.batch_size:(i + 1) * self.batch_size]

            src_labels = labels[i * self.batch_size:(i + 1) * self.batch_size]

            [acc, ema_acc] = self.sess.run ([self.test_acc, self.test_ema_acc],
                                            feed_dict={self.test_data: src_images,
                                                       self.test_class_label: src_labels})
            source_acc += acc
            source_ema_acc += ema_acc
        source_acc = source_acc / self.train_iter
        source_ema_acc = source_ema_acc / self.train_iter

        return source_acc, source_ema_acc



    def generate_reconstructed_images(self, args):

        A_images, A_labels = load_svhn (self.svhn_dir, split='train')  # for now source is svhn
        B_images, B_labels = load_mnist (self.mnist_dir, split='train')
        C_images, C_labels = load_mnist_m (self.mnist_m_dir, split='train')
        D_images, D_labels = load_usps (self.usps_dir, split='train')

        # print(A_images.shape)
        # print(B_images.shape)
        init_op = tf.global_variables_initializer ()
        self.sess.run (init_op)

        num_samples = 50
        sample_index = np.random.permutation (5000)
        A_images = A_images[sample_index[100:100 + num_samples]]
        B_images = B_images[sample_index[100:100 + num_samples]]
        C_images = C_images[sample_index[100:100 + num_samples]]
        D_images = D_images[sample_index[100:100 + num_samples]]
        colored_B_images = np.tile (B_images, (1, 1, 3))
        colored_D_images = np.tile (D_images, (1, 1, 3))
        # A_images = A_images[1000:1000 + num_samples]
        # B_images = B_images[1000:1000 + num_samples]
        # A_images = tf.image.resize_images (A_images,[256,256])
        # B_images = tf.image.resize_images (B_images,[256,256])
        A_images = np.array (A_images).astype (np.float32)
        B_images = np.array (B_images).astype (np.float32)
        C_images = np.array (C_images).astype (np.float32)
        D_images = np.array (D_images).astype (np.float32)
        if self.load (args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        [reconstructed_A, reconstructed_A_shared, reconstructed_A_private] = self.sess.run (
            [self.test_reconstructed_image, self.test_shared_reconstructed_image,
             self.test_private_reconstructed_image],
            feed_dict={self.test_image: A_images})
        print ("size source:", A_images.shape)
        print ("size reconst source", reconstructed_A.shape)
        # colored_fake_B = np.tile(fake_B,(1,1,3))
        save_images (np.concatenate ((A_images, reconstructed_A), axis=2), [num_samples, 1],
                     './{}/svhn_{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((A_images, reconstructed_A_private), axis=2), [num_samples, 1],
                     './{}/svhn_private{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((A_images, reconstructed_A_shared), axis=2), [num_samples, 1],
                     './{}/svhn_shared{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))

        [reconstructed_B, reconstructed_B_shared, reconstructed_B_private] = self.sess.run (
            [self.test_reconstructed_image, self.test_shared_reconstructed_image,
             self.test_private_reconstructed_image],
            feed_dict={self.test_image: colored_B_images})
        save_images (np.concatenate ((colored_B_images, reconstructed_B), axis=2), [num_samples, 1],
                     './{}/mnist_{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((colored_B_images, reconstructed_B_shared), axis=2), [num_samples, 1],
                     './{}/mnist_shared{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((colored_B_images, reconstructed_B_private), axis=2), [num_samples, 1],
                     './{}/mnist_private{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))


        [reconstructed_C, reconstructed_C_shared, reconstructed_C_private] = self.sess.run (
            [self.test_reconstructed_image, self.test_shared_reconstructed_image,
             self.test_private_reconstructed_image],
            feed_dict={self.test_image: C_images})
        save_images (np.concatenate ((C_images, reconstructed_C), axis=2), [num_samples, 1],
                     './{}/mnist_m_{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((C_images, reconstructed_C_shared), axis=2), [num_samples, 1],
                     './{}/mnist_m_shared{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((C_images, reconstructed_C_private), axis=2), [num_samples, 1],
                     './{}/mnist_m_private{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))

        [reconstructed_D, reconstructed_D_shared, reconstructed_D_private] = self.sess.run (
            [self.test_reconstructed_image, self.test_shared_reconstructed_image,
             self.test_private_reconstructed_image],
            feed_dict={self.test_image: colored_D_images})
        save_images (np.concatenate ((colored_D_images, reconstructed_D), axis=2), [num_samples, 1],
                     './{}/mnist_{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((colored_D_images, reconstructed_D_shared), axis=2), [num_samples, 1],
                     './{}/mnist_shared{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))
        save_images (np.concatenate ((colored_D_images, reconstructed_D_private), axis=2), [num_samples, 1],
                     './{}/mnist_private{:02d}_{:04d}.jpg'.format (args.sample_dir, 1, 1))


    ##########################################################################################


    def save(self, checkpoint_dir, step):
        model_name = "dad.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    def new_sample_model(self, args):


        A_images, A_labels = load_svhn(self.svhn_dir, split='train') # for now source is svhn
        B_images, B_labels = load_mnist(self.mnist_dir, split='train')
        print(A_images.shape)
        print(B_images.shape)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        idx = range(1,11)
        num_samples = 50
        sample_index = np.random.permutation(40000)
        A_images = A_images[sample_index[100:100+num_samples]]
        B_images = B_images[sample_index[100:100+num_samples]]
        colored_B_images = np.tile(B_images,(1,1,3))
        #A_images = A_images[1000:1000 + num_samples]
        #B_images = B_images[1000:1000 + num_samples]
        #A_images = tf.image.resize_images (A_images,[256,256])
        #B_images = tf.image.resize_images (B_images,[256,256])
        A_images = np.array(A_images).astype(np.float32)
        B_images = np.array(B_images).astype(np.float32)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        fake_A, fake_B = self.sess.run(
            [self.testA, self.testB],
            feed_dict={self.test_A: A_images, self.test_B: B_images})
        #colored_fake_B = np.tile(fake_B,(1,1,3))
        save_images(np.concatenate((colored_B_images, fake_A),axis=2), [num_samples, 1],
                    './{}/Aa2_{:02d}_{:04d}.jpg'.format(args.sample_dir, 1, 1))
        save_images(np.concatenate((A_images,fake_B),axis=2), [num_samples, 1],
                    './{}/Bb2_{:02d}_{:04d}.jpg'.format(args.sample_dir, 1, 1))
    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testA'))
        dataB = glob('./datasets/{}/*.jpg'.format(self.dataset_dir+'/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = zip(dataA[:self.batch_size], dataB[:self.batch_size])
        sample_images = [load_data(batch_file, False, True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
