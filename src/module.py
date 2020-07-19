from __future__ import division
import tensorflow as tf
from ops import *
from utils import *


def discriminator_binary(image, options, num_class= 2,reuse=False, name="discriminator_binary"):

    if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
        image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim*2, stride=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*4, stride=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (16 x 16 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (16 x 16 x 1)
        net_shape = h4.get_shape().as_list()
        a1 = tf.reshape(h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense(a1, num_class, activation=None)

        return logits


def classifier_source(image, options, num_class= 10,reuse=False, name="classifier_source"):

    if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
        image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim*2, stride=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*4, stride=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (16 x 16 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (16 x 16 x 1)
        net_shape = h4.get_shape().as_list()
        a1 = tf.reshape(h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense(a1, num_class, activation=None)

        return logits
def discriminator_source(image, options, num_class= 11,reuse=False, name="discriminator_source"):

    if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
        image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim*2, stride=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*4, stride=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (16 x 16 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (16 x 16 x 1)
        net_shape = h4.get_shape().as_list()
        a1 = tf.reshape(h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense(a1, num_class, activation=None)

        return logits


def discriminator_target(image, options, num_class= 10,reuse=False, name="discriminator_target"):

    if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
        image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(batch_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(batch_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(batch_norm(conv2d(h2, options.df_dim*8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        net_shape = h4.get_shape().as_list()
        a1 = tf.reshape(h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense(a1, num_class + 1 , activation=None)

        return logits

def generator_resnet(image, options, reuse=False, name="generator"):
    if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
        image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        if ema_var:
            print("ema is valid")
            return ema_var
        else:
            print ("ema is not valid")
            return var
        #return ema_var if ema_var else var
    return ema_getter
def encoder(image, options, reuse=False, name="encoder"):
    #if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
     #   image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        feature = residule_block(r4, options.gf_dim*4, name='g_r5')
        print ("shape of latent feature issssssssssssssss:     ", feature)
        #r6 = residule_block(r5, options.gf_dim*4, name='g_r6')
        #r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        #r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        #r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        #d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        #d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        #d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        #d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        #d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        #pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return feature

def decoder(feature, options, reuse=False, name="decoder"):
    #if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
     #   image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            #print ("size of xxxxxxxxx",x)
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            #print ("size of yyyyyyyyyyy", y)
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        #c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        #c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        #c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        #c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        #r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        #r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        #r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        #r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        #r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(feature, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        image = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return image


def discriminator(image, options, num_domains=3, reuse=False, name="discriminator"):
    with tf.variable_scope (name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope ().reuse_variables ()
        else:
            assert tf.get_variable_scope ().reuse == False
        
        h0 = lrelu (conv2d (image, options.df_dim, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu (batch_norm (conv2d (h0, options.df_dim * 2, stride=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu (batch_norm (conv2d (h1, options.df_dim * 4, stride=1, name='d_h2_conv'), 'd_bn2'))
        # h2 is (16 x 16 x self.df_dim*4)
        h3 = lrelu (batch_norm (conv2d (h2, options.df_dim * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (16 x 16 x self.df_dim*8)
        h4 = conv2d (h3, 1, stride=1, name='d_h3_pred')
        # h4 is (16 x 16 x 1)
        net_shape = h4.get_shape ().as_list ()
        a1 = tf.reshape (h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        
        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense (a1, num_domains, activation=None)
        
        return logits


def classifier(image, options, num_class=10, reuse=False, name="classifier", getter=None):
    with tf.variable_scope (name, custom_getter=getter):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope ().reuse_variables ()
        else:
            assert tf.get_variable_scope ().reuse == False
        
        h0 = lrelu (conv2d (image, options.df_dim, name='d_h0_conv'))
        #        h0 is (16 x 16 x self.df_dim)
        h1 = lrelu (batch_norm (conv2d (h0, options.gf_dim * 2, stride=1, name='d_h1_conv'), 'd_bn1'))
         #       h1 is (16 x 16 x self.df_dim*2)
        h2 = lrelu (batch_norm (conv2d (h1, options.gf_dim * 4, stride=1, name='d_h2_conv'), 'd_bn2'))
          #      h2 is (16 x 16 x self.df_dim*4)
        h3 = lrelu (batch_norm (conv2d (h2, options.gf_dim * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
           #     h3 is (16 x 16 x self.df_dim*8)
        h4 = conv2d (h3, 1, stride=1, name='d_h3_pred')
            #    h4 is (16 x 16 x 1)
        net_shape = h4.get_shape ().as_list ()
        a1 = tf.reshape (h4, [-1, net_shape[1] * net_shape[2] * net_shape[3]])


        #h0 = lrelu (conv2d (image, options.df_dim, name='d_h0_conv'))
            #        h0 is (16 x 16 x self.df_dim)
        #h1 = lrelu (batch_norm (conv2d (h0, options.df_dim, name='d_h1_conv'), 'd_bn1'))
            #       h1 is (16 x 16 x self.df_dim*2)
        #h2 = lrelu (batch_norm (conv2d (h1, options.df_dim, name='d_h2_conv'), 'd_bn2'))
            #      h2 is (16 x 16 x self.df_dim*4)
        #h3 = lrelu (batch_norm (conv2d (h2, options.df_dim, name='d_h3_conv'), 'd_bn3'))
            #     h3 is (16 x 16 x self.df_dim*8)
        # h4 = conv2d (h3, 1, stride=1, name='d_h3_pred')
            #    h4 is (16 x 16 x 1)
        #net_shape = h1.get_shape ().as_list ()
        #a1 = tf.reshape (h1, [-1, net_shape[1] * net_shape[2] * net_shape[3]])



        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense (a1, num_class, activation=None)
        
        return logits
def classifier_old(feature, options, reuse=False, num_class = 10, name="classifier"):
    #if image.get_shape()[3] == 1:
        # For mnist dataset, replicate the gray scale image 3 times.
     #   image = tf.image.grayscale_to_rgb(image)

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        #c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        #c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        #c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        #c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        #r1 = residule_block(c3, options.gf_dim*4, name='g_r1')
        #r2 = residule_block(r1, options.gf_dim*4, name='g_r2')
        #r3 = residule_block(r2, options.gf_dim*4, name='g_r3')
        #r4 = residule_block(r3, options.gf_dim*4, name='g_r4')
        #r5 = residule_block(r4, options.gf_dim*4, name='g_r5')
        r6 = residule_block(feature, options.gf_dim*4, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4, name='g_r9')
        net_shape = r9.get_shape ().as_list ()
        a1 = tf.reshape (r9, [-1, net_shape[1] * net_shape[2] * net_shape[3]])

        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense (a1, num_class, activation=None)
        
        #d1 = deconv2d(r9, options.gf_dim*2, 3, 2, name='g_d1_dc')
        #d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        #d2 = deconv2d(d1, options.gf_dim, 3, 2, name='g_d2_dc')
        #d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        #d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        #image = tf.nn.tanh(conv2d(d2, options.num_domains, 7, 1, padding='VALID', name='g_pred_c'))

        return logits
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mae_criterion(in_, target):
    #return tf.reduce_mean((in_-target)**2)
    return tf.reduce_mean(tf.abs(in_ - target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def entropy_criterion2(logits1,logits2):
    l1 = tf.nn.softmax(logits1)
    l2 = tf.nn.softmax(logits2)
    return tf.reduce_mean((l1 - l2)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def softmax_criterion(logits,labels,N):
  #onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N, 1, 0)
  onehot_labels = tf.one_hot(labels, N, 1, 0)

  return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
      labels= onehot_labels, logits=logits)) #TODO: for me too!

def softmax_criterion2(logits,labels,N):
  #onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N, 1, 0)
  onehot_labels = 1 - tf.one_hot(labels, N, 1, 0)

  return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
      labels= onehot_labels, logits=logits)) #TODO: for me too!

def softmaxce_criterion_maximization(logits,labels,N):
  onehot_labels = tf.one_hot(labels, N, 1., 0.)
  onehot_labels = 1. - onehot_labels
  return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
      labels=onehot_labels, logits=logits))

def softmaxce_criterion_maximization2(logits,labels,N):
  l = tf.nn.softmax(logits[:,0:N])
  return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
      labels=l, logits=logits[:,0:N]))

def entropy_criterion(logits):
    ll = tf.nn.softmax(logits)
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=ll, logits=logits))
    #return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=ll, logits=logits))

def discriminator_entropy_criterion(logits):
    ll = tf.nn.softmax (logits[:,0:-1])
    return tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits_v2 (labels=ll, logits=logits[:,0:-1]))
def new_entropy_criterion(logits,num_replication):
    ll = tf.nn.softmax(logits)
    ave_normalized_logits = tf.reduce_mean(ll,axis=0)
    #multiply = tf.constant([num_replication])
    #ave_logits = tf.reshape (tf.tile (ave_logits, multiply), [multiply[0], tf.shape (ave_logits)[0]])
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=ll, logits=logits)) \
             - tf.reduce_mean (tf.multiply (ave_normalized_logits,tf.log(ave_normalized_logits)))
                #- tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(labels=ll, logits=ave_logits))
    #return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=ll, logits=logits))

def generator_criterion(discriminator_logits,classifier_logits):
    nor_discriminator_logits = tf.nn.softmax(discriminator_logits)
    nor_classifier_logits = tf.nn.softmax(classifier_logits)
    padding = tf.constant([[0,0],[0,1]])
    return tf.reduce_mean(tf.reduce_sum(tf.abs( nor_discriminator_logits - tf.pad(nor_classifier_logits, padding, "CONSTANT")),axis=1))
    #return tf.reduce_mean (
     #   tf.reduce_sum ( (nor_discriminator_logits - tf.pad (nor_classifier_logits, padding, "CONSTANT"))**2, axis=1))


def generator_criterion_cross_entropy(discriminator_logits, classifier_logits):
    nor_classifier_logits = tf.nn.softmax (classifier_logits + 1.)
    padding = tf.constant ([[0, 0], [0, 1]])
    return tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits (
        labels=tf.pad (nor_classifier_logits, padding, "CONSTANT"), logits=discriminator_logits))


def normalize_perturbation(d, scope=None):
        
        shape = d.get_shape ().as_list ()
        #print ("jjjjjjjjjjjjjjjjjjj", len(shape))
        #output = tf.nn.l2_normalize(d,axis=range(1, len(shape)))
        output = tf.nn.l2_normalize (d, axis=tf.convert_to_tensor([1,2,3]))
        return output
def scale_gradient(x, scale, scope=None, reuse=None):
    
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
        return output
def noise(x, std, phase, scope=None, reuse=None):
    
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
        return output
def perturb_image(x, p, shared_encoder, classifier, options, num_class, pert='vat', scope=None):
    
        radius = 3.5
        eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

        # Predict on randomly perturbed image
        eps_p = classifier(shared_encoder(x + eps,options, reuse = True, name="shared_encoder"),
                           options = options, num_class = num_class, reuse=True,name="classifier")
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=p, logits=eps_p)

        # Based on perturbed image, get direction of greatest error
        eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

        # Use that direction as adversarial perturbation
        eps_adv = normalize_perturbation(eps_adv)
        x_adv = tf.stop_gradient(x + radius * eps_adv)

        return x_adv
def vat_loss(x, p, shared_encoder, classifier, options, num_class):
    
        x_adv = perturb_image(x, p, shared_encoder, classifier, options, num_class)
        p_adv = classifier(shared_encoder(x_adv,options, reuse = True, name="shared_encoder"),options, num_class = num_class, reuse=True,name="classifier")
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(p), logits=p_adv))

        return loss



def hellinger_loss(logits,power):
    ll = tf.nn.softmax(logits)
    #ll = ll**(1.0/power)
    ll = tf.pow(ll,1.0/power)
    return tf.reduce_mean(tf.reduce_prod(ll,axis=1))

def dense_discriminator(signal, options, reuse=False, name="dense_discriminator"):
    with tf.variable_scope(name):
        # signal is is 1 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        h0 = tf.layers.dense(signal, options.df_dim*4, activation=tf.nn.relu)
        h1 = tf.layers.dense(h0, options.df_dim*2, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, options.df_dim*1, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 1, activation = None)
        return h3

