"""
Tensorflow related part
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import wandb 
import numpy as np

from src import param
from utils import sigmoid


class TFParts(object):
    '''
    TensorFlow-related things.
    Keep TensorFlow-related components in a neat shell.
    '''

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, dim_r=None):
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = self._dim_r = dim  # dimension of both relation and ontology.
        if dim_r:
            self._dim_r = dim_r
        self._batch_size = batch_size
        self.batch_size_eval = batch_size*16
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = 1
        self._p_psl = 0.2
        self._soft_size = 1
        self._prior_psl = 0
        self._psl = psl

        assert psl == False

    def build_basics(self):
#         print("reset")
        tf.reset_default_graph()
        with tf.variable_scope("graph", initializer=tf.truncated_normal_initializer(0, 0.3)):
            # Variables (matrix of embeddings/transformations)
            self._ht = ht = tf.get_variable(
                name='ht',  # for t AND h
                shape=[self.num_cons, self._dim],
                dtype=tf.float32)

            self._r = r = tf.get_variable(
                name='r',
                shape=[self.num_rels, self._dim_r],
                dtype=tf.float32)
#             self._output_ht = ht = tf.placeholder(
#                 name='output_ht',  # for t AND h
#                 shape=[self.num_cons, self._dim],
#                 dtype=tf.float32)

#             self._output_r = r = tf.placeholder(
#                 name='output_r',
#                 shape=[self.num_rels, self._dim_r],
#                 dtype=tf.float32)
            
            self._A_h_index = A_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_h_index')
            self._A_r_index = A_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_r_index')
            self._A_t_index = A_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_t_index')
            
            # for uncertain graph w in (h,r,t,w)
            self._A_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_A_w')
            # negative sample index : [batch_size, numberofsample] ( batch_size* ( h'1,h'2, ...))
            self._A_neg_hn_index = A_neg_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_hn_index')
            self._A_neg_rel_hn_index = A_neg_rel_hn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_rel_hn_index')
            self._A_neg_t_index = A_neg_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_t_index')
            self._A_neg_h_index = A_neg_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_h_index')
            self._A_neg_rel_tn_index = A_neg_rel_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_rel_tn_index')
            self._A_neg_tn_index = A_neg_tn_index = tf.placeholder(
                dtype=tf.int64,
                shape=(None, self._neg_per_positive),
                name='A_neg_tn_index')

            self._B_neg_hn_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._neg_per_positive],
                name='_B_neg_hn_w')

            self._B_neg_tn_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self._neg_per_positive],
                name='_B_neg_tn_w')

            self._B_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_B_w')

            self._A_semi_h_index = A_semi_h_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_h_index')
            self._A_semi_r_index = A_semi_r_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_r_index')
            self._A_semi_t_index = A_semi_t_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None],
                name='A_semi_t_index')

            self._A_semi_w = tf.placeholder(
                dtype=tf.float32,
                shape=[None],
                name='_A_semi_w')
            

            # no normalization
            self._h_batch = tf.nn.embedding_lookup(ht, A_h_index)
            self._t_batch = tf.nn.embedding_lookup(ht, A_t_index)
            self._r_batch = tf.nn.embedding_lookup(r, A_r_index)
            # index to embedding
            self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, A_neg_hn_index)
            self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, A_neg_rel_hn_index)
            self._neg_t_con_batch = tf.nn.embedding_lookup(ht, A_neg_t_index)
            self._neg_h_con_batch = tf.nn.embedding_lookup(ht, A_neg_h_index)
            self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, A_neg_rel_tn_index)
            self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, A_neg_tn_index)

            self._semi_h_batch = tf.nn.embedding_lookup(ht, A_semi_h_index)
            self._semi_t_batch = tf.nn.embedding_lookup(ht, A_semi_t_index)
            self._semi_r_batch = tf.nn.embedding_lookup(r, A_semi_r_index)
            
            
            if self._psl:
                # psl batches
                self._soft_h_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_h_index')
                self._soft_r_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_r_index')
                self._soft_t_index = tf.placeholder(
                    dtype=tf.int64,
                    shape=[self._soft_size],
                    name='soft_t_index')

                # for uncertain graph and psl
                self._soft_w = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self._soft_size],
                    name='soft_w_lower_bound')

                self._soft_h_batch = tf.nn.embedding_lookup(ht, self._soft_h_index)
                self._soft_t_batch = tf.nn.embedding_lookup(ht, self._soft_t_index)
                self._soft_r_batch = tf.nn.embedding_lookup(r, self._soft_r_index)


    def build_optimizer(self):
        if self._psl: 
            self._A_loss = tf.add(self.main_loss, self.psl_loss)
        else:
            self._A_loss = self.main_loss

        # Optimizer
        self._lr = lr = tf.placeholder(tf.float32)
        self._opt = opt = tf.train.AdamOptimizer(lr)

        # This can be replaced by
        # self._train_op_A = train_op_A = opt.minimize(A_loss)
        self._gradient = gradient = opt.compute_gradients(self._A_loss)  # splitted for debugging

        self._train_op = opt.apply_gradients(gradient)

        # Saver
        self._saver = tf.train.Saver(max_to_keep=1000)

    def build(self):
        self.build_basics()
        self.define_main_loss()  # abstract method. get self.main_loss
        if self._psl:
            self.define_psl_loss()  # abstract method. get self.psl_loss
        self.build_optimizer()

    def compute_psl_loss(self):
        self.prior_psl0 = tf.constant(self._prior_psl, tf.float32)
        self.psl_error_each = tf.square(tf.maximum(self._soft_w + self.prior_psl0 - self.psl_prob, 0))
        self.psl_mse = tf.reduce_mean(self.psl_error_each)
        self.psl_loss = self.psl_mse * self._p_psl

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size


class UKGE_logi_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl = False):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


class UKGE_rect_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.reg_scale = reg_scale
        self.build()

    # override
    def define_main_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)

        self._f_prob_h = f_prob_h = self.w * htr + self.b
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.this_loss = this_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

        # L2 regularizer
        self._regularizer = regularizer = tf.add(tf.add(tf.divide(tf.nn.l2_loss(self._h_batch), self.batch_size),
                                                        tf.divide(tf.nn.l2_loss(self._t_batch), self.batch_size)),
                                                 tf.divide(tf.nn.l2_loss(self._r_batch), self.batch_size))

        self.main_loss = tf.add(this_loss, self.reg_scale * regularizer)

    # override
    def define_psl_loss(self):
        self.psl_prob = self.w * tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1) + self.b
        self.compute_psl_loss()  # in tf_parts









class TransE_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r, numpy= False):
        if numpy:
            return np.sum((h + r - t)**2, axis=-1)
        else: #tf
            return tf.reduce_sum((h + r - t)**2, -1)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = self.score_function(self._h_batch, self._t_batch, self._r_batch)

        self._f_score_hn = f_score_hn = self.score_function(self._neg_hn_con_batch, self._neg_t_con_batch, self._neg_rel_hn_batch)

        self._f_score_tn = f_score_tn = self.score_function(self._neg_h_con_batch, self._neg_tn_con_batch, self._neg_rel_tn_batch)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class TransE_m2_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r):
        return tf.reduce_sum((h + r - t)**2, -1)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_prob_h = f_prob_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression
        self._f_score_h = f_score_h = tf.square(tf.subtract(f_prob_h, self._A_w))

        self._f_prob_hn = f_prob_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(f_prob_hn), 1)

        self._f_prob_tn = f_prob_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)
        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(f_prob_tn), 1)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_prob_hn, f_prob_tn), 2),
                        tf.tile(tf.expand_dims(f_prob_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts




class TransE_m3_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def score_function(self, h, t, r):
        return tf.sigmoid(self.w*(2 - tf.reduce_sum((h + r - t)**2, -1)) + self.b)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch,
            self._neg_t_con_batch, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch,
            self._neg_tn_con_batch, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))

    def define_psl_loss(self):
        self.compute_psl_loss()  # in tf_parts


# no sigmoid
class TransE_m3_1_TF(TransE_m3_TF):

    def score_function(self, h, t, r):
        return self.w*(4 - tf.reduce_sum((h + r - t)**2, -1)) + self.b



# Uncertain TransE
# + semisupervised_v2
class TransE_m3_3_TF(TransE_m3_TF):


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)


        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)

        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch,
            self._neg_t_con_batch, 
            self._neg_rel_hn_batch)
            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

    def define_psl_loss(self):
        self.compute_psl_loss()  # in tf_parts









class DistMult_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_score_h = f_score_h = htr # tf.sigmoid(self.w * htr + self.b)  # logistic regression

        # self._f_score_hn = f_score_hn = tf.sigmoid(self.w * (
        #     tf.reduce_sum(
        #         tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        # ) + self.b)
        self._f_score_hn = f_score_hn = tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)

        self._f_score_tn = f_score_tn = tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class DistMult_m2_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._htr = htr = tf.reduce_sum(
            tf.multiply(self._r_batch, tf.multiply(self._h_batch, self._t_batch, "element_wise_multiply"),
                        "r_product"),
            1)
        self._f_score_h = f_score_h = tf.sigmoid(self.w * htr + self.b)  # logistic regression

        self._f_score_hn = f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)

        self._f_score_tn = f_score_tn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_tn_batch, tf.multiply(self._neg_h_con_batch, self._neg_tn_con_batch)), 2)
        ) + self.b)

        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts




class ComplEx_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()

    def split_embed_real_imag(self, vec, numpy = False):
        if numpy:
            return np.split(vec, [int(self._dim/2)], axis=-1)
        else: #tensorflow
            split_size = [int(self._dim/2), int(self._dim/2)]
            return tf.split(vec,  split_size, -1)

    # override
    def build_basics(self):
        TFParts.build_basics(self)
        # split the embedding into real and imaginary part
        assert self._dim%2 == 0
        
        self._h_batch_real, self._h_batch_imag = self.split_embed_real_imag(self._h_batch)
        self._t_batch_real, self._t_batch_imag = self.split_embed_real_imag(self._t_batch)
        self._r_batch_real, self._r_batch_imag = self.split_embed_real_imag(self._r_batch)

        self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag = self.split_embed_real_imag(self._neg_hn_con_batch)
        self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag = self.split_embed_real_imag(self._neg_rel_hn_batch)
        self._neg_t_con_batch_real, self._neg_t_con_batch_imag   = self.split_embed_real_imag(self._neg_t_con_batch)
        self._neg_h_con_batch_real, self._neg_h_con_batch_imag   = self.split_embed_real_imag(self._neg_h_con_batch)
        self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag = self.split_embed_real_imag(self._neg_rel_tn_batch)
        self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag = self.split_embed_real_imag(self._neg_tn_con_batch)            


    def score_function(self, h_real, h_imag, t_real, t_image, r_real, r_imag, numpy = False):
        if numpy:
            return np.sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, axis=-1)
        else: #tf
            return tf.reduce_sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            
        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


class ComplEx_m1_1_TF(ComplEx_m1_TF):
    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False):
        if numpy:
            return np.sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, axis=-1)
        else: #tf
            return tf.reduce_sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False)





class ComplEx_m3_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    # override
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = tf.square(self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag) - self._A_w)

        self._f_score_hn = f_score_hn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)


        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size


class ComplEx_m4_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)


    def score_function(self, h_real, h_imag, t_real, t_image, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            return sigmoid(numpy_w*np.sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, axis=-1) + numpy_b)
        else: #tf
            return tf.sigmoid(
                self.w * 
                tf.reduce_sum(h_real * t_real * r_real + h_imag * t_real * r_real + h_real * t_real * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False) +
                self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            
        self.main_loss = (tf.reduce_sum(
                tf.maximum(
                    tf.reshape(tf.add(tf.subtract( 
                        tf.divide(tf.add(f_score_hn, f_score_tn), 2),
                        tf.tile(tf.expand_dims(f_score_h, -1),  [1, self._neg_per_positive])
                    ), tf.tile(tf.expand_dims(self._A_w, -1),  [1, self._neg_per_positive])), [self._batch_size*self._neg_per_positive])
                , 0), 
            )) / self._batch_size

    def define_psl_loss(self):        
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class ComplEx_m5_1_TF(ComplEx_m1_TF):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
        ComplEx_m1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            return sigmoid(numpy_w*np.sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, axis=-1) + numpy_b)
        else: #tf
            return tf.sigmoid(
                self.w * 
                tf.reduce_sum(h_real * t_real * r_real + h_imag * t_imag * r_real + h_real * t_imag * r_imag - h_imag * t_real * r_imag, -1, keep_dims = False) +
                self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):        
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


    

# increase the weight of neg sample in loss
class ComplEx_m5_2_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))





# uncertain complex with semisupervised_v2
class ComplEx_m5_3_TF(ComplEx_m5_1_TF):

    # override
    def build_basics(self):
        ComplEx_m5_1_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)
        self._semi_r_batch_real, self._semi_r_batch_imag = self.split_embed_real_imag(self._semi_r_batch)


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))



# semisupervised_v2
# 20200414 / batch_size in loss function
class ComplEx_m5_4_TF(ComplEx_m5_3_TF):


    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        # self.main_loss = tf.add(tf.reduce_mean(f_score_semi)*self.batch_size, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_mean(f_score_semi)*self.batch_size, tf.reduce_sum(f_score_h))/self.batch_size






# with --semisupervised_neg_v2
class ComplEx_m6_1_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn =self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn - self._B_neg_hn_w), 1)


        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag) - self._B_neg_hn_w), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

# increase the weight of neg sample in loss
class ComplEx_m6_2_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn =self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn - self._B_neg_hn_w), 1)


        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag) - self._B_neg_hn_w), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))






# = m5 - MSE + negative logarithm loss
class ComplEx_m7_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            

        self.main_loss = tf.reduce_sum(
                (
                    -tf.math.log(tf.sigmoid(f_score_h - tf.reduce_mean(tf.divide((f_score_hn + f_score_tn), 2), -1)))
                )*self._A_w 
                # (
                #     -tf.math.log(tf.sigmoid(f_score_h )) -
                #     tf.reduce_mean(tf.math.log(tf.sigmoid(-f_score_hn)), -1)   -
                #     tf.reduce_mean(tf.math.log(tf.sigmoid(-f_score_tn)), -1)
                # )*self._A_w 
            , -1) / self._batch_size



# = m5  + negative logarithm loss 
class ComplEx_m8_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

        self._f_score_tn = f_score_tn = self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)
            

        self.main_loss = tf.reduce_sum(
                (
                    -tf.math.log(tf.sigmoid(f_score_h - tf.reduce_mean(tf.divide((f_score_hn + f_score_tn), 2), -1)))
                )*self._A_w 
            , -1) / self._batch_size + tf.reduce_mean(
                tf.add(tf.divide(tf.add(
                    tf.reduce_mean(tf.square(f_score_tn)), 
                    tf.reduce_mean(tf.square(f_score_hn)))
                , 2) * self._p_neg, tf.square(f_score_h - self._A_w)), -1)


# m5_1 + auto encoder (2 layers)
class ComplEx_m9_1_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        self._autoencoder_h_dense1 = tf.layers.dense(h_batch, 1024)
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(self._autoencoder_h_dense1, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size

# m5_1 + auto encoder(1 layer)
class ComplEx_m9_2_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        # self._autoencoder_h_dense1 = tf.layers.dense(h_batch, 1024)
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(h_batch, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size


# m5_1 + auto encoder (2 layers + relu)
class ComplEx_m9_3_TF(ComplEx_m5_1_TF):
    
    def define_main_loss(self):
        ComplEx_m5_1_TF.define_main_loss(self)

        y_autoencoder_h = tf.one_hot(self._A_h_index, self._num_cons)

        h_batch = tf.concat([self._h_batch_real, self._h_batch_imag], -1)
        # t_batch = tf.concat([self._t_batch_real, self._t_batch_imag], -1)
        self._autoencoder_h_dense1 = tf.nn.relu(tf.layers.dense(h_batch, 1024))
        self._autoencoder_h_output = self._autoencoder_h_dense2 = tf.layers.dense(self._autoencoder_h_dense1, self._num_cons)
        # self._autoencoder_h_output = tf.nn.softmax(self._autoencoder_h_dense2)

        lambda_autoencoder = 1.0

        self.main_loss += lambda_autoencoder*tf.nn.softmax_cross_entropy_with_logits(logits = self._autoencoder_h_output, labels = y_autoencoder_h) / self.batch_size




# semi supervised - confidence weighted
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w)* self._B_w)


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn*(1 - self._B_neg_hn_w) ), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)*(1 - self._B_neg_hn_w) ), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size



# semi supervised - confidence weighted
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_1_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn*(1 - self._B_neg_hn_w) ), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)*(1 - self._B_neg_tn_w) ), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_2_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_sum(tf.square(f_score_hn)*(self._B_neg_hn_w + 0.00001), 1) / (0.00001*self._batch_size + tf.reduce_sum(self._B_neg_hn_w, -1))

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag))*(self._B_neg_tn_w + 0.00001), 1) / (0.00001*self._batch_size + tf.reduce_sum(self._B_neg_tn_w, -1))
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size


# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1 or --semisupervised_v1_1
class ComplEx_m10_3_TF(ComplEx_m10_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_sum(tf.square(f_score_hn)*(tf.math.exp(self._B_neg_hn_w + 0.00001)), 1) / tf.reduce_sum(tf.math.exp(0.00001 + self._B_neg_hn_w), -1) * self._neg_per_positive

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag,  
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag))*(tf.math.exp(self._B_neg_tn_w + 0.00001)), 1) / tf.reduce_sum(tf.math.exp(0.00001 + self._B_neg_tn_w), -1) * self._neg_per_positive
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size



# semi supervised - confidence weighted 2 - select those with higher weight
# run with --semisupervised_v1_1
class ComplEx_m10_4_TF(ComplEx_m5_1_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)
            
        f_score_h = tf.square((f_score_h - self._A_w))*self._B_w


        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)
            
        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), -1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag,  
            self._neg_rel_tn_batch_real, self._neg_rel_tn_batch_imag)), -1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size








class RotatE_m1_TF(ComplEx_m1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, dim_r=int(dim/2))
        self._gamma = gamma # 24
        self._epsilon = 0.2
        self.build()

    def split_embed_real_imag(self, vec, numpy = False):
        if numpy:
            return np.split(vec, [int(self._dim/2)], axis=-1)
        else: #tf
            split_size = [int(self._dim/2), int(self._dim/2)]
            return tf.split(vec,  split_size, -1)


    # override
    def build_basics(self):
        TFParts.build_basics(self)
        # split the embedding into real and imaginary part
        assert self._dim%2 == 0
        
        self._h_batch_real, self._h_batch_imag = self.split_embed_real_imag(self._h_batch)
        self._t_batch_real, self._t_batch_imag = self.split_embed_real_imag(self._t_batch)

        self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag = self.split_embed_real_imag(self._neg_hn_con_batch)
        self._neg_t_con_batch_real, self._neg_t_con_batch_imag   = self.split_embed_real_imag(self._neg_t_con_batch)
        self._neg_h_con_batch_real, self._neg_h_con_batch_imag   = self.split_embed_real_imag(self._neg_h_con_batch)
        self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag = self.split_embed_real_imag(self._neg_tn_con_batch)            


    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            r_phrase = r/magic
            r_real = np.cos(r_phrase)
            r_imag = np.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_real * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r_phrase)
            r_imag = tf.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_real * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size

    def define_psl_loss(self):

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts



class RotatE_m2_TF(RotatE_m1_TF):

    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        # magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            # r_phrase = r/magic
            r_real = np.cos(r)
            r_imag = np.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)
        else: #tf
            
            # r_phrase = r/magic
            r_real = tf.cos(r)
            r_imag = tf.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)


class RotatE_m2_1_TF(RotatE_m2_TF):
    def define_main_loss(self):
        print('define main loss')
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_sum(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_sum(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = tf.reduce_mean(
            tf.add(tf.add(f_score_tn, f_score_hn), f_score_h))
    

class RotatE_m2_2_TF(RotatE_m2_1_TF):
    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            raise NotImplementedError
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r)
            r_imag = tf.sin(r)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)
    





# complex (ht -r)
class RotatE_m3_TF(ComplEx_m5_1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma=1):
        self._gamma = gamma # 24
        ComplEx_m5_1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag


            return sigmoid(numpy_w*(self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)) + numpy_b)
        else: #tf
            
            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)


# Simplified Uncertain RotatE
# complex (ht -r)
# without semi-supervised
class RotatE_m3_1_TF(RotatE_m3_TF):

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:
            raise NotImplementedError
        else: #tf
            
            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)

# Simplified Uncertain RotatE
# complex (ht -r)
# + semisupervised_v2_2
class RotatE_m3_2_TF(RotatE_m3_TF):
# override
    def build_basics(self):
        RotatE_m3_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)
        self._semi_r_batch_real, self._semi_r_batch_imag = self.split_embed_real_imag(self._semi_r_batch)



    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

# Simplified Uncertain RotatE
# complex (ht -r) 
# + semisupervised_v2 
# positive:generated=1:1 weight

class RotatE_m3_3_TF(RotatE_m3_2_TF):

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch_real, self._r_batch_imag)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch_real, self._semi_r_batch_imag)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch_real, self._neg_rel_hn_batch_imag)

            
        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive / 2, tf.reduce_sum(f_score_h)) / self.batch_size
        # 20200414 / batch_size in loss function





# complex (hr -t)
class RotatE_m4_TF(ComplEx_m5_1_TF):

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, gamma):
        self._gamma = gamma # 24
        ComplEx_m5_1_TF.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)

    def score_function(self, h_real, h_imag, t_real, t_imag, r_real, r_imag, numpy = False, numpy_w = None, numpy_b = None):
        if numpy:

            score_real = h_real * t_real - h_imag * t_imag
            score_imag = h_real * t_imag + h_imag * t_real
            score_real = score_real - r_real
            score_imag = score_imag - r_imag


            return sigmoid(numpy_w*(self._gamma - np.sum(np.square(score_real) + np.square(score_imag), axis=-1)) + numpy_b)
        else: #tf
            
            score_real = h_real * t_real - h_imag * t_imag
            score_imag = h_real * t_imag + h_imag * t_real
            score_real = score_real - r_real
            score_imag = score_imag - r_imag

            return tf.sigmoid(self.w*(self._gamma - tf.reduce_sum(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)


# Uncertain RotatE
# "real" rotatE + MSE loss
# semisupervised_v2 positive:generated=1:1 weight
# self.r_batch_real and self.r_batch_imag are not used
# 20200414 / batch_size in loss function
class RotatE_m5_TF(RotatE_m1_TF):
    # override
    def build_basics(self):
        RotatE_m1_TF.build_basics(self)

        self._semi_h_batch_real, self._semi_h_batch_imag = self.split_embed_real_imag(self._semi_h_batch)
        self._semi_t_batch_real, self._semi_t_batch_imag = self.split_embed_real_imag(self._semi_t_batch)

    def score_function(self, h_real, h_imag, t_real, t_imag, r, numpy = False):
        pi = 3.14159265358979323846
        magic = (self._gamma + self._epsilon)/self._dim/pi
        
        if numpy:
            raise NotImplementedError
        else: #tf
            
            r_phrase = r/magic
            r_real = tf.cos(r_phrase)
            r_imag = tf.sin(r_phrase)

            score_real = h_real * r_real - h_imag * r_imag
            score_imag = h_real * r_imag + h_imag * r_real
            score_real = score_real - t_real
            score_imag = score_imag - t_imag

            return tf.sigmoid(self.w*(2 - tf.reduce_mean(tf.square(score_real) + tf.square(score_imag), -1)) + self.b)
    
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch_real, self._semi_h_batch_imag,
            self._semi_t_batch_real, self._semi_t_batch_imag, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

            
        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi) / self._neg_per_positive / 2, tf.reduce_sum(f_score_h)) / self.batch_size

# Uncertain RotatE
# rotatE m5 without semi-supervised
class RotatE_m5_1_TF(RotatE_m5_TF):
    
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        

        self._f_score_h = f_score_h = self.score_function(
            self._h_batch_real, self._h_batch_imag,
            self._t_batch_real, self._t_batch_imag, 
            self._r_batch)

        f_score_h = tf.square(f_score_h - self._A_w)

        self._f_score_hn = f_score_hn = self.score_function(
            self._neg_hn_con_batch_real, self._neg_hn_con_batch_imag,
            self._neg_t_con_batch_real, self._neg_t_con_batch_imag, 
            self._neg_rel_hn_batch)

        f_score_hn = tf.reduce_mean(tf.square(f_score_hn), 1)

        self._f_score_tn = f_score_tn = tf.reduce_mean(tf.square(self.score_function(
            self._neg_h_con_batch_real, self._neg_h_con_batch_imag,
            self._neg_tn_con_batch_real, self._neg_tn_con_batch_imag, 
            self._neg_rel_tn_batch)), 1)
            
        self.main_loss = (tf.reduce_sum(
            tf.add(tf.divide(tf.add(f_score_tn, f_score_hn), 2) * self._p_neg, f_score_h))) / self._batch_size







# + semisupervised_v2_2
class UKGE_logi_m1_TF(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl = False):
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()


    def score_function(self, h, t, r, numpy=False):
        htr = tf.reduce_sum(
            tf.multiply(r, tf.multiply(h, t, "element_wise_multiply"),
                        "r_product"),
            1)

        return tf.sigmoid(self.w * htr + self.b)

    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        
        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)
        f_score_h = tf.square(tf.subtract(f_score_h, self._A_w))

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = _f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)


        self.main_loss = tf.add(tf.reduce_sum(f_score_semi), tf.reduce_sum(f_score_h))

    def define_psl_loss(self):
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts


# UKGE logi with semi-supervised (uncertain DistMult)
# + semisupervised_v2
class UKGE_logi_m2_TF(UKGE_logi_m1_TF):

    
    def define_main_loss(self):
        # distmult on uncertain graph
        print('define main loss')

        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")

        
        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)
        f_score_h = tf.square(tf.subtract(f_score_h, self._A_w))

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        f_score_semi = tf.square(f_score_semi - self._A_semi_w)


        # evaluation only
        self._f_score_hn = _f_score_hn = tf.sigmoid(self.w * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b)
#         print(_f_score_hn)

        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/ 2 /self._neg_per_positive, tf.reduce_sum(f_score_h)) / self.batch_size
        # 20200414 / batch_size in loss function

class DistMult_VAE(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl, ver):
        # now [256, 128, 64, 64] 
        self._n_hidden = [256,256]
        self.score = [self._n_hidden[-1], 256, 256]
        self._alpha = 0.1
        self.beta = 0.01
        self.gamma = 0
        self._ver = ver
        self.mask_ratio = 0.05
        self.VAE_R = None # VAE(dim=dim*3, n_hidden=self._n_hidden)
        self.VAE_E = None # VAE(dim=dim*3, n_hidden=self._n_hidden)
        self.seperator = 1
        self.head = int(self.seperator)
        self.delta = 1 / ((self.seperator+1)*self.seperator/2)
        self.margin = 0.3
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()
    def kl_divergence(self, p, q): 
        return tf.reduce_sum(p * tf.log(tf.math.maximum(p,1e-6)/tf.math.maximum(q,1e-6)))
    def MLP_score_function(self, h, t=None, r=None, numpy=False,attention=None):
#         print(attention)
        kl=0
        
        if t != None:
            a = []
            htr = []
            for i in range(self.seperator):
                htr.append(h[i]+r[i])
                for j in range(len(self.score)-1):
                    htr[i] = tf.add(tf.matmul(htr[i], self.weights_score[j]), self.biases_score[j])
#                 print(t[i])
#                 print(htr[i])
                x = tf.reduce_sum(tf.multiply(htr[i],t[i]),axis=1)
#                 print(htr[i])
                a.append(tf.math.sigmoid(0.01*(self.w_main * (x + self.gamma) + self.b_main)))
#             print(a)
            ans = tf.stack(a,axis=1)
            print(ans)
            for i in range(self.seperator-1):
                for j in range(i+1,self.seperator):
                    kl += self.kl_divergence(a[i],a[j])

        if attention == None:
            result = tf.reduce_sum(ans, axis=1)
        else:
            result = tf.reduce_sum(tf.multiply(ans, attention,  "attention"), axis=1)

        return result, kl
    def score_function(self, h, t=None, r=None, numpy=False,attention=None):
#         print(attention)
        kl = 0
        if t != None:
            top_k = 2
            a = []
            htr = []
            for i in range(self.seperator):
                htr.append(tf.reduce_sum(
                    tf.multiply(r[i], tf.multiply(h[i], t[i], "element_wise_multiply"),
                                "r_product"),
                    1))
                a.append(tf.math.sigmoid(0.1 * (self.w_main * (htr[i] + self.gamma) + self.b_main)))
            ans = tf.stack(a,axis=1)
#             print(ans)
            for i in range(self.seperator-1):
                for j in range(i+1,self.seperator):
                    kl += self.kl_divergence(a[i],a[j])
#             print(ans)
#             result = tf.math.top_k(ans,k=top_k)
    #         print(result)
        else:
            top_k = 2
            a = []
            htr = []
            for i in range(self.seperator):
                a.append(self.w_main * (tf.reduce_sum(h[i], 1) + self.gamma) + self.b_main)
            ans = tf.stack(a,axis=1)
#             print(ans)
#             ans, _ = tf.math.top_k(ans,k=top_k)
#         print(result)
        if attention == None:
            result = tf.reduce_sum(ans, axis=1)
        else:
#             ans = tf.math.maximum(ans*0.005, ans*0.05)
#             ans = 1/ (1+tf.math.exp(-0.1*ans))
            result = tf.reduce_sum(tf.multiply(ans, attention,  "attention"), axis=1)
#         result = tf.math.minimum(result, 1)
#         result = tf.sigmoid(result)
#         print(result)
        return result, kl
    def attention(self, h, r, t):
#       attention mech
        
        if self._ver == 2:
            features = tf.concat([h,r,t],axis=1)
            x = tf.add(tf.matmul(features, self.weights_attention), self.biases_attention)
#             att = tf.sigmoid(x)
            att = tf.nn.softmax(x+1e-5)
#         att = tf.nn.softmax(att, axis=1)
#         att = tf.math.minimum(att, 1)
#         att = tf.math.maximum(att, 0)
        
        if self._ver == 3:
            features = []
            x = []
            
            for i in range(self.seperator):
                h[i] = tf.add(tf.matmul(h[i], self.weights_trans[i]),self.biases_trans[i])
                r[i] = tf.add(tf.matmul(r[i], self.weights_trans[i]),self.biases_trans[i])
                t[i] = tf.add(tf.matmul(t[i], self.weights_trans[i]),self.biases_trans[i])
                features.append(tf.concat([h[i],r[i],t[i]],axis=1))
                w = []
                for j in range(self.head):
                    w.append(tf.add(tf.matmul(features[i], self.weights_attention[j]), self.biases_attention[j]))
                x.append(tf.reduce_mean(tf.concat(w, axis=1), axis=1))
    #         print(x[0])
            att = tf.stack(x,axis=1)
    # #         print(ans)
#             att = tf.nn.leaky_relu(ans)
#             att,_ = tf.linalg.normalize(att+1e-5, ord=1, axis=1)
#             att = tf.sigmoid(att+0.1)
            att = tf.nn.softmax(att+1e-5)

        return att
    
    def define_main_loss(self):
        self.w_main = tf.Variable(0.0, name="weights_main")
        self.b_main = tf.Variable(0.0, name="bias_main")
        
        print('define main loss')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())
        

#         Ver 2: Distmult with MLP embedding
        if self._ver == 2:
            self.weights_attention = tf.Variable(tf.truncated_normal(shape=(self._dim*3, self.seperator), stddev=0.5), dtype=tf.float32, name="weight_attention")
            self.biases_attention = tf.Variable(tf.zeros(shape=(self.seperator)), dtype=tf.float32, name="bias_attention")
            
#             self.VAE_R = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
            self.VAE_E = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
#           Pos Sample [batch_size,dim]
            # pos_combined_features = tf.add(self._h_batch,self._r_batch)
            _, self.h_original_loss, self.h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._h_batch,
                                               targets=self._h_batch)
#             print(self._r_batch)
            _, self.r_original_loss, self.r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._r_batch,
                                               targets=self._r_batch)
            _, self.t_original_loss, self.t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._t_batch,
                                               targets=self._t_batch)
            
#             self._htr = htr = tf.reduce_sum(
#             tf.multiply(self.h_encoded_features, tf.multiply(self.r_encoded_features, self.t_encoded_features, 
#                                                              "element_wise_multiply_pos"),"r_product_pos"),1)
#             self._f_score_h = f_score_h = tf.sigmoid(self.get_dense_layer(htr, self.Denseweight, self.Densebiases, activation=None))
#             attention= self.attention(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features,self.weights_attention, self.biases_attention)
            attention= self.attention(self._h_batch,self._r_batch,self._t_batch)
#             print(attention)
            self._f_score_h, self._pos_kl = f_score_h, pos_kl  = self.score_function(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features,attention=attention)
            self._pos_loss = pos_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_h, self._A_w)))
        
        
#             print(self._pos_loss)
#             neg sample with VAE (h',r,t) [batch_size, neg_per_positive, dim]
#           only for evaluation
            self._neg_hn_con_batch = tf.reshape(self._neg_hn_con_batch,[-1,self._dim])
            self._neg_rel_hn_batch = tf.reshape(self._neg_rel_hn_batch,[-1,self._dim])
            self._neg_t_con_batch = tf.reshape(self._neg_t_con_batch,[-1,self._dim])
            _, self.neg_hn_loss,_, self.neg_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_hn_con_batch,
                                               targets=self._neg_hn_con_batch)
            _, self.neg_r_hn_loss,_, self.neg_r_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_rel_hn_batch,
                                               targets=self._neg_rel_hn_batch)
            _, self.neg_t_loss,_, self.neg_t_encoded_features = self.VAE_E.structure(
                                               features=self._neg_t_con_batch,
                                               targets=self._neg_t_con_batch)
            
#             self.hn_htr = hn_htr = tf.reduce_sum(
#             tf.multiply(self.neg_hn_encoded_features, tf.multiply(self.neg_r_hn_encoded_features, self.neg_t_encoded_features, 
#                                                              "element_wise_multiply_hneg"),"r_product_hneg"),1)
#             self._f_score_hn = f_score_hn = tf.sigmoid(self.w_main * (hn_htr + self.gamma) + self.b_main)
#             attention_neg = self.attention(self.neg_hn_encoded_features,self.neg_r_hn_encoded_features,self.neg_t_encoded_features,self.weights_attention, self.biases_attention )
            attention_neg = self.attention(self._neg_hn_con_batch,self._neg_rel_hn_batch,self._neg_t_con_batch)
#             print(attention_neg)
            pre_cvt_score_hn, neg_kl = self.score_function(self.neg_hn_encoded_features, self.neg_r_hn_encoded_features, self.neg_t_encoded_features, attention=attention_neg)
            self._f_score_hn = f_score_hn  = tf.reshape(pre_cvt_score_hn, [-1,self._neg_per_positive])
#             self.cvt_f_score_hn = tf.reshape(self._f_score_hn,[-1,self._neg_per_positive])
            self._loss_hn = loss_hn = tf.reduce_mean(tf.square(f_score_hn), 1)
            
    
#            semi part
            _, self.semi_h_loss, self.semi_h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_h_batch,
                                               targets=self._semi_h_batch)
            _, self.semi_r_loss, self.semi_r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_r_batch,
                                               targets=self._semi_r_batch)
            _, self.semi_t_loss, self.semi_t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_t_batch,
                                               targets=self._semi_t_batch)
#             print(tf.reduce_sum(self.hn_encoded_features, 1))
#             self.semi_htr = semi_htr = tf.reduce_sum(
#             tf.multiply(self.semi_h_encoded_features, tf.multiply(self.semi_r_encoded_features, self.semi_t_encoded_features, 
#                                                              "element_wise_multiply_tneg"),"r_product_tneg"),1)
#             self._f_score_semi = f_score_semi = tf.sigmoid(self.w_main * (semi_htr+ self.gamma) + self.b_main)
#             Sum Neg
#             print(self._f_score_semi)
#             print(self._A_semi_w)
#             attention_semi = self.attention(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features,self.weights_attention, self.biases_attention)
            attention_semi = self.attention(self._semi_h_batch,self._semi_r_batch,self._semi_t_batch)
#             print(attention_semi)
            self._f_score_semi, self._semi_kl = f_score_semi, semi_kl = self.score_function(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features,attention=attention_semi)
            self._semi_loss = semi_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_semi, self._A_semi_w)))
            self.sum_VAE_loss = (self.h_original_loss + self.r_original_loss + self.t_original_loss + self.semi_h_loss + self.semi_r_loss + self.semi_t_loss) * self.beta / 6
            self.regularizer = (self.VAE_E.get_regularized_loss()) * self._alpha # + self.VAE_R.get_regularized_loss()
            self.kl = (pos_kl+semi_kl)* self.delta *1e-7

        #         Ver 3: h*r*t
        if self._ver == 3:

#             self.weights_attention = tf.Variable(tf.truncated_normal(shape=(self._n_hidden[-1]*3, 1), stddev=0.5), dtype=tf.float32, name="weight_attention")
#             self.biases_attention = tf.Variable(0.0, dtype=tf.float32, name="bias_attention")
            self.weights_trans = {}
            self.biases_trans = {}
            self.weights_attention = {}
            self.biases_attention = {}
            self.weights_score = {}
            self.biases_score = {}
            for i in range(len(self.score)-1):
                self.weights_score[i] = tf.Variable(tf.truncated_normal(shape=(self.score[i],self.score[i+1]), stddev=0.5), dtype=tf.float32, name="weights_score")
                self.biases_score[i] = tf.Variable(tf.zeros(shape=(self.score[i+1])), dtype=tf.float32, name="biases_score")
            for i in range(self.head):
                
                self.weights_trans[i] = tf.Variable(tf.truncated_normal(shape=(self._n_hidden[-1],self._n_hidden[-1]), stddev=0.5), dtype=tf.float32, name="weight_attention")
                self.biases_trans[i] = tf.Variable(tf.zeros(shape=(self._n_hidden[-1])), dtype=tf.float32, name="bias_attention")
                self.weights_attention[i] = tf.Variable(tf.truncated_normal(shape=(self._n_hidden[-1]*3, 1), stddev=0.5), dtype=tf.float32, name="weight_attention")
                self.biases_attention[i] = tf.Variable(tf.zeros(shape=(1)), dtype=tf.float32, name="bias_attention")
#             self.VAE_R = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
            self.VAE_E = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
#           Pos Sample [batch_size,dim]
            # pos_combined_features = tf.add(self._h_batch,self._r_batch)
            _, self.h_original_loss, self.h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._h_batch,
                                               targets=self._h_batch)
#             print(self._r_batch)
            _, self.r_original_loss, self.r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._r_batch,
                                               targets=self._r_batch)
            _, self.t_original_loss, self.t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._t_batch,
                                               targets=self._t_batch)
            
#             self._htr = htr = tf.reduce_sum(
#             tf.multiply(self.h_encoded_features, tf.multiply(self.r_encoded_features, self.t_encoded_features, 
#                                                              "element_wise_multiply_pos"),"r_product_pos"),1)
#             self._f_score_h = f_score_h = tf.sigmoid(self.get_dense_layer(htr, self.Denseweight, self.Densebiases, activation=None))
            attention= self.attention(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features)
#             attention= self.attention(self._h_batch,self._r_batch,self._t_batch)
#             print(attention)
            self._f_score_h, self._pos_kl = f_score_h, pos_kl  = self.MLP_score_function(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features,attention=attention)
            self._pos_loss = pos_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_h, self._A_w)))
        
        
#             print(self._pos_loss)
#             neg sample with VAE (h',r,t) [batch_size, neg_per_positive, dim]
#           only for evaluation
            self._neg_hn_con_batch = tf.reshape(self._neg_hn_con_batch,[-1,self._dim])
            self._neg_rel_hn_batch = tf.reshape(self._neg_rel_hn_batch,[-1,self._dim])
            self._neg_t_con_batch = tf.reshape(self._neg_t_con_batch,[-1,self._dim])
            _, self.neg_hn_loss,_ ,self.neg_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_hn_con_batch,
                                               targets=self._neg_hn_con_batch)
            _, self.neg_r_hn_loss,_ ,self.neg_r_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_rel_hn_batch,
                                               targets=self._neg_rel_hn_batch)
            _, self.neg_t_loss,_ ,self.neg_t_encoded_features = self.VAE_E.structure(
                                               features=self._neg_t_con_batch,
                                               targets=self._neg_t_con_batch)
            
#             self.hn_htr = hn_htr = tf.reduce_sum(
#             tf.multiply(self.neg_hn_encoded_features, tf.multiply(self.neg_r_hn_encoded_features, self.neg_t_encoded_features, 
#                                                              "element_wise_multiply_hneg"),"r_product_hneg"),1)
#             self._f_score_hn = f_score_hn = tf.sigmoid(self.w_main * (hn_htr + self.gamma) + self.b_main)
            attention_neg = self.attention(self.neg_hn_encoded_features,self.neg_r_hn_encoded_features,self.neg_t_encoded_features)
#             attention_neg = self.attention(self._neg_hn_con_batch,self._neg_rel_hn_batch,self._neg_t_con_batch)
#             print(attention_neg)
            pre_cvt_score_hn, neg_kl = self.MLP_score_function(self.neg_hn_encoded_features, self.neg_r_hn_encoded_features, self.neg_t_encoded_features, attention=attention_neg)
            self._f_score_hn = f_score_hn  = tf.reshape(pre_cvt_score_hn, [-1,self._neg_per_positive])
#             self.cvt_f_score_hn = tf.reshape(self._f_score_hn,[-1,self._neg_per_positive])
            self._neg_loss = loss_hn = tf.reduce_mean(tf.square(pre_cvt_score_hn))
            self._neg_VAE_loss = (self.neg_hn_loss+self.neg_r_hn_loss+self.neg_t_loss)
    
#            semi part
            _, self.semi_h_loss, self.semi_h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_h_batch,
                                               targets=self._semi_h_batch)
            _, self.semi_r_loss, self.semi_r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_r_batch,
                                               targets=self._semi_r_batch)
            _, self.semi_t_loss, self.semi_t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_t_batch,
                                               targets=self._semi_t_batch)
#             print(tf.reduce_sum(self.hn_encoded_features, 1))
#             self.semi_htr = semi_htr = tf.reduce_sum(
#             tf.multiply(self.semi_h_encoded_features, tf.multiply(self.semi_r_encoded_features, self.semi_t_encoded_features, 
#                                                              "element_wise_multiply_tneg"),"r_product_tneg"),1)
#             self._f_score_semi = f_score_semi = tf.sigmoid(self.w_main * (semi_htr+ self.gamma) + self.b_main)
#             Sum Neg
#             print(self._f_score_semi)
#             print(self._A_semi_w)
            attention_semi = self.attention(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features)
#             attention_semi = self.attention(self._semi_h_batch,self._semi_r_batch,self._semi_t_batch)
#             print(attention_semi)
            self._f_score_semi, self._semi_kl = f_score_semi, semi_kl = self.MLP_score_function(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features,attention=attention_semi)
            self._semi_loss = semi_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_semi, self._A_semi_w)))
            self._semi_VAE_loss = self.semi_h_loss + self.semi_r_loss + self.semi_t_loss
            self.sum_VAE_loss = (self.h_original_loss + self.r_original_loss + self.t_original_loss + self._neg_VAE_loss) * self.beta / 6
            self.regularizer = (self.VAE_E.get_regularized_loss()) * self._alpha # + self.VAE_R.get_regularized_loss()
            self.kl = -(pos_kl+semi_kl) * self.delta * 1e-7
            
        if self._ver == 4:

            self.weights_attention = tf.Variable(tf.truncated_normal(shape=([self.seperator]), stddev=0.5), dtype=tf.float32, name="weight_attention")
            self.biases_attention =  tf.Variable(tf.zeros(shape=(1)), dtype=tf.float32, name="bias_attention")
#             self.VAE_R = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
            self.VAE_E = VAE(dim=self._dim, n_hidden=self._n_hidden, mask_ratio=self.mask_ratio,keep_prob=self.keep_prob, seperator=self.seperator)
#           Pos Sample [batch_size,dim]
            # pos_combined_features = tf.add(self._h_batch,self._r_batch)
            _, self.h_original_loss, self.h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._h_batch,
                                               targets=self._h_batch)
#             print(self._r_batch)
            _, self.r_original_loss, self.r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._r_batch,
                                               targets=self._r_batch)
            _, self.t_original_loss, self.t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._t_batch,
                                               targets=self._t_batch)
            
#             self._htr = htr = tf.reduce_sum(
#             tf.multiply(self.h_encoded_features, tf.multiply(self.r_encoded_features, self.t_encoded_features, 
#                                                              "element_wise_multiply_pos"),"r_product_pos"),1)
#             self._f_score_h = f_score_h = tf.sigmoid(self.get_dense_layer(htr, self.Denseweight, self.Densebiases, activation=None))
#             attention= self.attention(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features,self.weights_attention, self.biases_attention)
            attention= self.weights_attention
#             print(attention)
            self._f_score_h, self._pos_kl = f_score_h, pos_kl  = self.score_function(self.h_encoded_features,self.r_encoded_features,self.t_encoded_features,attention=attention)
            self._pos_loss = pos_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_h, self._A_w)))
        
        
#             print(self._pos_loss)
#             neg sample with VAE (h',r,t) [batch_size, neg_per_positive, dim]
#           only for evaluation
            self._neg_hn_con_batch = tf.reshape(self._neg_hn_con_batch,[-1,self._dim])
            self._neg_rel_hn_batch = tf.reshape(self._neg_rel_hn_batch,[-1,self._dim])
            self._neg_t_con_batch = tf.reshape(self._neg_t_con_batch,[-1,self._dim])
            _, self.neg_hn_loss,_, self.neg_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_hn_con_batch,
                                               targets=self._neg_hn_con_batch)
            _, self.neg_r_hn_loss,_, self.neg_r_hn_encoded_features = self.VAE_E.structure(
                                               features=self._neg_rel_hn_batch,
                                               targets=self._neg_rel_hn_batch)
            _, self.neg_t_loss,_, self.neg_t_encoded_features = self.VAE_E.structure(
                                               features=self._neg_t_con_batch,
                                               targets=self._neg_t_con_batch)
            
#             self.hn_htr = hn_htr = tf.reduce_sum(
#             tf.multiply(self.neg_hn_encoded_features, tf.multiply(self.neg_r_hn_encoded_features, self.neg_t_encoded_features, 
#                                                              "element_wise_multiply_hneg"),"r_product_hneg"),1)
#             self._f_score_hn = f_score_hn = tf.sigmoid(self.w_main * (hn_htr + self.gamma) + self.b_main)
#             attention_neg = self.attention(self.neg_hn_encoded_features,self.neg_r_hn_encoded_features,self.neg_t_encoded_features,self.weights_attention, self.biases_attention )
            attention_neg = self.weights_attention + self.biases_attention
#             print(attention_neg)
            pre_cvt_score_hn, neg_kl = self.score_function(self.neg_hn_encoded_features, self.neg_r_hn_encoded_features, self.neg_t_encoded_features, attention=attention_neg)
            self._f_score_hn = f_score_hn  = tf.reshape(pre_cvt_score_hn, [-1,self._neg_per_positive])
#             self.cvt_f_score_hn = tf.reshape(self._f_score_hn,[-1,self._neg_per_positive])
            self._loss_hn = loss_hn = tf.reduce_mean(tf.square(f_score_hn), 1)
            
    
#            semi part
            _, self.semi_h_loss, self.semi_h_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_h_batch,
                                               targets=self._semi_h_batch)
            _, self.semi_r_loss, self.semi_r_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_r_batch,
                                               targets=self._semi_r_batch)
            _, self.semi_t_loss, self.semi_t_encoded_features,_ = self.VAE_E.structure(
                                               features=self._semi_t_batch,
                                               targets=self._semi_t_batch)
#             print(tf.reduce_sum(self.hn_encoded_features, 1))
#             self.semi_htr = semi_htr = tf.reduce_sum(
#             tf.multiply(self.semi_h_encoded_features, tf.multiply(self.semi_r_encoded_features, self.semi_t_encoded_features, 
#                                                              "element_wise_multiply_tneg"),"r_product_tneg"),1)
#             self._f_score_semi = f_score_semi = tf.sigmoid(self.w_main * (semi_htr+ self.gamma) + self.b_main)
#             Sum Neg
#             print(self._f_score_semi)
#             print(self._A_semi_w)
#             attention_semi = self.attention(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features,self.weights_attention, self.biases_attention)
            attention_semi = self.weights_attention + self.biases_attention
#             print(attention_semi)
            self._f_score_semi, self._semi_kl = f_score_semi, semi_kl = self.score_function(self.semi_h_encoded_features,self.semi_r_encoded_features,self.semi_t_encoded_features,attention=attention_semi)
            self._semi_loss = semi_loss = tf.reduce_mean(tf.square(tf.subtract(self._f_score_semi, self._A_semi_w)))
            self.sum_VAE_loss = (self.h_original_loss + self.r_original_loss + self.t_original_loss + self.semi_h_loss + self.semi_r_loss + self.semi_t_loss) * self.beta / 6
            self.regularizer = (self.VAE_E.get_regularized_loss()) * self._alpha # + self.VAE_R.get_regularized_loss()
            self.kl = (pos_kl+semi_kl) * self.delta * 0
            
#         if self._ver == 5:
#             self._pos_loss = tf.cast(0,tf.float32)
#             self._semi_loss = tf.cast(0,tf.float32)
#             self.sum_VAE_loss = tf.cast(0,tf.float32)
#       Loss
        
        self._hinge_loss = tf.cast(0,tf.float32)
#         self._hinge_loss = tf.reduce_mean(tf.math.maximum(0., tf.subtract(self._f_score_hn, tf.tile(tf.expand_dims(self._f_score_h, -1),  [1, self._neg_per_positive])) + self.margin)) * 1e-2
        
        self.main_loss = self._pos_loss + self._neg_loss + self.sum_VAE_loss + self._hinge_loss + self.regularizer + self.kl + self._hinge_loss
#         print(self.main_loss)
        
            
        tf.summary.scalar('w_main', self.w_main)
        tf.summary.scalar('b_main', self.b_main)
        tf.summary.histogram('attention', attention)
        tf.summary.histogram('attention_argmax',tf.math.argmax(attention,-1))
        
        tf.summary.histogram('pos', self._f_score_h)
        tf.summary.histogram('pos_label', self._A_w)
        tf.summary.histogram('semi', self._f_score_semi)
        tf.summary.histogram('semi_label', self._A_semi_w)
        tf.summary.histogram('neg', self._f_score_hn)
        tf.summary.scalar('kl', self.kl)
        tf.summary.scalar('pos_loss', self._pos_loss)
        tf.summary.scalar('semi_loss', self._semi_loss)
        tf.summary.scalar('hinge_loss', self._hinge_loss)
        tf.summary.scalar('VAE_loss', self.sum_VAE_loss)
        tf.summary.scalar('regularizer', self.regularizer)
        tf.summary.scalar('loss', self.main_loss)
        self._sum_op = tf.summary.merge_all()
    
    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts
    
    
class VAE():
    def __init__(self, dim, n_hidden, mask_ratio,keep_prob, seperator):
        self.n_hidden = n_hidden
        self.weights = None
        self.biases = None
        self._dim = dim
        self.mask_ratio = mask_ratio
        self.keep_prob = keep_prob
        self.seperator = seperator
    def contrastive(self, features, targets):
        a = self.mask_data(features, self.mask_ratio)
        b = self.mask_data(features, self.mask_ratio)
        _, a_loss, a_feature = self.structure(
                                               features=a,
                                               targets=a)
        _, b_loss, b_feature = self.structure(
                                               features=b,
                                               targets=b)
        contrastive_l = tf.reduce_sum(tf.square(tf.reduce_sum(tf.subtract(a_feature, b_feature),1)))
        loss = contrastive_l + (a_loss + b_loss)*0.1
        y_, b_loss, features = self.structure(
                                               features=features,
                                               targets=targets)
        return (y_, loss, features)
        
    def structure(self, features, targets):
        seperator = self.seperator
        ### Variable
        if (not self.weights) and (not self.biases):
            with tf.variable_scope("VAE", initializer=tf.truncated_normal_initializer(-0.5, 0.5)):
                print("generate weight")
                self.weights = {}
                self.biases = {}
                
                n_encoder = [self._dim]+self.n_hidden #[256 128 64]
                for i, n in enumerate(n_encoder[:-1]):
                    if i < len(n_encoder)-2:
                        for j in range(seperator):
                            self.weights['encode{}-{}'.format(i+1,j+1)] = \
                                tf.Variable(tf.truncated_normal(
                                    shape=(n, n_encoder[i+1]), stddev=0.1), dtype=tf.float32)
                            self.biases['encode{}-{}'.format(i+1,j+1)] = \
                                tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)
#                             tf.summary.histogram('encode_weight{}-{}'.format(i+1,j+1),self.weights['encode{}-{}'.format(i+1,j+1)])
#                             tf.summary.histogram('encode_bias{}-{}'.format(i+1,j+1),self.biases['encode{}-{}'.format(i+1,j+1)])
                    else:
                        for j in range(seperator):
                            self.weights['encode{}-{}'.format(i+1,j+1)] = \
                                tf.Variable(tf.truncated_normal(
                                    shape=(n, n_encoder[i+1]), stddev=0.1), dtype=tf.float32)
                            self.biases['encode{}-{}'.format(i+1,j+1)] = \
                                tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)
#                             tf.summary.histogram('encode_weight{}-{}'.format(i+1,j+1),self.weights['encode{}-{}'.format(i+1,j+1)])
#                             tf.summary.histogram('encode_bias{}-{}'.format(i+1,j+1),self.biases['encode{}-{}'.format(i+1,j+1)])

                
                n_decoder = list(reversed(n_encoder))
                for i, n in enumerate(n_decoder[:-1]):
                    for j in range(seperator):
                        self.weights['decode{}-{}'.format(i+1, j+1)] = \
                            tf.Variable(tf.truncated_normal(
                                shape=(n, n_decoder[i+1]), stddev=0.1), dtype=tf.float32)
                        self.biases['decode{}-{}'.format(i+1, j+1)] = \
                            tf.Variable(tf.zeros(shape=(n_decoder[i+1])), dtype=tf.float32)
#                         tf.summary.histogram('decode_weight{}-{}'.format(i+1,j+1),self.weights['decode{}-{}'.format(i+1,j+1)])
#                         tf.summary.histogram('decode_bias{}-{}'.format(i+1,j+1),self.biases['decode{}-{}'.format(i+1,j+1)])
                
        activation = tf.nn.leaky_relu
            
#         print(features.shape)
#         print(features)
        encoder = []
#         feature = tf.split(features, num_or_size_splits=seperator, axis=1)
        for j in range(seperator):
            encoder.append(self.get_dense_layer(features,
                                           self.weights['encode1-{}'.format(j+1)],
                                           self.biases['encode1-{}'.format(j+1)],
                                           activation=activation))
    
        for i in range(1, len(self.n_hidden)-1):
            for j in range(seperator):
                encoder[j] = self.get_dense_layer(
                    encoder[j],
                    self.weights['encode{}-{}'.format(i+1, j+1)],
                    self.biases['encode{}-{}'.format(i+1, j+1)],
                    activation=activation,
                )
#                 encoder[j] = tf.nn.dropout(encoder[j], rate=0.1)
        encoder_mu = []
        encoder_rho = []
        sd = []
        eps = []
        Z_sample = []
        for j in range(seperator):
            encoder_mu.append(self.get_dense_layer(
                encoder[j],
                self.weights['encode{}-{}'.format(len(self.n_hidden), j+1)],
                self.biases['encode{}-{}'.format(len(self.n_hidden), j+1)],
            ))
    #         print(encoder_mu)
            encoder_rho.append(self.get_dense_layer(
                encoder[j],
                self.weights['encode{}-{}'.format(len(self.n_hidden), j+1)],
                self.biases['encode{}-{}'.format(len(self.n_hidden), j+1)],
            ))
    #         print(encoder_mu)
    #         encoder_mu, encoder_rho = tf.split(encoder, num_or_size_splits=2, axis=1)
            sd.append(tf.math.log(1 + tf.math.exp(encoder_rho[j])))
            eps.append(tf.random_normal(shape=tf.shape(encoder_mu[j]),mean=0, stddev=0.3, dtype=tf.float32))
            Z_sample.append(encoder_mu[j] + sd[j] * eps[j])
    #         print(sd * eps)
    #         print(Z_sample)
        decoder = []
        for j in range(seperator):
            decoder.append(self.get_dense_layer(Z_sample[j],
                                           self.weights['decode1-{}'.format(j+1)],
                                           self.biases['decode1-{}'.format(j+1)],
                                           activation=activation))
#         print(decoder)
        for i in range(1, len(self.n_hidden)-1):
            for j in range(seperator):
                decoder[j] = self.get_dense_layer(
                    decoder[j],
                    self.weights['decode{}-{}'.format(i+1, j+1)],
                    self.biases['decode{}-{}'.format(i+1, j+1)],
                    activation=activation,
                )
#                 decoder[j] = tf.nn.dropout(decoder[j], rate=0.1)
#         print(decoder)
#         print(self.weights['decode{}'.format(len(self.n_hidden)-1)])
#         print(self.biases['decode{}'.format(len(self.n_hidden)-1)])
        y_ = []
        for j in range(seperator):
            y_.append(self.get_dense_layer(
                decoder[j],
                self.weights['decode{}-{}'.format(len(self.n_hidden), j+1)],
                self.biases['decode{}-{}'.format(len(self.n_hidden), j+1)],
    #             activation=tf.nn.sigmoid,
            ))
#         print(y_)
        loss = 0
        for j in range(seperator):
            y_[j] = tf.reshape(y_[j],tf.shape(targets))
            kl_loss = -0.5*(encoder_rho[j]+1-encoder_mu[j]**2-tf.exp(encoder_rho[j]))
            loss += tf.reduce_mean(tf.pow(targets - y_[j], 2))+ tf.reduce_mean(kl_loss)
        loss/= seperator
        return (y_, loss, Z_sample, encoder_mu)
#               Conv VAE
#                 self.n_encoder = n_encoder = [1] + self.n_hidden # depth [8, 16, 32]
#                 for i, n in enumerate(n_encoder[:-1]): # 1d conv 
# #                     print(i)
#                     if i < len(n_encoder)-2:
# #                         print(i, n_encoder[i+1])
#                         self.weights['encode{}'.format(i+1)] = \
#                             tf.Variable(tf.truncated_normal(
#                                 shape=(3, n_encoder[i],n_encoder[i+1]), stddev=0.3), dtype=tf.float32)
#                         self.biases['encode{}'.format(i+1)] = \
#                             tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)
#                     else:
#                         self.weights['encode{}'.format(i+1)] = \
#                             tf.Variable(tf.truncated_normal(
#                                 shape=(1, n_encoder[i], n_encoder[i+1]), stddev=0.3), dtype=tf.float32)
#                         self.biases['encode{}'.format(i+1)] = \
#                             tf.Variable(tf.zeros(shape=(n_encoder[i+1])), dtype=tf.float32)


#                 self.n_decoder= n_decoder = list(reversed(n_encoder))
# #                 print(n_decoder)
#                 for i, n in enumerate(n_decoder[:]):
#                     if i == 0:
# #                         print(i, n_encoder[i+1])
#                         self.weights['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.truncated_normal(
#                                 shape=(3, n_decoder[i],n_decoder[i]), stddev=0.3), dtype=tf.float32)
#                         self.biases['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.zeros(shape=(n_decoder[i])), dtype=tf.float32)
#                     elif i == len(n_decoder)-1:
#                         self.weights['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.truncated_normal(
#                                 shape=(1, n_decoder[i], n_decoder[i-1]), stddev=0.3), dtype=tf.float32)
#                         self.biases['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.zeros(shape=(n_decoder[i])), dtype=tf.float32)
#                     else:
#                         self.weights['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.truncated_normal(
#                                 shape=(3, n_decoder[i], n_decoder[i-1]), stddev=0.3), dtype=tf.float32)
#                         self.biases['decode{}'.format(i+1)] = \
#                             tf.Variable(tf.zeros(shape=(n_decoder[i])), dtype=tf.float32)
# #         print(self.weights['encode1'])
#         ### Structure
#         activation = tf.nn.leaky_relu
# #         print(features.shape)
#         trans_features = tf.reshape(features,[-1,features.shape[1],1])
# #         print(trans_features)
#         encoder = self.get_conv_layer(trans_features,
#                                        self.weights['encode1'],
#                                        self.biases['encode1'],
#                                        activation=activation,
#                                        deconv=False)
# #         print(encoder)
#         for i in range(1, len(self.n_hidden)-1):
# #             print(i+1)
#             encoder = self.get_conv_layer(
#                 encoder,
#                 self.weights['encode{}'.format(i+1)],
#                 self.biases['encode{}'.format(i+1)],
#                 activation=activation,
#                 deconv=False,
#             )
        
#         encoder_mu = self.get_conv_layer(
#             encoder,
#             self.weights['encode{}'.format(len(self.n_hidden))],
#             self.biases['encode{}'.format(len(self.n_hidden))],
#             deconv=False,
#             stride=2
#         )
# #         print(encoder_mu)
#         encoder_rho = self.get_conv_layer(
#             encoder,
#             self.weights['encode{}'.format(len(self.n_hidden))],
#             self.biases['encode{}'.format(len(self.n_hidden))],
#             deconv=False,
#             stride=2
#         )
# #         print(encoder_mu)
# #         encoder_mu, encoder_rho = tf.split(encoder, num_or_size_splits=2, axis=1)
#         sd = tf.math.log(1 + tf.math.exp(encoder_rho))
#         eps = tf.random_normal(shape=tf.shape(encoder_mu),mean=0, stddev=0.2, dtype=tf.float32)
#         Z_sample = encoder_mu + sd * eps
# #         encoder_mu = tf.reshape(encoder_mu, [tf.shape(encoder_mu)[0],encoder_mu.shape[1]*encoder_mu.shape[2]])
# #         encoder_rho = tf.reshape(encoder_rho, [tf.shape(encoder_rho)[0],encoder_rho.shape[1]*encoder_rho.shape[2]])
        
# #         print(Z_sample)
#         ans = tf.layers.flatten(Z_sample)

#         batch_size = tf.shape(Z_sample)[0]
#         decoder = self.get_conv_layer(Z_sample,
#                                        self.weights['decode1'],
#                                        self.biases['decode1'],
#                                        activation=activation,
#                                        output_shape=tf.stack([batch_size,Z_sample.shape[1]*2,self.n_decoder[0]]),
#                                        deconv=True)
# #         print(decoder)
#         for i in range(1, len(self.n_hidden)):
# #             print(decoder)
#             decoder = self.get_conv_layer(
#                 decoder,
#                 self.weights['decode{}'.format(i+1)],
#                 self.biases['decode{}'.format(i+1)],
#                 activation=activation,
#                 output_shape=tf.stack([batch_size,decoder.shape[1]*2,self.n_decoder[i]]),
#                 deconv=True,)

#         y_ = self.get_conv_layer(
#             decoder,
#             self.weights['decode{}'.format(len(self.n_decoder))],
#             self.biases['decode{}'.format(len(self.n_decoder))],
#             deconv=True,
#             output_shape=tf.stack([batch_size, decoder.shape[1], self.n_decoder[-1]]),
#             stride=1,
#             b=True
# #             activation=tf.nn.sigmoid,
#         )
#         print(y_)
#         y_ = tf.reshape(y_,tf.shape(targets))
        
#         kl_loss = -0.5*(encoder_rho+1-encoder_mu**2-tf.exp(encoder_rho))
# #         print(kl_loss)
#         loss = tf.reduce_mean(tf.pow(targets - y_, 2))+ tf.reduce_mean(kl_loss)
# #         print(ans)
# #         print("end")
#         return (y_, loss, ans)

    def get_dense_layer(self, input_layer, weight, bias, activation=None):
        x = tf.add(tf.matmul(input_layer, weight), bias)
        if activation:
            x = activation(x)
        return x
    def dropout(self, nodes, keep_prob):
        if keep_prob == 0:
            return tf.zeros_like(nodes)

        mask = tf.random_uniform(tf.shape(nodes)) < keep_prob
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.divide(tf.multiply(mask, nodes), keep_prob)

    def get_conv_layer(self, input_layer, weight, bias, activation=None, deconv=False,output_shape=None,stride=2,b=True):
        if deconv:
            x = tf.nn.conv1d_transpose(input_layer, weight,strides=stride, output_shape=output_shape, padding='SAME')
#             print(x)
        else:
            x = tf.nn.conv1d(input_layer, weight,stride=stride,padding='SAME')
#             print(x)
        if b:
#             print(bias)
            x = tf.add(x, bias)
        if activation:
            x = activation(x)
        return x
    def get_regularized_loss(self):
        return tf.reduce_sum([tf.reduce_sum(
                        tf.pow(w, 2)/(1+tf.pow(w, 2))) for w in self.weights.values()]) \
                / tf.reduce_sum(
                    [tf.size(w, out_type=tf.float32) for w in self.weights.values()])
    
    def mask_data(self, y_true, mask_ratio, verbose=0):
#         print(y_true.get_shape())
        nf = tf.cast(y_true.get_shape()[1], tf.float32)
        mask_portion = tf.math.round( tf.math.multiply(nf,(1-mask_ratio)) )
        mask_portion = tf.cast(mask_portion, tf.int32)

        z = -tf.math.log(-tf.math.log(tf.random.uniform((1,y_true.get_shape()[1]),0,1))) 
        _, indices = tf.nn.top_k(z, mask_portion)
#         print(indices.get_shape())
        one_hots = tf.one_hot(indices, y_true.get_shape()[1])
#         print(one_hots.get_shape())
        mask = tf.reduce_max(one_hots, axis=1)
#         mask = tf.expand_dims(mask,axis=-1)
#         print(mask)
#         print(y_true[-1])
        mask_tiles = tf.tile(mask,[tf.shape(y_true)[1],1])
#         print(mask_tiles.get_shape())
        masked = tf.multiply(mask_tiles,y_true)

        return masked
    
# def get_gaussian_filter(shape=(None, dim), sigma=0.5):
#     """build the gaussain filter"""
#     m,n = [(ss-1.)/2. for ss in shape[1]]
#     x = tf.expand_dims(tf.range(-n,n+1,dtype=tf.float32),1)
#     y = tf.expand_dims(tf.range(-m,m+1,dtype=tf.float32),0)
#     h = tf.exp(tf.math.divide_no_nan(-((x*x) + (y*y)), 2*sigma*sigma))
#     h = tf.math.divide_no_nan(h,tf.reduce_sum(h))
#     return h

# def gaussian_blur(inp, shape=(3,3), sigma=0.5):
#     """Convolve using tf.nn.depthwise_conv2d"""
#     in_channel = tf.shape(inp)[-1]
#     k = get_gaussian_kernel(shape,sigma)
#     k = tf.expand_dims(k,axis=-1)
#     k = tf.repeat(k,in_channel,axis=-1)
#     k = tf.reshape(k, (*shape, in_channel, 1))
#     # using padding same to preserve size (H,W) of the input
#     conv = tf.nn.depthwise_conv2d(inp, k, strides=[1,1,1,1],padding="SAME")
#     return conv

from layer_utils import *
from layer import *
class KEGCN_DistMult(TFParts):
    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl):
#         required objects
        
#         adjacency matrix
        TFParts.__init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, p_neg, psl)
        self.build()
    def score_function(self, h, t, r, numpy=False):
        htr = tf.reduce_sum(
            tf.multiply(r, tf.multiply(h, t, "element_wise_multiply"),
                        "r_product"),
            1)

        return tf.sigmoid(self.w_main * htr + self.b_main)
    def update_lookup(self,ht,r):
        self._h_batch = tf.nn.embedding_lookup(ht, self._A_h_index)
        self._t_batch = tf.nn.embedding_lookup(ht, self._A_t_index)
        self._r_batch = tf.nn.embedding_lookup(r, self._A_r_index)
        # index to embedding
        self._neg_hn_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_hn_index)
        self._neg_rel_hn_batch = tf.nn.embedding_lookup(r, self._A_neg_rel_hn_index)
        self._neg_t_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_t_index)
        self._neg_h_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_h_index)
        self._neg_rel_tn_batch = tf.nn.embedding_lookup(r, self._A_neg_rel_tn_index)
        self._neg_tn_con_batch = tf.nn.embedding_lookup(ht, self._A_neg_tn_index)

        self._semi_h_batch = tf.nn.embedding_lookup(ht, self._A_semi_h_index)
        self._semi_t_batch = tf.nn.embedding_lookup(ht, self._A_semi_t_index)
        self._semi_r_batch = tf.nn.embedding_lookup(r, self._A_semi_r_index)

        
    def define_main_loss(self):
        
        self.w_main = tf.Variable(0.0, name="weights_main")
        self.b_main = tf.Variable(0.0, name="bias_main")
        self.KEGCN = KEGCN(input_dim=self.dim,w_main = self.w_main, b_main= self.b_main)
#         self.output_ht = 
#         self.output_r = 
        print('define main loss')
        
#         new_ht, new_r = self.KEGCN.structure([self._ht, self._r])
#         print(self._ht)
        self.new_ht, self.new_r = self.KEGCN.structure([self._ht, self._r], [self._A_h_index, self._A_t_index, self._A_r_index, self._A_w, self._A_entinv, self._A_relinv])
        self.update_lookup(self.new_ht, self.new_r)
        self._f_score_h = f_score_h = self.score_function(
            self._h_batch,
            self._t_batch, 
            self._r_batch)
        self._pos_loss = tf.reduce_mean(tf.square(tf.subtract(f_score_h, self._A_w)))

        self._f_score_semi = f_score_semi = self.score_function(
            self._semi_h_batch,
            self._semi_t_batch, 
            self._semi_r_batch)

        self._semi_loss = tf.reduce_mean(tf.square(f_score_semi - self._A_semi_w))


        # evaluation only
        self._f_score_hn = _f_score_hn = tf.sigmoid(self.w_main * (
            tf.reduce_sum(
                tf.multiply(self._neg_rel_hn_batch, tf.multiply(self._neg_hn_con_batch, self._neg_t_con_batch)), 2)
        ) + self.b_main)
#         print(_f_score_hn)
        self._neg_loss = tf.reduce_mean(tf.square(_f_score_hn)) / self._neg_per_positive
        self._hinge_loss = tf.cast(0,tf.float32)
        self.sum_VAE_loss = tf.cast(0,tf.float32)
        self.regularizer = tf.cast(0,tf.float32)
#         self.main_loss = self._pos_loss + self._semi_loss + self._hinge_loss + self.beta * self.sum_VAE_loss + self._alpha * self.regularizer
        # self.main_loss = tf.add(tf.reduce_sum(f_score_semi)/self._neg_per_positive, tf.reduce_sum(f_score_h))
        self.main_loss = tf.add(self._semi_loss, self._pos_loss) 
#       KEGCN
        tf.summary.histogram('pos', self._f_score_h)
        tf.summary.histogram('pos_label', self._A_w)
        tf.summary.histogram('semi', self._f_score_semi)
        tf.summary.histogram('semi_label', self._A_semi_w)
        tf.summary.histogram('neg', self._f_score_hn)
        tf.summary.scalar('loss', self.main_loss)
        self._sum_op = tf.summary.merge_all()

    def define_psl_loss(self):
        self.w = tf.Variable(0.0, name="weights")
        self.b = tf.Variable(0.0, name="bias")
        
        self.psl_prob = tf.sigmoid(self.w*tf.reduce_sum(
            tf.multiply(self._soft_r_batch,
                        tf.multiply(self._soft_h_batch, self._soft_t_batch)),
            1)+self.b)
        self.compute_psl_loss()  # in tf_parts
        
class KEGCN():
    def __init__(self, input_dim, w_main, b_main, layer_num=1, dropout=0.05, mask_ratio=0.):
        self.input_dim = input_dim
        self.alpha = 0.5
        self.beta = 0.5
        self.act = tf.nn.leaky_relu
        self.mode = "DistMult"
        self.sparse_inputs = True
        self.layer_num = layer_num
        self.mask_ratio = mask_ratio
        self.loss = 0
        self.vars = {}
        self.highway = False
        self.normalize = True
        self.layers = [AutoRelConv(input_dim, 1, w_main, b_main) for i in range(layer_num)]
    
    def structure(self, inputs, quiry):
        ht_embed = inputs[0]
        rel_embed = inputs[1]
        for layer in self.layers:
            ht_embed, rel_embed = layer.call([ht_embed, rel_embed], quiry)
        return ht_embed, rel_embed
#      inputs: 
class AutoRelConv():
    def __init__(self, input_dim, w_main, b_main, dropout=0.05):
        self.inputs = input_dim
        self.alpha = 0.8
        self.beta = 0.8
        self.act = tf.nn.leaky_relu
        self.mode = "DistMult"
        self.sparse_inputs = True
        self.dropout = dropout
        self.loss = 0
        self.rel_update = True
        self.bias = True
        self.transform = True
        self.vars = {}
        self.highway = False
        self.normalize = True
        self.w_main = w_main
        self.b_main = b_main
        init=[glorot, glorot]
        with tf.variable_scope('KEGCN_vars'):
            # vars: embedding of nodes
            for i in range(1):
                if self.transform:
                    self.vars['ent_weights_' + str(i)] = init[0]([self.inputs, self.inputs],
                                                            name='ent_weights_' + str(i))
                    self.loss += tf.nn.l2_loss(self.vars['ent_weights_' + str(i)])

            if self.bias:
                self.vars['ent_bias'] = zeros([self.inputs], name='ent_bias')
                self.vars['rel_bias'] = zeros([self.inputs], name='rel_bias')

        if self.highway:
            self.kernel_gate_ent = glorot([self.inputs, self.inputs])
            self.bias_gate_ent = zeros([self.inputs])
    def score_function(self, h, t, r, numpy=False):
        htr = tf.reduce_sum(
            tf.multiply(r, tf.multiply(h, t, "element_wise_multiply"),
                        "r_product"),
            1)

        return tf.sigmoid(self.w_main * htr + self.b_main)
    def call(self, inputs, quiry):
        x = inputs
        # dropout
        if self.dropout:
            x[0] = tf.nn.dropout(x[0], 1-self.dropout)
            x[1] = tf.nn.dropout(x[1], 1-self.dropout)

        # convolve
        ent_supports = list()
        rel_supports = list()
        pre_ent = x[0]
        pre_rel = x[1]

        # normalize the relation embedding in RotatE and QuatE
        if self.mode == "RotatE":
            rel_shape = pre_rel.shape
            pre_rel = tf.math.l2_normalize(tf.reshape(pre_rel, [-1, 2]), axis=1)
            pre_rel = tf.reshape(pre_rel, rel_shape)
        elif self.mode == "QuatE":
            rel_shape = pre_rel.shape
            pre_rel = tf.math.l2_normalize(tf.reshape(pre_rel, [-1, 4]), axis=1)
            pre_rel = tf.reshape(pre_rel, rel_shape)

        h,r,t,w, ent_invsum, rel_invsum = quiry

        ent_message, rel_message, loss = self._message(pre_ent, pre_rel, h, r, t, w, self.mode)

        ent_update = ent_invsum * ent_message * w
        ent_support = pre_ent + self.alpha * ent_update 

        if rel_message is not None:
            rel_update = rel_invsum * rel_message
            rel_support = pre_rel + self.beta * rel_update
        else:
            rel_support = pre_rel

        if self.transform:
            ent_support = dot(ent_support, self.vars['ent_weights_' + str(0)])

        ent_supports.append(ent_support)
        rel_supports.append(rel_support)

        output_ent = tf.add_n(ent_supports)
        output_rel = tf.add_n(rel_supports)

        if self.bias:
            output_ent += self.vars['ent_bias']
            output_rel += self.vars['rel_bias']

        output_ent = self.act(output_ent)
        output_rel = self.act(output_rel)

        if self.normalize:
            output_ent = tf.math.l2_normalize(output_ent, axis=1)
            output_rel = tf.math.l2_normalize(output_rel, axis=1)

        return output_ent, output_rel
    

    def _message(self, ent_emb, rel_emb, h, r, t, w, mode="DistMult"):
        
        ent_head = tf.gather(ent_emb, h)
        ent_tail = tf.gather(ent_emb, t)
        rel = tf.gather(rel_emb, r)
        if mode == "None":
            loss = - tf.reduce_sum((ent_head - ent_tail)**2)
        elif mode == "TransE":
            loss = - tf.reduce_sum((ent_head + rel - ent_tail)**2)
        elif mode == "TransH":
            rel_dim = tf.cast(tf.shape(rel)[1]/2, tf.int32)
            rel_1, rel_2 = rel[:, :rel_dim], rel[:, rel_dim:]
            rel_2_norm = tf.math.l2_normalize(rel_2, axis=1)
            ent_head_new = ent_head - tf.reduce_sum(ent_head * rel_2_norm, 1, True)/10. * rel_2_norm
            ent_tail_new = ent_tail - tf.reduce_sum(ent_tail * rel_2_norm, 1, True)/10. * rel_2_norm
            loss = - tf.reduce_sum((ent_head_new + rel_1 - ent_tail_new)**2)
        elif mode == "TransD":
            rel_dim = tf.cast(tf.shape(rel_emb)[1]/2, tf.int32)
            rel_1, rel_2 = rel[:, :rel_dim], rel[:, rel_dim:]
            ent_dim = tf.cast(tf.shape(ent_emb)[1]/2, tf.int32)
            ent_head_1, ent_head_2 = ent_head[:, :ent_dim], ent_head[:, ent_dim:]
            ent_tail_1, ent_tail_2 = ent_tail[:, :ent_dim], ent_tail[:, ent_dim:]
            ent_head_new = ent_head_1 - tf.reduce_sum(ent_head_1 * ent_head_2, 1, True)/10. * rel_2
            ent_tail_new = ent_tail_1 - tf.reduce_sum(ent_tail_1 * ent_tail_2, 1, True)/10. * rel_2
            loss = - tf.reduce_sum((ent_head_new + rel_1 - ent_tail_new)**2)
        elif mode == "DistMult":
            if param.ver != 2:
                loss = tf.reduce_sum(self.score_function(ent_head, rel, ent_tail))
            else:
                loss = -tf.reduce_sum(self.score_function(ent_head, rel, ent_tail))
        elif mode == "RotatE":
            loss = -tf.reduce_sum((multiply_complex(ent_head, rel) - ent_tail)**2)
        elif mode == "QuatE":
            loss = -tf.reduce_sum((multiply_quater(ent_head, rel) - ent_tail)**2)
#         print(loss)
        ent_message, rel_message = tf.gradients(loss, [ent_emb, rel_emb])
        if mode == "None" or self.rel_update == False:
            rel_message = None
#         print(ent_message)
#         print(rel_message)
        return ent_message, rel_message, loss