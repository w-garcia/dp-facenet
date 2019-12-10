from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
import os

from insightface.recognition.backbones.resnet_v1 import ResNet_v1_50
from insightface.recognition.data.generate_data import GenerateData
from insightface.recognition.models.models import MyModel
from insightface.recognition.predict import get_embeddings
from numpy.random import RandomState

# from cleverhans.compat import flag
# from cleverhans.model import Model
# from cleverhans.utils_tf import model_eval
# from cleverhans.attacks import FastGradientMethod
# from cleverhans.utils import AccuracyReport, set_log_level

from query_hop_skip_jump import query_hop_skip_jump
from custom_logging import get_logger
from util import pickle_write

tf.enable_eager_execution()

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

SEED = 9
prng = RandomState(SEED)


class InsightfaceOracle:
    def __init__(self, vd):
        self.vd = vd
        self.model = vd.model
        self.victim = None
    
    def set_victim(self, xv):
        xv = tf.reshape(xv, (1,) + tuple(tf.shape(xv).numpy()))
        self.victim = xv

    def query(self, x, thresh=0.2):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        sim = self.white_box_query(x)
        predict = tf.greater_equal(sim, thresh)

        return predict

    def white_box_query(self, x):
        xv = self.victim
        orig_shape = tuple(tf.shape(x).numpy())
        if len(orig_shape) < 4:
            x = tf.reshape(x, (1,) + tuple(tf.shape(x).numpy()))

        emb1 = get_embeddings(self.model, x)
        emb2 = get_embeddings(self.model, xv)
        sim = self.vd._cal_cos_sim(emb1, emb2)
        
        if len(orig_shape) < 4:
            sim = sim[0]

        return sim


def _cal_metric(sim, label, thresh):
        tp = tn = fp = fn = 0
        predict = tf.greater_equal(sim, thresh)
        for i in range(len(predict)):
            if predict[i] and label[i]:
                tp += 1
            elif predict[i] and not label[i]:
                fp += 1
            elif not predict[i] and label[i]:
                fn += 1
            else:
                tn += 1
        acc = (tp + tn) / len(predict)
        p = 0 if tp + fp == 0 else tp / (tp + fp)
        r = 0 if tp + fn == 0 else tp / (tp + fn)
        fpr = 0 if fp + tn == 0 else fp / (fp + tn)
        return acc, p, r, fpr


class Valid_Data:
    def __init__(self, tag, model, data, save_dir):
        self.tag = tag
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.logger = get_logger(__name__)

    @staticmethod
    def _cal_cos_sim(emb1, emb2):
        return tf.reduce_sum(emb1 * emb2, axis=-1)
    
    def attack(x):
        return self.attack_obj.generate(x, **hsja_params)
    
    def _get_sim_label(self):
        sims = None
        labels = None
        for image1, image2, label in self.data:
            emb1 = get_embeddings(self.model, image1)
            emb2 = get_embeddings(self.model, image2)
            sim = self._cal_cos_sim(emb1, emb2)
            if sims is None:
                sims = sim
            else:
                sims = tf.concat([sims, sim], axis=0)

            if labels is None:
                labels = label
            else:
                labels = tf.concat([labels, label], axis=0)

        return sims, labels
    
    def _get_adv_sim_label(self, oracle):
        sims = None
        labels = None
        l_queries = []
        for image1, image2, label in self.data:
            # data[:n/2]: label is True
            # data[n/2:]: label is False
            sims_ = []
            labels_ = []
            for image1_single, image2_single, label_single in zip(image1, image2, label):
                oracle.set_victim(image2_single)
                x_adv, queries = query_hop_skip_jump([image1_single], oracle)
                l_queries.append(queries)

                if len(x_adv) != 0:
                    adv_image1 = x_adv[0]
                else:
                    continue

                self.logger.info(f"Finished HSJA with {queries} queries")
                sims_.append(oracle.white_box_query(adv_image1))
                labels_.append(label_single)

                # if len(sims_) >= 4:
                #     break
                
            sim = sims_
            label = labels_

            if sims is None:
                sims = sim
            else:
                sims = tf.concat([sims, sim], axis=0)

            if labels is None:
                labels = label
            else:
                labels = tf.concat([labels, label], axis=0)
            
            # break
            if len(sims) >= 128:
                break

        return sims, labels, l_queries
    
    def _cal_metric_fpr(self, sim, label, below_fpr=0.001):
        acc = p = r = thresh = 0
        for t in np.linspace(-1, 1, 100):
            thresh = t
            acc, p, r, fpr = _cal_metric(sim, label, thresh)
            if fpr <= below_fpr:
                break

        return acc, p, r, thresh

    def get_metric_clean(self, thresh=0.2, below_fpr=0.001):
        sim, label = self._get_sim_label()
        pickle_write(os.path.join(self.save_dir, "clean_simlabel.pkl"), (sim, label))

        acc, p, r, fpr = _cal_metric(sim, label, thresh)
        acc_fpr, p_fpr, r_fpr, thresh_fpr = self._cal_metric_fpr(sim, label, below_fpr)
        return acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr

    def get_metric_adv(self, oracle, thresh=0.2, below_fpr=0.001):
        sim, label, queries = self._get_adv_sim_label(oracle)
        pickle_write(os.path.join(self.save_dir, "adv_simlabelqueries.pkl"), (sim, label, queries))
        
        acc, p, r, fpr = _cal_metric(sim, label, thresh)
        acc_fpr, p_fpr, r_fpr, thresh_fpr = self._cal_metric_fpr(sim, label, below_fpr)
        return acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr  


def parse_args(argv):
    parser = argparse.ArgumentParser(description='valid model')
    parser.add_argument('--config_path', type=str, help='path to config path', default='configs/config.yaml')
    parser.add_argument('--save_dir', type=str, help='path to save dir', default='save/')
    args = parser.parse_args(argv)

    return args


def main():
    args = parse_args(sys.argv[1:])
    # logger.info(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(args.config_path) as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    gd = GenerateData(config)
    valid_data = gd.get_val_data(config['valid_num'])
    model = MyModel(ResNet_v1_50, embedding_size=config['embedding_size'])
    
    ckpt_dir = os.path.join("insightface/recognition", os.path.expanduser(config['ckpt_dir']))
    ckpt = tf.train.Checkpoint(backbone=model.backbone)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_dir)))
    
    tag = args.config_path.split('/')[-1].replace(".yaml", "")
    vd_save_dir = os.path.join(args.save_dir, tag)
    if not os.path.exists(vd_save_dir):
        os.makedirs(vd_save_dir)

    vd = Valid_Data(tag, model, valid_data, vd_save_dir)
    thresh, fpr_rate = 0.2, 0.001
    oracle = InsightfaceOracle(vd)
    acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = vd.get_metric_adv(oracle, thresh, fpr_rate)
    print(f"adv: acc={acc}, p={p}, r={r}, fpr={fpr}, acc_fpr={acc_fpr}, p_fpr={p_fpr}, r_fpr={r_fpr}, thresh_fpr={thresh_fpr}")

    acc, p, r, fpr, acc_fpr, p_fpr, r_fpr, thresh_fpr = vd.get_metric_clean(thresh, fpr_rate)
    print(f"clean: acc={acc}, p={p}, r={r}, fpr={fpr}, acc_fpr={acc_fpr}, p_fpr={p_fpr}, r_fpr={r_fpr}, thresh_fpr={thresh_fpr}")


if __name__ == '__main__':
    # logger.info("hello, insightface/recognition")
    main()
