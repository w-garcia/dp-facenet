import tensorflow as tf
import argparse 
import matplotlib.pyplot as plt
import os
import numpy as np

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

from attack import _cal_metric
from util import pickle_load

GOLDEN_RATIO = 1.618
onecolumn_width = 3.335
onecolumn_height = onecolumn_width / GOLDEN_RATIO

twocolumn_width = onecolumn_width * 2
twocolumn_height = twocolumn_width / GOLDEN_RATIO
twocolumn_height_half = twocolumn_height / 2

plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', labelsize=8)

MAX_X = 150


def plot_acc(tr_x, val_x, tag):
    if len(val_x) == 0:
        raise ValueError("Supplied an empty accuracies record.")

    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.13, bottom=0.19, right=0.98, top=0.90)

    # axes.plot(range(len(tr_x)), tr_x, label='train', color='red')
    axes.plot(range(len(val_x)), val_x, label='val', color='blue')
    axes.set_xlabel("train iteration")
    axes.set_ylabel("accuracy")
    axes.set_ylim(bottom=0.00, top=1.00)
    
    f.legend()
    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/acc_{tag}.pdf')


def process_DP_eps_acc(tr_file, val_file, tag):
    
    def eps_acc_from_log(log_file):
        l_eps = []
        l_acc = []
        
        for summary in summary_iterator(log_file):
            for v in summary.summary.value:
                t = tensor_util.MakeNdarray(v.tensor)
                # print(v.tag, summary.step, float(t), type(t))

                if v.tag == 'eps':
                    l_eps.append(float(t))
                elif v.tag == 'acc':
                    l_acc.append(float(t))

        return l_eps, l_acc
    
    _, tr_l_acc = eps_acc_from_log(tr_file)
    val_l_eps, val_l_acc = eps_acc_from_log(val_file)
    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.12, bottom=0.19, right=0.95, top=0.90)

    axes.plot(range(len(val_l_eps)), val_l_eps, label='val', color='red')
    axes.set_xlabel("train iteration")
    axes.set_ylabel("epsilon")

    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/eps_{tag}.pdf')

    plot_acc(tr_l_acc[:MAX_X], val_l_acc[:MAX_X], tag)


def process_acc(tr_file, val_file, tag):
    def acc_from_log(log_file):
        l_acc = []
        for summary in summary_iterator(log_file):
            for v in summary.summary.value:
                t = tensor_util.MakeNdarray(v.tensor)
                # print(v.tag, summary.step, float(t), type(t))

                if v.tag == 'acc':
                    l_acc.append(float(t))

        return l_acc
    
    tr_l_acc = acc_from_log(tr_file)
    val_l_acc = acc_from_log(val_file)

    plot_acc(tr_l_acc[:MAX_X], val_l_acc[:MAX_X], tag)


def draw_curve(sim, label, tag):
    P = []
    R = []
    TPR = []
    FPR = []
    # sim, label = self._get_sim_label()
    for thresh in np.linspace(-1, 1, 100):
        acc, p, r, fpr = _cal_metric(sim, label, thresh)
        P.append(p)
        R.append(r)
        TPR.append(r)
        FPR.append(fpr)

    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.13, bottom=0.19, right=0.98, top=0.90)
    axes.axis([0, 1, 0, 1])
    axes.set_xlabel("R")
    axes.set_ylabel("P")
    axes.plot(R, P, color="r", linestyle="--", marker="*", linewidth=1.0)
    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/RP_{tag}.pdf')

    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.13, bottom=0.19, right=0.98, top=0.90)
    axes.axis([0, 1, 0, 1])
    axes.set_xlabel("FPR")
    axes.set_ylabel("TPR")
    axes.plot(FPR, TPR, color="r", linestyle="--", marker="*", linewidth=1.0) 
    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/FPR_TPR_{tag}.pdf')


if __name__ == '__main__':
    # opt = argparse.ArgumentParser()
    # opt.add_argument('log_file', help="Path to tfevents file")
    # opt = vars(opt.parse_args())

    DP_arcface_log_val_path = "insightface/recognition/logs/DP_arcface_delta-0.001_lr-0.005_emb-512/20191209-212154/valid/events.out.tfevents.1575944514.c42a-s29.ufhpc.152218.143.v2"
    DP_arcface_log_tr_path = "insightface/recognition/logs/DP_arcface_delta-0.001_lr-0.005_emb-512/20191209-212154/train/events.out.tfevents.1575944514.c42a-s29.ufhpc.152218.135.v2"
    arcface_log_val_path = "insightface/recognition/logs/arcface_lr-0.005_emb-512/20191209-195158/valid/events.out.tfevents.1575939118.c42a-s29.ufhpc.128458.141.v2"
    arcface_log_tr_path = "insightface/recognition/logs/arcface_lr-0.005_emb-512/20191209-195158/train/events.out.tfevents.1575939118.c42a-s29.ufhpc.128458.133.v2"

    if not os.path.exists('figs'):
        os.makedirs('figs')

    process_DP_eps_acc(DP_arcface_log_tr_path, DP_arcface_log_val_path, "DP_arcface")
    # process_DP_eps_acc(DP_triplet_log_path, "DP_triplet")
    process_acc(arcface_log_tr_path, arcface_log_val_path, "arcface")
    # process_acc(triplet_log_path, "triplet")

    for tag in os.listdir('save'):
        adv_sim, adv_label, queries = pickle_load(os.path.join('save', tag, 'adv_simlabelqueries.pkl'))
        clean_sim, clean_label = pickle_load(os.path.join('save', tag, 'clean_simlabel.pkl'))
        
        draw_curve(adv_sim, adv_label, f"adv_{tag}")
        draw_curve(clean_sim, clean_label, f"clean_{tag}")
        
        with open(f'figs/hsja_queries_{tag}.txt', 'w') as f:
            mean, std = np.mean(queries), np.std(queries)
            f.write(f'{mean} +- {std}\n')

