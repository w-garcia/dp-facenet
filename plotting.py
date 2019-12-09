import tensorflow as tf
import argparse 
import matplotlib.pyplot as plt
import os
import numpy as np

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

GOLDEN_RATIO = 1.618
onecolumn_width = 3.335
onecolumn_height = onecolumn_width / GOLDEN_RATIO

twocolumn_width = onecolumn_width * 2
twocolumn_height = twocolumn_width / GOLDEN_RATIO
twocolumn_height_half = twocolumn_height / 2

plt.rc('xtick', labelsize=6)
plt.rc('ytick', labelsize=6)
plt.rc('axes', labelsize=8)


def plot_acc(x, tag):
    if len(x) == 0:
        raise ValueError("Supplied an empty accuracies record.")

    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.13, bottom=0.19, right=0.98, top=0.90)

    axes.plot(range(len(x)), x, label='acc', color='blue')
    axes.set_xlabel("train iteration")
    axes.set_ylabel("accuracy")
    axes.set_ylim(bottom=0.00, top=1.00)
            
    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/acc_{tag}.pdf')


def process_DP_eps_acc(log_file, tag):
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

    f, axes = plt.subplots(1, 1)
    f.subplots_adjust(left=0.12, bottom=0.19, right=0.95, top=0.90)

    axes.plot(range(len(l_eps)), l_eps, label='eps', color='red')
    axes.set_xlabel("train iteration")
    axes.set_ylabel("epsilon")

    f.set_size_inches(onecolumn_width, onecolumn_height)
    f.savefig(f'figs/eps_{tag}.pdf')

    plot_acc(l_acc[:100], tag)

def process_acc(log_file, tag):
    l_acc = []

    for summary in summary_iterator(log_file):
        for v in summary.summary.value:
            t = tensor_util.MakeNdarray(v.tensor)
            print(v.tag, summary.step, float(t), type(t))

            if v.tag == 'acc':
                l_acc.append(float(t))
    
    plot_acc(l_acc[:100], "arcface")


if __name__ == '__main__':
    # opt = argparse.ArgumentParser()
    # opt.add_argument('log_file', help="Path to tfevents file")
    # opt = vars(opt.parse_args())

    DP_arcface_log_path = "insightface/recognition/logs/DP_delta-0.001_lr-0.01_emb-512/20191206-185440/valid/events.out.tfevents.1575676480.c42a-s21.ufhpc.59947.154.v2"
    DP_triplet_log_path = "insightface/recognition/logs/DP_triplet_delta-0.001_lr-0.01_emb-512/20191206-182005/train/events.out.tfevents.1575674405.c42a-s25.ufhpc.59419.135.v2"
    arcface_log_path = "insightface/recognition/logs/lr-0.025_emb-512/20191204-120322/valid/events.out.tfevents.1575479002.c42a-s17.ufhpc.228394.92.v2"
    triplet_log_path = ""

    if not os.path.exists('figs'):
        os.makedirs('figs')

    process_DP_eps_acc(DP_arcface_log_path, "DP_arcface")
    process_DP_eps_acc(DP_triplet_log_path, "DP_triplet")
    process_acc(arcface_log_path, "arcface")
    process_acc(triplet_log_path, "triplet")

