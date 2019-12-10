import os
import random
import string
import numpy as np
import _pickle as pickle

from custom_logging import get_logger
from scipy import misc
from datetime import datetime

util_logger = get_logger(__name__)


def get_timer_suffix(date_object):
    return date_object.strftime('%m%d%y-%H%M%S')


def get_random_upper_string(n):
    # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(n))


def save_my_img(my_dir, my_name, img):
    t_path = os.path.join(my_dir, '{}_{}.png'.format(my_name, get_random_upper_string(8)))
    misc.imsave(t_path, img)

    return t_path


def get_time_stamp():
    date_object = datetime.now()
    return date_object.strftime('%m%d%y-%H%M%S')


def concept_to_onehot(concept_list, val):
    res = np.zeros(len(concept_list))
    idx = list(concept_list).index(val)
    res[idx] = 1
    return res


def pickle_write(fpath, obj):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)

    return obj


def log_dict(logger, d):
    for key in d:
        logger.info("\t\t{}: {}".format(key, d[key]))


def is_valid_sample(mu, std, o_sig, p_sig, noise_sd):
    eps_norm = np.linalg.norm(np.random.normal(loc=mu, scale=std, size=o_sig.shape), 2) * noise_sd
    if np.linalg.norm(p_sig - o_sig, 2) < eps_norm:
        return True
    else:
        return False
