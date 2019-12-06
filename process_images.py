import os
import argparse
import numpy as np
import shutil
from tqdm import tqdm

SEED = 9


def process(opt):
    if not os.path.exists(opt['root_dir']):
        raise Exception("Supplied source directory doesn't exist")
    
    tr_frac, val_frac, tst_frac = [float(s) for s in opt['split']]
    base_dir = opt['root_dir'].split('/')[-1]
    dst_base_dir = f"{base_dir}_split"
    dst_base_dir = opt['root_dir'].replace(base_dir, dst_base_dir)
    print(f"train_split={tr_frac},\tval_split={val_frac},\ttest_split={tst_frac}")

    for node in tqdm(os.listdir(opt['root_dir'])):
        node_path = os.path.join(opt['root_dir'], node) 
        if os.path.isdir(node_path):
            imgs = np.random.permutation(os.listdir(node_path))
            n = len(imgs)
            num_train = int(tr_frac * n)
            num_val = int(val_frac * n)
            
            train_imgs = imgs[:num_train]
            val_imgs = imgs[num_train:num_train + num_val]
            tst_imgs = imgs[num_train + num_val:]
            assert len(set(train_imgs).intersection(set(val_imgs))) == 0
            assert len(set(val_imgs).intersection(set(tst_imgs))) == 0

            for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, tst_imgs]):
                new_node_path = os.path.join(dst_base_dir, split, node)
                if not os.path.exists(new_node_path):
                    os.makedirs(new_node_path)
                
                for img in split_imgs:
                    from_path = os.path.join(node_path, img)
                    dst_path = os.path.join(new_node_path, img)
                    shutil.copy2(from_path, dst_path)
            

if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--root_dir', required=True)
    opt.add_argument('--split', nargs='+', default=[0.5, 0.25, 0.25]) 
    opt = opt.parse_args()
    process(vars(opt))

