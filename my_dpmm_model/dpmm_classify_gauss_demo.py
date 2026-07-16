import os
import bnpy
import numpy as np
import torch
from matplotlib import pylab
import seaborn as sns
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir,'results', 'dpmm_classify_gauss')
os.makedirs(save_dir, exist_ok=True)

from gen_gaussian_data import *
from my_dpmm_model import BNPModel

dataset = generate_gaussian_clusters()
plot_clusters(dataset)


old_and_current_data = np.empty((0,2))


datasetv2 = generate_gaussian_clusters()
datasetv2[1]['obs'] = np.vstack((datasetv2[1]['obs'],datasetv2[3]['obs']))
datasetv2[2]['obs'] = np.vstack((datasetv2[2]['obs'],datasetv2[4]['obs']))
datasetv2.pop(-1)
datasetv2.pop(-1)

new_dataset_ratio = 0.5  # when fit a dpmm for the second time, old data is sampled from current dpmm clusters, then it is combined  with new data to fit dpmm.

#####################
# TRAIN: Train dpmm model
#####################
dpmm = BNPModel(save_dir, gamma0=4, num_lap=100, sF=0.1)
task_id = 0

for partial_dataset in datasetv2:  # fit data one by one, mimic lifelong learning.
    task_id = task_id + 1

    print(f'dataset shape is {partial_dataset["obs"].shape}')
    # old_and_current_data = np.vstack((old_and_current_data, partial_dataset['obs']))
    # dpmm.fit(torch.from_numpy(partial_dataset['obs']))
    if task_id > 1:
        num_to_sample = int((1-new_dataset_ratio)*partial_dataset['obs'].shape[0]/new_dataset_ratio)
        print(f'num to sample = {num_to_sample}')
        old_and_current_data = np.vstack((dpmm.sample_all(num_samples=num_to_sample).cpu().numpy(), partial_dataset['obs']))
    else:
        old_and_current_data = partial_dataset['obs']
    dpmm.fit(torch.from_numpy(old_and_current_data))
    print(f'current task id is {partial_dataset["id"]}, total {task_id} tasks have been fitted.')
    dpmm.show_clusters_over_time(data=old_and_current_data)

print('dpmm components:')
print(dpmm.components)

dpmm.save_model(save_dir)

#####################
# EVAL: input some 2d points and dpmm give them assigments (to the related cluster).
#####################
eval_save_dir = os.path.join(save_dir,'eval')

eval_dpmm = BNPModel(eval_save_dir)
load_path = save_dir
eval_dpmm.load_model(load_path)
# print(eval_dpmm.model)        # bnpy HModel object
# print(eval_dpmm.info_dict)    # Training metadata
print('eval_dpmm components:')
print(eval_dpmm.components)

resp, y_pred = eval_dpmm.cluster_assignments(torch.from_numpy(dataset[1]['obs']))  # resp has shape of (num_sample, componet) and y_pred has shape of(num_sample,)
print(f'y_pred: {y_pred}')
print(f'types: {type(resp)} and {type(y_pred)}')
print(f'shapes: {resp.shape} and {y_pred.shape}')
# print(resp)
# plt.scatter(dataset[1]['obs'][:,0],dataset[1]['obs'][:,1])
# plt.show()


