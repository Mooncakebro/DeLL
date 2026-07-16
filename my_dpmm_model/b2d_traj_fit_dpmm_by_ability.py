'''
run this code under ~/dpmm_model/model (after properly setting b2d_train).
'''

# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import sys
import json
import os
import carla # 如果不直接使用CARLA的Transform，可以考虑用numpy/scipy实现变换
import torch.utils.data
import numpy as np
import torchvision.transforms
from PIL import Image, ImageDraw # ImageDraw 用于绘图
from loguru import logger
import math
import pickle
from torch.utils.data import Dataset, DataLoader
import yaml
import random

# 获取当前脚本的上两级目录（LEGION/my_code）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # parent_dir=my_code
sys.path.append(parent_dir)
# from dataset.b2d_1000_dataset import ScenarioDataset
# from dataset.b2d_ability_dataset import AbilityDataset
from team_code.ability_data import Ability_CARLA_Data
from my_dpmm_model import BNPModel

from utils import weighted_kl_divergence, collect_samples_for_tsne, collect_samples_for_tsne_v2, visualize_tsne, \
plot_losses, reset_optimizer, convert_tensor_to_list, purge_invalid_values, cluster_and_evaluate, collect_samples_for_cluster_eval, print_data_info, \
combine_skill_dataloaders
import torch.optim as optim
from team_code.config import GlobalConfig
from plot_cluster_traj import visualize_all_waypoints

from datetime import datetime


def train_dpmm(dpmm, config, skill_dataloaders):
    """dpmm学习主训练循环"""
    print(f"Starting dpmm learning at {datetime.now()}")

    # used_dataloaders = []
    skill_id = 0

    for skill, dataloaders in skill_dataloaders.items():
        # print(f"\n{'*' * 50}")
        print(f"\n{'*' * 15} Training Skill {skill} {'*' * 15} ")
        # print(f"{'*' * 50}")

        # 遍历所有任务
        for task_id, dataloader in enumerate(dataloaders):
            # print(f"\n{'=' * 50}")
            # print(f"Training Task {task_id+1}/{len(dataloaders)}: {dataloader.dataset.scen_name}")
            # print(f"{'=' * 50}")

            # update dpmm every how many batches
            dpmm_update_freq = len(dataloader)//config['dpmm_update_per_epoch']

            # 2. 训练当前任务
            for epoch in range(config["epochs_per_task"]):
                print(f"\nEpoch {epoch + 1}/{config['epochs_per_task']}")
                current_f_traj_list = []
                # 3. 遍历批次数据
                for batch_idx, batch in enumerate(dataloader):
                    batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                    # print(f'shape of flatten traj: {batch["flatten_trajectory_points"].detach().shape}')
                    batch_size = batch['route'].size(0)
                    print(f'route shape: {batch["route"].shape}')
                    flatten_route = batch['route'].detach().reshape(batch_size, -1)
                    print(f'flatten_route shape: {flatten_route.shape}')
                    current_f_traj_list.append(flatten_route)
                    if (batch_idx + 1) % dpmm_update_freq == 0:
                        print(f"Updating DPMM at iteration {batch_idx}...")
                        if task_id > 0 or skill_id > 0:
                            K = len(dpmm.components)
                            new_task_data_ratio = 1 / (1+K) 
                            # num_to_sample = int((1 - config['new_task_data_ratio']) * len(current_f_traj_list) / config['new_task_data_ratio'])
                            num_to_sample = int((1 - new_task_data_ratio) * len(current_f_traj_list) / new_task_data_ratio)
                            print(f'num_to_sample: {num_to_sample }')
                            # 合并历史z
                            z_samples = torch.cat((dpmm.sample_all(num_samples=num_to_sample), torch.cat(current_f_traj_list, dim=0)), dim=0)
                        else:
                            z_samples = torch.cat(current_f_traj_list, dim=0)
                        # print(f'z_samples shape: {z_samples.shape}')
                        z_samples = purge_invalid_values(z_samples, "z_samples")
                        dpmm.fit(z_samples)
                        current_f_traj_list = []
                        
                        #track dpmm clusters
                        tracked_clusters = sorted([{'cluster_id': data['id'],'mu': data['mu'],'var': data['var']} for data in dpmm.current_clusters.values()], key=lambda x: x['cluster_id'])
                        tracked_clusters = convert_tensor_to_list(tracked_clusters)
                        tracked_clusters_path = os.path.join('results/'+exp_time+'/track_cluster_log', str(skill_id)+'-'+str(task_id)+'-'+str(epoch)+'-'+str(batch_idx)+"-tracked_clusters.json")
                        with open(tracked_clusters_path, 'w') as f:
                            json.dump(tracked_clusters, f, indent=4)
                        print(f"Saved tracked_cluster to {tracked_clusters_path}")
                        components = sorted(dpmm.components, key = lambda x: x['k'])
                        components_path = os.path.join('results/'+exp_time+'/component_log', str(skill_id)+'-'+str(task_id)+'-'+str(epoch)+'-'+str(batch_idx)+"-components.json")
                        with open(components_path, 'w') as f:
                            json.dump(components, f, indent=4)
                        print(f"Saved component to {components_path}")

            # 12. 任务完成后可视化
            # used_dataloaders.append(dataloader)
            # print(f"Generating t-SNE visualization for task {task_id}...")
            # tsne_z_samples, tsne_scen_labels = collect_samples_for_tsne(model, used_dataloaders, device)
            # tsne_z_samples, tsne_scen_labels = collect_samples_for_tsne_v2(used_dataloaders, device)
            # visualize_tsne(tsne_z_samples, tsne_scen_labels, used_dataloaders, 'results/'+exp_time+'/cluster_fig')

            # eval cluster 
            # X, _ = collect_samples_for_cluster_eval(used_dataloaders, num_per_dataloader=100)
            # _ = cluster_and_evaluate(X, dpmm, os.path.join('./results',exp_time,'eval_cluster',f'{task_id}th_clustering_results.json'))
        # dpmm.save_model(os.path.join(dpmm_save_dir, str(skill_id)))
        skill_id = skill_id +1

    print(f"dpmm learning completed at {datetime.now()}")
    dpmm.save_model(dpmm_save_dir)
    visualize_all_waypoints(date_dir=os.path.join('./results',exp_time))


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    exp_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    print(f'exp start at {exp_time}')

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Current Working Directory:", os.getcwd())
    dpmm_save_dir = os.path.join(script_dir, 'results', exp_time, 'dpmm_model')

    # 获取当前脚本所在目录
    # project_root = "../../"
    # scenario_dirs = [os.path.join(project_root, "b2d_1000", d) for d in
    #                  os.listdir(os.path.join(project_root, "b2d_1000"))]
    # data_root = "../../b2d_1000_train"
    # data_root = "../../b2d_143_train"
    data_root = "../../b2d_mini_v2"

    # for d in scenario_dirs:
    #     print(d)
    #     # 加载scen_skill_desc_list（需替换为实际路径）
    # with open('../text_enco/scen_skill_desc_list.pkl', 'rb') as f:
    #     scen_skill_desc_list = pickle.load(f)
        # print(scen_skill_desc_list)

    config = GlobalConfig()

    rank = int(os.environ['RANK'])  # Rank across all processes
    if config.local_rank == -999:  # For backwards compatibility
        local_rank = int(os.environ['LOCAL_RANK'])  # Rank on Node
    else:
        local_rank = int(config.local_rank)
        world_size = int(os.environ['WORLD_SIZE'])  # Number of processes
    print(f'RANK, LOCAL_RANK and WORLD_SIZE in environ: {rank}/{local_rank}/{world_size}')

    device = torch.device(f'cuda:{local_rank}')

    # assign scenario dataloaders to skill(ability) datalaoders and check
    # skill_dataloaders = {'EmergencyBrake':[], 'TrafficSign':[], 'Merging':[], 'Overtaking':[], 'GiveWay':[]}
    skill_dataloaders = {'Give_Way':[], 'Overtaking':[], 'Merging':[], 'Traffic_Sign':[], 'Emergency_Brake':[], 'No_Scenario': []}

    for k in skill_dataloaders.keys():
        ability = k
        print(ability)

        dataset = Ability_CARLA_Data(root=config.data_roots,
                         config=config,
                         estimate_class_distributions=config.estimate_class_distributions,
                         estimate_sem_distribution=config.estimate_semantic_distribution,
                         shared_dict=None,
                         rank=rank,
                         validation=False,
                         ability=ability)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )

        skill_dataloaders[k].append(dataloader)

    print(skill_dataloaders)

    # 初始化参数配置
    dpmm_config = {
        # "dpmm_update_freq": 10,  # DPMM更新间隔
        # "dpmm_update_every_epoch": True,  # DPMM更新间隔
        "dpmm_update_per_epoch": 1,  # DPMM更新间隔
        "epochs_per_task": 1,  # 每个任务训练轮数 5 may be enough
        # "batch_size": 16,  # 批次大小
        # "learning_rate": 1e-4,  # 学习率
        "latent_dim": 2*10,  # 潜在空间维度
        "save_dir": "results",  # 保存路径
        "gamma0": 6,  # DPMM初始参数
        "num_lap": 1000,
        "sF": 1e-5,
        # "new_task_data_ratio": 0.5,
        "w_kl_beta": 1,
        "hist_frame_nums":5,
        "future_frame_nums":5,
    }
    # 创建保存目录（如果不存在）
    os.makedirs('results/'+exp_time, exist_ok=True)
    os.makedirs('results/'+exp_time+'/component_log', exist_ok=True)
    os.makedirs('results/'+exp_time+'/track_cluster_log', exist_ok=True)
    os.makedirs('results/'+exp_time+'/ckpt', exist_ok=True)
    # os.makedirs(os.path.join('./results',exp_time,'eval_clustering'), exist_ok=True)
    # 保存 config 到 JSON 文件
    dpmm_config_path = os.path.join('results/'+exp_time, "config.json")
    with open(dpmm_config_path, 'w') as f:
        json.dump(dpmm_config, f, indent=4)
    print(f"Saved config to {dpmm_config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpmm = BNPModel(save_dir=dpmm_save_dir, gamma0=dpmm_config["gamma0"], num_lap=dpmm_config["num_lap"], sF=dpmm_config["sF"])  # DPMM模型

    train_dpmm(dpmm, dpmm_config, skill_dataloaders)