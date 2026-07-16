'''
Online alternating Network-DPMM training script.

This script extends ``my_train_ability_wTFFdeQtd.py`` to co-train the
TransFuser++ network with two DPMM knowledge spaces (trajectory anchors and
fused-feature anchors) inside a single training loop.  When a DPMM's ring
buffer fills, the DPMM is fitted (on rank 0), new anchors are broadcast to
all ranks, and the model's decoders/encoders are hot-swapped — all without
interrupting the training loop.

Usage (2-GPU example)::

    CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 \
    torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 \
        --rdzv_id=1234576890 --rdzv_backend=c10d \
        core_team_code/online_dpmm/my_train_ability_wTFFdeQtd_online.py \
        --logdir /path/to/logdir --root_dir /path/to/dataset_root/ \
        --id exp_online_000 --cpu_cores 8 --online_dpmm 1

Key differences from the original script:
    - ``--online_dpmm 1`` enables the online alternating DPMM mode.
    - Two ring buffers (traj + fuseFeat) are filled every batch.
    - When a buffer fills, the corresponding DPMM is fitted and anchors are
      hot-swapped into the model.
    - No separate offline DPMM fitting scripts are needed.
'''

import argparse
import json
import os
import pathlib
import datetime
import random
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
from diskcache import Cache
import torchmetrics

# --- Path setup: core_team_code/ must come BEFORE team_code/ so that our
# modified modules (my_model_wTFFdeQtd, traj_front_door_encoder, etc.) are
# found first.  Reverse iteration because insert(0, ...) pushes earlier
# entries down.
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_TEAM_CODE = os.path.join(_THIS_DIR, '..')
_TEAM_CODE = os.path.join(_CORE_TEAM_CODE, '..', 'team_code')
for _p in [_TEAM_CODE, _CORE_TEAM_CODE, _THIS_DIR]:  # reversed: last wins at pos 0
    if _p not in sys.path:
        sys.path.insert(0, _p)

# config and core model/data imports now resolve to core_team_code/ first
from config import GlobalConfig
from my_model_wTFFdeQtd import LidarCenterNet
from ability_data import Ability_CARLA_Data
from plant import PlanT
from online_dpmm_manager import OnlineDPMMManager

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)

# On some systems it is necessary to increase the limit on open file descriptors.
try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
except (ModuleNotFoundError, ImportError) as e:
    print(e)


def load_checkpoint_ignore_anchors(model, checkpoint_path, device, strict=False):
    """Load checkpoint, skipping anchor buffers (they'll be set by DPMM).

    Handles DDP-wrapped models by unwrapping to the inner module.
    Keys ending with ``.anchors`` are silently skipped (unlike strict=False
    which would still warn about missing keys).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_state_dict = (checkpoint if not isinstance(checkpoint, dict)
                            else checkpoint.get('model_state_dict', checkpoint))

    # Get inner module if DDP-wrapped
    model_module = model.module if hasattr(model, 'module') else model
    model_state_dict = model_module.state_dict()

    filtered_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.endswith('.anchors') and 'anchors' in key:
            if key in model_state_dict:
                print(f"跳过anchor数据: {key} | ckpt={value.shape}"
                      f" | model={model_state_dict[key].shape}")
            else:
                print(f"跳过anchor数据: {key} | ckpt={value.shape}"
                      f" | (not in current model)")
            continue
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                print(f"跳过不匹配的参数: {key} | ckpt={value.shape}"
                      f" | model={model_state_dict[key].shape}")
        else:
            print(f"跳过不存在的参数: {key}")

    model_module.load_state_dict(filtered_state_dict, strict=False)
    return model


@record
def main():
    torch.cuda.empty_cache()

    config = GlobalConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default=config.id, help='Unique experiment identifier.')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=config.lr, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                        help='Batch size for one GPU.')
    parser.add_argument('--logdir', type=str, required=True, help='Directory to log data and models to.')
    parser.add_argument('--load_file', type=str, default=config.load_file,
                        help='Model to load for initialization.')
    parser.add_argument('--setting', type=str, default=config.setting,
                        help='What training setting to use.')
    parser.add_argument('--root_dir', type=str, required=True, nargs='+',
                        help='Root directory of your training data')
    parser.add_argument('--schedule_reduce_epoch_01', type=int, default=config.schedule_reduce_epoch_01)
    parser.add_argument('--schedule_reduce_epoch_02', type=int, default=config.schedule_reduce_epoch_02)
    parser.add_argument('--backbone', type=str, default=config.backbone)
    parser.add_argument('--image_architecture', type=str, default=config.image_architecture)
    parser.add_argument('--lidar_architecture', type=str, default=config.lidar_architecture)
    parser.add_argument('--use_velocity', type=int, default=config.use_velocity)
    parser.add_argument('--n_layer', type=int, default=config.n_layer)
    parser.add_argument('--val_every', type=int, default=config.val_every)
    parser.add_argument('--sync_batch_norm', type=int, default=config.sync_batch_norm)
    parser.add_argument('--zero_redundancy_optimizer', type=int, default=config.zero_redundancy_optimizer)
    parser.add_argument('--use_disk_cache', type=int, default=config.use_disk_cache)
    parser.add_argument('--lidar_seq_len', type=int, default=config.lidar_seq_len)
    parser.add_argument('--realign_lidar', type=int, default=int(config.realign_lidar))
    parser.add_argument('--use_ground_plane', type=int, default=int(config.use_ground_plane))
    parser.add_argument('--use_controller_input_prediction', type=int,
                        default=int(config.use_controller_input_prediction))
    parser.add_argument('--use_wp_gru', type=int, default=int(config.use_wp_gru))
    parser.add_argument('--pred_len', type=int, default=config.pred_len)
    parser.add_argument('--estimate_class_distributions', type=int,
                        default=int(config.estimate_class_distributions))
    parser.add_argument('--use_focal_loss', type=int, default=int(config.use_focal_loss))
    parser.add_argument('--use_cosine_schedule', type=int, default=int(config.use_cosine_schedule))
    parser.add_argument('--augment', type=int, default=int(config.augment))
    parser.add_argument('--use_plant', type=int, default=int(config.use_plant))
    parser.add_argument('--learn_origin', type=int, default=int(config.learn_origin))
    parser.add_argument('--local_rank', type=int, default=int(config.local_rank))
    parser.add_argument('--train_sampling_rate', type=int, default=int(config.train_sampling_rate))
    parser.add_argument('--use_amp', type=int, default=int(config.use_amp))
    parser.add_argument('--use_grad_clip', type=int, default=int(config.use_grad_clip))
    parser.add_argument('--use_color_aug', type=int, default=int(config.use_color_aug))
    parser.add_argument('--use_semantic', type=int, default=int(config.use_semantic))
    parser.add_argument('--use_depth', type=int, default=int(config.use_depth))
    parser.add_argument('--detect_boxes', type=int, default=int(config.detect_boxes))
    parser.add_argument('--use_bev_semantic', type=int, default=int(config.use_bev_semantic))
    parser.add_argument('--estimate_semantic_distribution', type=int,
                        default=int(config.estimate_semantic_distribution))
    parser.add_argument('--use_discrete_command', type=int, default=int(config.use_discrete_command))
    parser.add_argument('--gru_hidden_size', type=int, default=config.gru_hidden_size)
    parser.add_argument('--use_cutout', type=int, default=int(config.use_cutout))
    parser.add_argument('--add_features', type=int, default=int(config.add_features))
    parser.add_argument('--freeze_backbone', type=int, default=int(config.freeze_backbone))
    parser.add_argument('--learn_multi_task_weights', type=int,
                        default=int(config.learn_multi_task_weights))
    parser.add_argument('--transformer_decoder_join', type=int,
                        default=int(config.transformer_decoder_join))
    parser.add_argument('--bev_down_sample_factor', type=int, default=config.bev_down_sample_factor)
    parser.add_argument('--perspective_downsample_factor', type=int,
                        default=int(config.perspective_downsample_factor))
    parser.add_argument('--gru_input_size', type=int, default=int(config.gru_input_size))
    parser.add_argument('--num_repetitions', type=int, default=config.num_repetitions)
    parser.add_argument('--bev_grid_height_downsample_factor', type=int,
                        default=int(config.bev_grid_height_downsample_factor))
    parser.add_argument('--wp_dilation', type=int, default=int(config.wp_dilation))
    parser.add_argument('--use_tp', type=int, default=int(config.use_tp))
    parser.add_argument('--continue_epoch', type=int, default=int(config.continue_epoch))
    parser.add_argument('--max_height_lidar', type=float, default=float(config.max_height_lidar))
    parser.add_argument('--smooth_route', type=int, default=int(config.smooth_route))
    parser.add_argument('--use_speed_weights', type=int, default=int(config.use_speed_weights))
    parser.add_argument('--max_num_bbs', type=int, default=int(config.max_num_bbs))
    parser.add_argument('--use_optim_groups', type=int, default=int(config.use_optim_groups))
    parser.add_argument('--weight_decay', type=float, default=float(config.weight_decay))
    parser.add_argument('--use_label_smoothing', type=int, default=int(config.use_label_smoothing))
    parser.add_argument('--cpu_cores', type=int, required=True,
                        help='How many cpu cores are available on the machine.')
    parser.add_argument('--tp_attention', type=int, default=int(config.tp_attention))
    parser.add_argument('--multi_wp_output', type=int, default=int(config.multi_wp_output))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--crop_image', type=int, default=int(config.crop_image))
    parser.add_argument('--input_path_to_target_speed_network', type=int,
                        default=int(config.input_path_to_target_speed_network))
    parser.add_argument('--predict_checkpoint_len', type=int, default=config.predict_checkpoint_len)
    parser.add_argument('--max_x', type=int, default=int(config.max_x))
    parser.add_argument('--crop_bev_height_only_from_behind', type=int,
                        default=int(config.crop_bev_height_only_from_behind))
    parser.add_argument('--lidar_resolution_height', type=int, default=config.lidar_resolution_height)
    parser.add_argument('--dataset_cache_name', type=str, default='dataset_cache')
    parser.add_argument('--cosine_t0', type=int, default=int(config.cosine_t0))
    parser.add_argument('--compile', type=int, default=int(config.compile))
    parser.add_argument('--compile_mode', type=str, default=str(config.compile_mode))

    # --- Online DPMM arguments ---
    parser.add_argument('--online_dpmm', type=int, default=int(config.online_dpmm),
                        help='0=static anchors (old mode), 1=online alternating DPMM mode.')
    parser.add_argument('--traj_dpmm_buffer_size', type=int, default=config.traj_dpmm_buffer_size)
    parser.add_argument('--fusefeat_dpmm_buffer_size', type=int, default=config.fusefeat_dpmm_buffer_size)
    parser.add_argument('--dpmm_update_start_step', type=int, default=config.dpmm_update_start_step)
    parser.add_argument('--dpmm_update_freq_steps', type=int, default=config.dpmm_update_freq_steps)
    parser.add_argument('--traj_dpmm_replay_ratio', type=float, default=config.traj_dpmm_replay_ratio)
    parser.add_argument('--fusefeat_dpmm_replay_ratio', type=float,
                        default=config.fusefeat_dpmm_replay_ratio)
    parser.add_argument('--use_traj_front_door_encoder', type=int,
                        default=int(config.use_traj_front_door_encoder))
    parser.add_argument('--use_prior_fuseFeat', type=int, default=int(config.use_prior_fuseFeat))

    args = parser.parse_args()
    args.logdir = os.path.join(args.logdir, args.id)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if bool(args.use_disk_cache):
        tmp_folder = str(os.environ.get('SCRATCH', '/tmp'))
        tmp_folder = tmp_folder + '/' + args.dataset_cache_name
        print('Tmp folder for dataset cache: ', tmp_folder)
        shared_dict = Cache(directory=tmp_folder, size_limit=int(768 * 1024**3))
    else:
        shared_dict = None

    rank = int(os.environ['RANK'])
    if args.local_rank == -999:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = int(args.local_rank)
    world_size = int(os.environ['WORLD_SIZE'])
    print(f'RANK/LOCAL_RANK/WORLD_SIZE: {rank}/{local_rank}/{world_size}')

    device = torch.device(f'cuda:{local_rank}')

    torch.distributed.init_process_group(backend='nccl',
                                          init_method='env://',
                                          world_size=world_size,
                                          rank=rank,
                                          timeout=datetime.timedelta(minutes=15))

    ngpus_per_node = torch.cuda.device_count()
    ncpus_per_node = args.cpu_cores
    num_workers = int(ncpus_per_node / ngpus_per_node)
    torch.cuda.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    config.initialize(**vars(args))
    config.debug = int(os.environ.get('DEBUG_CHALLENGE', 0))

    # --- Loss weight setup (same as original) ---
    if config.use_plant:
        config.detailed_loss_weights['loss_semantic'] = 0.0
        config.detailed_loss_weights['loss_bev_semantic'] = 0.0
        config.detailed_loss_weights['loss_depth'] = 0.0
        config.detailed_loss_weights['loss_center_heatmap'] = 0.0
        config.detailed_loss_weights['loss_wh'] = 0.0
        config.detailed_loss_weights['loss_offset'] = 0.0
        config.detailed_loss_weights['loss_yaw_class'] = 0.0
        config.detailed_loss_weights['loss_yaw_res'] = 0.0
        config.detailed_loss_weights['loss_velocity'] = 0.0
        config.detailed_loss_weights['loss_brake'] = 0.0
    else:
        config.detailed_loss_weights['loss_forcast'] = 0.0

    if not config.use_controller_input_prediction:
        config.detailed_loss_weights['loss_target_speed'] = 0.0
        config.detailed_loss_weights['loss_checkpoint'] = 0.0
    if not config.use_wp_gru:
        config.detailed_loss_weights['loss_wp'] = 0.0
    if not config.use_semantic:
        config.detailed_loss_weights['loss_semantic'] = 0.0
    if not config.use_bev_semantic:
        config.detailed_loss_weights['loss_bev_semantic'] = 0.0
    if not config.use_depth:
        config.detailed_loss_weights['loss_depth'] = 0.0
    if not config.detect_boxes:
        config.detailed_loss_weights['loss_center_heatmap'] = 0.0
        config.detailed_loss_weights['loss_wh'] = 0.0
        config.detailed_loss_weights['loss_offset'] = 0.0
        config.detailed_loss_weights['loss_yaw_class'] = 0.0
        config.detailed_loss_weights['loss_yaw_res'] = 0.0
        config.detailed_loss_weights['loss_velocity'] = 0.0
        config.detailed_loss_weights['loss_brake'] = 0.0
    if config.lidar_seq_len == 1 and config.seq_len == 1:
        config.detailed_loss_weights['loss_velocity'] = 0.0
        config.detailed_loss_weights['loss_brake'] = 0.0
    if config.freeze_backbone:
        config.detailed_loss_weights['loss_semantic'] = 0.0
        config.detailed_loss_weights['loss_bev_semantic'] = 0.0
        config.detailed_loss_weights['loss_depth'] = 0.0
        config.detailed_loss_weights['loss_center_heatmap'] = 0.0
        config.detailed_loss_weights['loss_wh'] = 0.0
        config.detailed_loss_weights['loss_offset'] = 0.0
        config.detailed_loss_weights['loss_yaw_class'] = 0.0
        config.detailed_loss_weights['loss_yaw_res'] = 0.0
        config.detailed_loss_weights['loss_velocity'] = 0.0
        config.detailed_loss_weights['loss_brake'] = 0.0
    if config.multi_wp_output:
        config.detailed_loss_weights['loss_selection'] = 1.0

    if args.learn_multi_task_weights:
        for k in config.detailed_loss_weights:
            if config.detailed_loss_weights[k] > 0.0:
                config.detailed_loss_weights[k] = torch.nn.Parameter(
                    torch.zeros(1, dtype=torch.float32, requires_grad=True))
            else:
                config.detailed_loss_weights[k] = None
        config.detailed_loss_weights = torch.nn.ParameterDict(config.detailed_loss_weights)
    else:
        factor = 1.0 / sum(config.detailed_loss_weights.values())
        for k in config.detailed_loss_weights:
            config.detailed_loss_weights[k] = config.detailed_loss_weights[k] * factor

    # --- Model (dataset is created inside the ability loop below) ---
    if config.use_plant:
        model = PlanT(config)
    else:
        model = LidarCenterNet(config)

    if args.learn_multi_task_weights:
        for k in config.detailed_loss_weights:
            if config.detailed_loss_weights[k] is not None:
                model.register_parameter(name='weight_' + k, param=config.detailed_loss_weights[k])
    model.cuda(device=device)

    start_epoch = 0
    if not args.load_file is None:
        print('=============load=================')
        load_name = str(pathlib.Path(args.load_file).stem)
        if args.continue_epoch:
            start_epoch = int(''.join(filter(str.isdigit, load_name))) + 1
        model = load_checkpoint_ignore_anchors(model, args.load_file, device)
        print("anchor加载成功")

    if config.freeze_backbone:
        model.backbone.requires_grad_(False)
        if config.detect_boxes:
            model.head.requires_grad_(False)
        if config.use_semantic:
            model.semantic_decoder.requires_grad_(False)
        if config.use_bev_semantic:
            model.bev_semantic_decoder.requires_grad_(False)
        if config.use_depth:
            model.depth_decoder.requires_grad_(False)

    if bool(args.sync_batch_norm):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    find_unused_parameters = False
    if config.use_plant:
        find_unused_parameters = True
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=None, output_device=None,
        broadcast_buffers=False, find_unused_parameters=find_unused_parameters)

    if config.use_optim_groups:
        params = model.module.create_optimizer_groups(config.weight_decay)
    else:
        params = model.parameters()

    if bool(args.zero_redundancy_optimizer):
        optimizer = ZeroRedundancyOptimizer(params, optimizer_class=optim.AdamW, lr=args.lr, amsgrad=True)
    else:
        optimizer = optim.AdamW(params, lr=args.lr, amsgrad=True)

    if not args.load_file is None and not config.freeze_backbone and args.continue_epoch:
        optimizer.load_state_dict(
            torch.load(args.load_file.replace('model_', 'optimizer_'), map_location=device))

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum(np.prod(p.size()) for p in model_parameters)
    if rank == 0:
        print('Total trainable parameters: ', num_params)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    # Dataloaders are created inside the ability loop below.

    # --- Logdir ---
    if ((not os.path.isdir(args.logdir)) and (rank == 0)):
        print('Created dir:', args.logdir, rank)
        os.makedirs(args.logdir, exist_ok=True)

    if rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        with open(os.path.join(args.logdir, 'args.txt'), 'w', encoding='utf-8') as f:
            json.dump(args.__dict__, f, indent=2)
        json_config = jsonpickle.encode(config)
        with open(os.path.join(args.logdir, 'config.json'), 'w') as f2:
            f2.write(json_config)
    else:
        writer = None

    # --- Scheduler / Scaler ---
    if config.use_cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.cosine_t0, T_mult=config.cosine_t_mult)
    else:
        milestones = [args.schedule_reduce_epoch_01, args.schedule_reduce_epoch_02]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=config.multi_step_lr_decay)
    scaler = torch.amp.GradScaler(device, enabled=bool(config.use_amp))
    if not args.load_file is None and not config.freeze_backbone and args.continue_epoch:
        scheduler.load_state_dict(torch.load(args.load_file.replace('model_', 'scheduler_'), map_location=device))
        scaler.load_state_dict(torch.load(args.load_file.replace('model_', 'scaler_'), map_location=device))

    # --- Online DPMM Manager ---
    online_dpmm_manager = None
    if config.online_dpmm:
        online_dpmm_manager = OnlineDPMMManager(
            config=config, logdir=args.logdir, device=device,
            rank=rank, world_size=world_size)
        if rank == 0:
            print(f'[OnlineDPMM] Enabled online alternating DPMM training.')
            print(f'  Traj buffer:    {config.traj_dpmm_buffer_size} samples, '
                  f'fit_min={config.traj_dpmm_fit_min_samples}')
            print(f'  FuseFeat buffer: {config.fusefeat_dpmm_buffer_size} samples, '
                  f'fit_min={config.fusefeat_dpmm_fit_min_samples}')
            print(f'  Update start step: {config.dpmm_update_start_step}, '
                  f'freq: {config.dpmm_update_freq_steps}')

    # --- Engine with sequential ability training ---
    ability_list = getattr(config, 'selected_ability_list',
                           [config.selected_ability])
    epochs_per_ability = args.epochs  # --epochs now means epochs per ability
    if rank == 0:
        print(f'Sequential ability training: {ability_list}')
        print(f'Epochs per ability: {epochs_per_ability} '
              f'(total={epochs_per_ability * len(ability_list)})')

    trainer = OnlineEngine(
        model=model, optimizer=optimizer,
        dataloader_train=None, dataloader_val=None,
        args=args, config=config, writer=writer, device=device,
        rank=rank, world_size=world_size, cur_epoch=start_epoch,
        scheduler=scheduler, scaler=scaler,
        online_dpmm_manager=online_dpmm_manager)

    for ability_idx, ability_name in enumerate(ability_list):
        if rank == 0:
            print(f'\n{"="*60}')
            print(f'Ability {ability_idx+1}/{len(ability_list)}: {ability_name}')
            print(f'{"="*60}')

        # --- Per-ability log subdirectory ---
        ability_logdir = os.path.join(args.logdir, ability_name)
        trainer.ability_logdir = ability_logdir
        trainer.cur_epoch = 0  # reset epoch counter per ability
        if rank == 0:
            os.makedirs(ability_logdir, exist_ok=True)

        # --- Load latest checkpoint from previous ability ---
        if ability_idx > 0:
            prev_ability = ability_list[ability_idx - 1]
            prev_latest = os.path.join(args.logdir, prev_ability, 'latest.pth')
            if os.path.isfile(prev_latest):
                if rank == 0:
                    print(f'Loading checkpoint from previous ability: {prev_latest}')
                model = load_checkpoint_ignore_anchors(model, prev_latest, device)
            else:
                if rank == 0:
                    print(f'Warning: no latest.pth found from {prev_ability}, continuing...')

        # --- Re-create dataset for this ability ---
        config.selected_ability = ability_name
        train_set = Ability_CARLA_Data(
            root=config.mini_dataset_root,
            config=config,
            estimate_class_distributions=config.estimate_class_distributions,
            estimate_sem_distribution=config.estimate_semantic_distribution,
            shared_dict=shared_dict,
            rank=rank,
            validation=False,
            ability=ability_name,
        )

        if args.setting != 'all':
            val_set = Ability_CARLA_Data(
                root=config.dataset_root, config=config, shared_dict=shared_dict,
                rank=rank, validation=True, ability=ability_name)
        else:
            val_set = None

        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_set, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        dataloader_train = DataLoader(
            train_set, sampler=sampler_train, batch_size=args.batch_size,
            worker_init_fn=seed_worker, generator=g_cuda,
            num_workers=num_workers, pin_memory=False, drop_last=True)

        if val_set is not None:
            sampler_val = torch.utils.data.distributed.DistributedSampler(
                val_set, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
            dataloader_val = DataLoader(
                val_set, sampler=sampler_val, batch_size=args.batch_size,
                worker_init_fn=seed_worker, generator=g_cuda,
                num_workers=num_workers, pin_memory=False, drop_last=True)
        else:
            sampler_val = None
            dataloader_val = None

        # Reset cosine scheduler for this ability so LR starts fresh
        if config.use_cosine_schedule:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=config.cosine_t0, T_mult=config.cosine_t_mult)
            trainer.scheduler = scheduler

        # Update engine's dataloaders
        trainer.dataloader_train = dataloader_train
        trainer.dataloader_val = dataloader_val
        trainer.iters_per_epoch = len(dataloader_train)

        for epoch in range(0, epochs_per_ability):
            print(f'[{ability_name}] Epoch {epoch}, lr: {scheduler.get_last_lr()}')
            sampler_train.set_epoch(epoch)
            trainer.train()
            torch.cuda.empty_cache()

            if ((args.setting != 'all') and (epoch % args.val_every == 0) and dataloader_val is not None):
                trainer.validate()
                torch.cuda.empty_cache()

            if not config.use_cosine_schedule:
                scheduler.step()

            if bool(args.zero_redundancy_optimizer):
                optimizer.consolidate_state_dict(0)
            if rank == 0:
                trainer.save()

            trainer.cur_epoch += 1
            trainer.global_epoch += 1

    if rank == 0:
        print(f'\n{"="*60}')
        print('Sequential ability training complete.')
        print(f'{"="*60}')


# ======================================================================
# Engine
# ======================================================================

class OnlineEngine(object):
    """Training engine with online alternating DPMM support.

    This extends the original ``Engine`` from ``my_train_ability_wTFFdeQtd.py``
    by integrating ``OnlineDPMMManager`` into the training loop.  When
    ``online_dpmm_manager`` is None, it behaves identically to the original
    engine.
    """

    def __init__(self, model, optimizer, dataloader_train, dataloader_val,
                 args, config, writer, device, scheduler, scaler,
                 rank=0, world_size=1, cur_epoch=0, online_dpmm_manager=None):
        self.cur_epoch = cur_epoch
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.args = args
        self.config = config
        self.writer = writer
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.step = 0
        self.global_epoch = 0  # never resets across abilities — used for TB logging
        self.vis_save_path = self.args.logdir + r'/visualizations'
        self.scheduler = scheduler
        self.iters_per_epoch = 0 if dataloader_train is None else len(dataloader_train)
        self.scaler = scaler
        self.online_dpmm_manager = online_dpmm_manager

        if self.config.debug:
            pathlib.Path(self.vis_save_path).mkdir(parents=True, exist_ok=True)

        self.detailed_loss_weights = config.detailed_loss_weights
        self.ability_logdir = args.logdir  # overridden per ability

    # ------------------------------------------------------------------
    # load_data_compute_loss (same as original)
    # ------------------------------------------------------------------

    def load_data_compute_loss(self, data, validation=False):
        future_bounding_box_label = None
        if self.config.detect_boxes or self.config.use_plant:
            bounding_box_label = data['bounding_boxes'].to(self.device, dtype=torch.float32)
            if not self.config.use_plant:
                bb_center_heatmap = data['center_heatmap'].to(self.device, dtype=torch.float32)
                bb_wh = data['wh'].to(self.device, dtype=torch.float32)
                bb_yaw_class = data['yaw_class'].to(self.device, dtype=torch.long)
                bb_yaw_res = data['yaw_res'].to(self.device, dtype=torch.float32)
                bb_offset = data['offset'].to(self.device, dtype=torch.float32)
                bb_velocity = data['velocity'].to(self.device, dtype=torch.float32)
                bb_brake_target = data['brake_target'].to(self.device, dtype=torch.long)
                bb_pixel_weight = data['pixel_weight'].to(self.device, dtype=torch.float32)
                bb_avg_factor = data['avg_factor'].to(self.device, dtype=torch.float32)
            else:
                future_bounding_box_label = data['future_bounding_boxes'].to(self.device, dtype=torch.long)
        else:
            bounding_box_label = None
            bb_center_heatmap = None
            bb_wh = None
            bb_yaw_class = None
            bb_yaw_res = None
            bb_offset = None
            bb_velocity = None
            bb_brake_target = None
            bb_pixel_weight = None
            bb_avg_factor = None

        if self.config.use_wp_gru:
            ego_waypoint = data['ego_waypoints'].to(self.device, dtype=torch.float32)
        else:
            ego_waypoint = None

        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_next = data['target_point_next'].to(self.device, dtype=torch.float32)
        command = data['command'].to(self.device, dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32).unsqueeze(1)

        if self.config.use_twohot_target_speeds:
            target_speed = data['target_speed_twohot'].to(self.device, dtype=torch.float32)
        else:
            target_speed = data['target_speed'].to(self.device, dtype=torch.long)

        if self.config.use_plant:
            checkpoint = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)
            light_hazard = data['light'].to(self.device, dtype=torch.int32).unsqueeze(1)
            stop_hazard = data['stop_sign'].to(self.device, dtype=torch.int32).unsqueeze(1)
            junction = data['junction'].to(self.device, dtype=torch.int32).unsqueeze(1)
            route = data['route'][:, :self.config.num_route_points].to(self.device, dtype=torch.float32)
            pred_wp, pred_target_speed, pred_checkpoint, pred_future_bounding_box = self.model(
                bounding_boxes=bounding_box_label, route=route, target_point=target_point,
                light_hazard=light_hazard, stop_hazard=stop_hazard, junction=junction, velocity=ego_vel)
        elif self.args.backbone in ('transFuser', 'aim', 'bev_encoder'):
            checkpoint = data['route'][:, :self.config.predict_checkpoint_len].to(self.device, dtype=torch.float32)
            rgb = data['rgb'].to(self.device, dtype=torch.float32)
            if self.config.use_semantic:
                semantic_label = data['semantic'].to(self.device, dtype=torch.long)
            else:
                semantic_label = None
            if self.config.use_bev_semantic:
                bev_semantic_label = data['bev_semantic'].to(self.device, dtype=torch.long)
            else:
                bev_semantic_label = None
            if self.config.use_depth:
                depth_label = data['depth'].to(self.device, dtype=torch.float32)
            else:
                depth_label = None
            if self.config.lidar_seq_len > 1:
                lidar = data['temporal_lidar'].to(self.device, dtype=torch.float32)
            else:
                lidar = data['lidar'].to(self.device, dtype=torch.float32)

            pred_wp, pred_target_speed, pred_trajectories, pred_traj_probs, \
                pred_semantic, pred_bev_semantic, pred_depth, pred_bounding_box, \
                _, pred_wp_1, selected_path = self.model(
                    rgb=rgb, lidar_bev=lidar, target_point=target_point,
                    ego_vel=ego_vel, command=command,
                    target_point_next=target_point_next if self.config.two_tp_input else None)
        else:
            raise ValueError('The chosen vision backbone does not exist.')

        compute_loss = self.model.module.compute_loss
        losses = compute_loss(
            pred_wp=pred_wp, pred_target_speed=pred_target_speed,
            pred_trajectories=pred_trajectories, pred_traj_probs=pred_traj_probs,
            pred_semantic=pred_semantic, pred_bev_semantic=pred_bev_semantic,
            pred_depth=pred_depth, pred_bounding_box=pred_bounding_box,
            waypoint_label=ego_waypoint, target_speed_label=target_speed,
            checkpoint_label=checkpoint, semantic_label=semantic_label,
            bev_semantic_label=bev_semantic_label, depth_label=depth_label,
            center_heatmap_label=bb_center_heatmap, wh_label=bb_wh,
            yaw_class_label=bb_yaw_class, yaw_res_label=bb_yaw_res,
            offset_label=bb_offset, velocity_label=bb_velocity,
            brake_target_label=bb_brake_target, pixel_weight_label=bb_pixel_weight,
            avg_factor_label=bb_avg_factor, pred_wp_1=pred_wp_1,
            selected_path=selected_path)

        metrics = {}
        if validation:
            if self.config.use_semantic and not self.config.use_plant:
                ss_miou = torchmetrics.functional.jaccard_index(
                    pred_semantic, semantic_label, task='multiclass',
                    num_classes=self.config.num_semantic_classes).item()
                metrics['semantic_miou'] = ss_miou
            if self.config.use_bev_semantic and not self.config.use_plant:
                valid_bev_pixels = self.model.module.valid_bev_pixels
                visible_bev_semantic_label = valid_bev_pixels.squeeze(1).int() * bev_semantic_label
                visible_bev_semantic_label = (valid_bev_pixels.squeeze(1).int() - 1) + visible_bev_semantic_label
                bev_ss_miou = torchmetrics.functional.jaccard_index(
                    pred_bev_semantic, visible_bev_semantic_label, task='multiclass',
                    ignore_index=-1, num_classes=self.config.num_bev_semantic_classes).item()
                metrics['bev_semantic_miou'] = bev_ss_miou

        self.step += 1
        return losses, metrics

    # ------------------------------------------------------------------
    # Train loop (with online DPMM integration)
    # ------------------------------------------------------------------

    def train(self):
        self.model.train()
        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_loss_weights}
        self.optimizer.zero_grad(set_to_none=False)

        for i, data in enumerate(tqdm(self.dataloader_train, disable=self.rank != 0)):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
                losses, _ = self.load_data_compute_loss(data, validation=False)
                loss = torch.zeros(1, dtype=torch.float32, device=self.device)
                for key, value in losses.items():
                    if self.config.learn_multi_task_weights:
                        precision = torch.exp(-self.detailed_loss_weights[key])
                        loss += precision * value + self.detailed_loss_weights[key]
                        detailed_losses_epoch[key] += float(precision * value + self.detailed_loss_weights[key])
                    else:
                        loss += self.detailed_loss_weights[key] * value
                        detailed_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))

            self.scaler.scale(loss).backward()

            if self.config.use_grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=int(self.config.grad_clip_max_norm),
                    error_if_nonfinite=True)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            num_batches += 1
            loss_epoch += float(loss.item())

            # --- Online DPMM: fill buffers and maybe update ---
            if self.online_dpmm_manager is not None:
                self.online_dpmm_manager.step()

                # 1. Fill traj buffer with ground-truth routes
                traj_data = data['route'][:, :self.config.predict_checkpoint_len]
                self.online_dpmm_manager.fill_traj_buffer(traj_data)

                # 2. Fill fuseFeat buffer with model's last_joined_features
                model_module = self.model.module
                if hasattr(model_module, 'last_joined_features') and model_module.last_joined_features is not None:
                    self.online_dpmm_manager.fill_fusefeat_buffer(model_module.last_joined_features)

                # 3. Maybe update traj DPMM
                traj_updated = self.online_dpmm_manager.maybe_update_traj_dpmm(model_module)

                # 4. Maybe update fuseFeat DPMM
                fusefeat_updated = self.online_dpmm_manager.maybe_update_fusefeat_dpmm(model_module)

                # Log DPMM stats
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar('dpmm/traj_buffer_filled',
                                           float(self.online_dpmm_manager.traj_buffer.num_filled),
                                           self.online_dpmm_manager.global_step)
                    self.writer.add_scalar('dpmm/fusefeat_buffer_filled',
                                           float(self.online_dpmm_manager.fusefeat_buffer.num_filled),
                                           self.online_dpmm_manager.global_step)
                    self.writer.add_scalar('dpmm/traj_num_clusters',
                                           float(len(self.online_dpmm_manager.traj_dpmm.components)),
                                           self.online_dpmm_manager.global_step)
                    self.writer.add_scalar('dpmm/fusefeat_num_clusters',
                                           float(len(self.online_dpmm_manager.fusefeat_dpmm.components)),
                                           self.online_dpmm_manager.global_step)
                    self.writer.add_scalar('dpmm/update_count',
                                           float(self.online_dpmm_manager.update_count),
                                           self.online_dpmm_manager.global_step)

            if self.config.use_cosine_schedule:
                self.scheduler.step(self.cur_epoch + i / self.iters_per_epoch)

        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')

    # ------------------------------------------------------------------
    # Validate (same as original)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch = defaultdict(float)

        for data in tqdm(self.dataloader_val, disable=self.rank != 0):
            losses, metrics = self.load_data_compute_loss(data, validation=True)
            loss = torch.zeros(1, dtype=torch.float32, device=self.device)
            for key, value in losses.items():
                if self.config.learn_multi_task_weights:
                    precision = torch.exp(-self.detailed_loss_weights[key])
                    loss += precision * value + self.detailed_loss_weights[key]
                    detailed_val_losses_epoch[key] += float(value)
                else:
                    loss += self.detailed_loss_weights[key] * value
                    detailed_val_losses_epoch[key] += float(self.detailed_loss_weights[key] * float(value.item()))
            for key, value in metrics.items():
                detailed_val_losses_epoch[key] += float(value)
            num_batches += 1
            loss_epoch += float(loss.item())
            del losses, metrics

        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        if self.rank == 0:
            self.writer.add_scalar(prefix + 'loss_total', loss_epoch / num_batches, self.global_epoch)
            for key, value in detailed_losses_epoch.items():
                self.writer.add_scalar(prefix + key, value / num_batches, self.global_epoch)

    # ------------------------------------------------------------------
    # Save (with DPMM state)
    # ------------------------------------------------------------------

    def save(self):
        save_dir = self.ability_logdir
        os.makedirs(save_dir, exist_ok=True)

        model_file = os.path.join(save_dir, f'model_{self.cur_epoch:04d}.pth')
        optimizer_file = os.path.join(save_dir, f'optimizer_{self.cur_epoch:04d}.pth')
        scaler_file = os.path.join(save_dir, f'scaler_{self.cur_epoch:04d}.pth')
        scheduler_file = os.path.join(save_dir, f'scheduler_{self.cur_epoch:04d}.pth')

        torch.save(self.model.module.state_dict(), model_file)
        torch.save(self.optimizer.state_dict(), optimizer_file)
        torch.save(self.scaler.state_dict(), scaler_file)
        torch.save(self.scheduler.state_dict(), scheduler_file)

        # Always save a "latest" copy so the next ability can find it
        latest_file = os.path.join(save_dir, 'latest.pth')
        torch.save(self.model.module.state_dict(), latest_file)

        # Save DPMM models (goes to args.logdir/dpmm_online/)
        if self.online_dpmm_manager is not None:
            self.online_dpmm_manager.save()

        # Remove last epoch's files to avoid accumulating storage,
        # EXCEPT at cosine-schedule minima (same as the original TF++ script).
        if self.cur_epoch > 0:
            keep_prev_epoch = False
            if self.config.use_cosine_schedule:
                # Keep milestones: 6, 14, 30, 62, 126, 254, 510
                # (cosine_t0=1, cosine_t_mult=2 → [7,15,31,63,...];
                #  the save deletes model_{cur-1} unless cur is a milestone,
                #  so model_{6,14,30,...} survive)
                keep_epochs = [
                    sum([self.config.cosine_t0 * self.config.cosine_t_mult**i for i in range(n)])
                    for n in range(3, 10)
                ]
                if self.cur_epoch in keep_epochs:
                    keep_prev_epoch = True
            last_model_file = os.path.join(save_dir, f'model_{self.cur_epoch - 1:04d}.pth')
            last_optimizer_file = os.path.join(save_dir, f'optimizer_{self.cur_epoch - 1:04d}.pth')
            last_scaler_file = os.path.join(save_dir, f'scaler_{self.cur_epoch - 1:04d}.pth')
            last_scheduler_file = os.path.join(save_dir, f'scheduler_{self.cur_epoch - 1:04d}.pth')
            if not keep_prev_epoch:
                for f in [last_model_file, last_optimizer_file, last_scaler_file, last_scheduler_file]:
                    if os.path.isfile(f):
                        os.remove(f)


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed()) % 2**32
    rank = int(os.environ['RANK'])
    worker_seed = worker_seed + rank * 1000
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print(f'Rank: {rank}, Worker id: {worker_id}, worker_seed: {worker_seed}')


if __name__ == '__main__':
    available_start_methods = mp.get_all_start_methods()
    if 'fork' in available_start_methods:
        mp.set_start_method('fork')
    elif 'spawn' in available_start_methods:
        mp.set_start_method('spawn')
    elif 'forkserver' in available_start_methods:
        mp.set_start_method('forkserver')
    print('Start method of multiprocessing:', mp.get_start_method())
    main()