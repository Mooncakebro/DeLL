import os
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
from tqdm import tqdm

# Import your existing modules
from my_model_wTFFdeQtd import LidarCenterNet
from config import GlobalConfig
from ability_data import Ability_CARLA_Data
from torch.utils.data import DataLoader
from my_dpmm_model import BNPModel

import pathlib
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import ujson  # Like json but faster
import gzip

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)

from utils import print_data_info


class FeatureExtractor:
    """
    Extracts joined_checkpoint_features from trained model for DPMM clustering
    """
    
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device
        self.model = self.load_model(model_path)
        self.feature_buffer = []
        self.sample_buffer = []
        
    def load_model(self, model_path):
        """Load trained model"""
        model = LidarCenterNet(self.config)
        
        if self.config.sync_batch_norm:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        print(f"Loaded model from {model_path}")
        return model
    
    def extract_features(self, dataloader, max_num_batches=None):
        """
        Extract joined_checkpoint_features from model
        
        Returns:
            features: tensor of shape (N, 11, 256) - joined_checkpoint_features
            samples: corresponding input data for reference
        """
        self.feature_buffer = []
        
        batch_count = 0
        with torch.no_grad():
            # for batch in tqdm(dataloader, desc="Extracting features"):
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=bool(self.config.use_amp)):
                    # if num_batches and batch_count >= num_batches:
                    #     break
                        
                    # Move data to device
                    # batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    #         for k, v in batch.items()}
                    
                    # Forward pass to get features
                    # print_data_info(batch['rgb'])
                    features = self._forward_pass(batch)
                    
                    if features is not None:
                        self.feature_buffer.append(features)  # might need to save every several batches
                    
                    batch_count += 1
        
        if self.feature_buffer:
            all_features = torch.cat(self.feature_buffer, dim=0)
            print(f"Extracted {len(all_features)} feature samples")
            print_data_info(all_features)
            return all_features
        else:
            return None
    
    def _forward_pass(self, data):
        """Perform forward pass and extract joined_checkpoint_features"""
        # try:
        # Get model outputs

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

        with torch.no_grad():
            pred_wp, pred_target_speed, pred_trajectories, pred_traj_probs, \
            pred_semantic, pred_bev_semantic, pred_depth, pred_bounding_box, \
            attention_weights, pred_wp_1, selected_path = self.model(
                rgb=rgb,
                lidar_bev=lidar,
                target_point=target_point,
                ego_vel=ego_vel,
                command=command
            )
        
        # For debugging: print model outputs
        # print(f"Model outputs - trajectories: {pred_trajectories.shape if pred_trajectories is not None else 'None'}")
        # print(f"Model outputs - traj_probs: {pred_traj_probs.shape if pred_traj_probs is not None else 'None'}")
        
        # Extract the actual joined_checkpoint_features from the model
        # This depends on your model implementation
        features = self._extract_joined_features_from_model()
        # print(f"Model outputs - features: {features.shape if features is not None else 'None'}")
        
        # # Prepare sample data for reference
        # samples = {
        #     'target_point': batch['target_point'].cpu(),
        #     'speed': batch['speed'].cpu(),
        #     'command': batch['command'].cpu()
        # }
        
        return features
            
        # except Exception as e:
        #     print(f"Error in forward pass: {e}")
        #     return None
    
    def _extract_joined_features_from_model(self):
        """
        Extract the actual joined_checkpoint_features from model internals
        This needs to be adapted based on your model implementation
        """
        # Method 1: If your model stores intermediate features
        if hasattr(self.model, 'last_joined_features'):
            # print('get joined_checkpoint_features from last_joined_features')
            # print(f'self.model.last_joined_features shape: {self.model.last_joined_features.shape}')
            return self.model.last_joined_features.detach()
        
        else:
            print('get no joined_checkpoint_features')
            return 
        
        # # Method 2: Hook into specific layer (adapt layer name as needed)
        # features = []
        
        # def hook_fn(module, input, output):
        #     features.append(output.detach().cpu())
        
        # # Register hook - adjust layer name based on your model architecture
        # target_layer = None
        # for name, module in self.model.named_modules():
        #     if 'join' in name and isinstance(module, torch.nn.TransformerDecoder):
        #         target_layer = module
        #         break
        
        # if target_layer is None:
        #     print("Warning: Could not find join layer, using alternative extraction")
        #     # Alternative: use the GRU features before trajectory decoding
        #     for name, module in self.model.named_modules():
        #         if 'checkpoint_query' in name:
        #             print(f"Found checkpoint query: {name}")
        
        # # For now, return dummy features - YOU NEED TO IMPLEMENT THIS BASED ON YOUR MODEL
        # print("Warning: Using dummy features - implement proper feature extraction")
        # return torch.randn(1, 11, 256)  # Dummy features


class DpmmFeatureTrainer:
    """
    Trains DPMM on extracted joined_checkpoint_features
    """
    
    def __init__(self, dpmm_config, save_dir, load_path=None):
        self.dpmm_config = dpmm_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.cluster_dir = self.save_dir / "tracked_clusters"
        self.feature_dir = self.save_dir / "extracted_features"
        self.cluster_dir.mkdir(exist_ok=True)
        self.feature_dir.mkdir(exist_ok=True)

        self.dpmm_save_dir = self.save_dir / "dpmm_model"
        self.dpmm_save_dir.mkdir(exist_ok=True)

        self.dpmm = BNPModel(
            save_dir=str(self.dpmm_save_dir),
            gamma0=dpmm_config["gamma0"],
            num_lap=dpmm_config["num_lap"],
            sF=dpmm_config["sF"]
        )
        if load_path != None:
            self.dpmm.load_model(load_path)
            print(f'DPMM model load from {load_path}.')
        
    def train_dpmm_on_features(self, features, dataset_name, epochs=1, iterations_per_epoch=1):
        """
        Train DPMM on extracted features with periodic sampling
        """
        print(f"Starting DPMM training on {len(features)} features")
        
        # Flatten features for DPMM: (N, 11, 256) -> (N, 2816)
        features_flat = features.reshape(features.shape[0], -1)
        print(f"Flattened features shape: {features_flat.shape}")
        
        # Split data for incremental training
        num_samples = len(features_flat)
        samples_per_iteration = max(1, num_samples // iterations_per_epoch)
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            for iteration in range(iterations_per_epoch):
                print(f"Iteration {iteration + 1}/{iterations_per_epoch}")
                
                # Get current batch of features
                start_idx = iteration * samples_per_iteration
                end_idx = min((iteration + 1) * samples_per_iteration, num_samples)
                current_features = features_flat[start_idx:end_idx]
                # print_data_info(current_features)

                # if iteration > 0 or epoch > 0:
                # Sample from DPMM and combine with new data
                K = len(self.dpmm.components)
                new_data_ratio = 1 if K==0 else 0.5
                num_to_sample = int(min((1 - new_data_ratio) * len(current_features) / new_data_ratio, 5000))
                
                print(f"Sampling {num_to_sample} from DPMM, adding {len(current_features)} new features")
                
                # Sample from current DPMM
                sampled_features = self.dpmm.sample_all(num_samples=num_to_sample)
                # print_data_info(sampled_features)
                
                # Combine sampled and new features
                combined_features = torch.cat([sampled_features, current_features], dim=0)
                # else:
                #     # First iteration - just use current features
                #     combined_features = current_features
                
                # Clean data
                combined_features = self._purge_invalid_values(combined_features, "combined_features")
                
                if len(combined_features) > 0:
                    # Fit DPMM
                    self.dpmm.fit(combined_features)
                    
                    # Save tracked clusters
                    self.save_tracked_clusters(
                        dataset_name=dataset_name,
                        epoch=epoch,
                        iteration=iteration,
                        features=current_features,
                    )
                    
                    # Print current cluster info
                    self.print_cluster_info(epoch, iteration)
    
    def save_tracked_clusters(self, dataset_name, epoch, iteration, features=None):
        """Save tracked clusters in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # base_name = f"{dataset_name}-epoch{epoch:02d}-iter{iteration:03d}-{timestamp}"
        base_name = f"{timestamp}"
        
        # Get current clusters
        tracked_clusters = self.dpmm.get_current_cluster_list()
        
        # Prepare data for saving
        save_data = {
            'tracked_clusters': tracked_clusters,
            'cluster_history': self.dpmm.get_all_cluster_history(),
            'components': self.dpmm.components,
            'metadata': {
                'dataset_name': dataset_name,
                'epoch': epoch,
                'iteration': iteration,
                'timestamp': timestamp,
                'num_clusters': len(tracked_clusters),
                'dpmm_config': self.dpmm_config
            }
        }
        
        # Add features and samples if provided
        if features is not None:
            save_data['features'] = {
                'shape': features.shape,
                'mean': features.mean(dim=0).tolist(),
                'std': features.std(dim=0).tolist()
            }
        
        # Save in multiple formats
        self._save_as_pickle(save_data, base_name)
        # self._save_as_torch(save_data, base_name)
        self._save_as_json(save_data, base_name)
        
        print(f"Saved tracked clusters: {base_name}")
    
    def _save_as_pickle(self, data, base_name):
        """Save as pickle file"""
        pickle_path = self.cluster_dir / f"{base_name}-tracked_clusters.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_as_torch(self, data, base_name):
        """Save as torch file"""
        torch_path = self.cluster_dir / f"{base_name}-tracked_clusters.pt"
        
        # Convert to tensor-friendly format
        torch_data = {
            'tracked_clusters': data['tracked_clusters'],
            'components': data['components'],
            'metadata': data['metadata']
        }
        
        torch.save(torch_data, torch_path)
    
    def _save_as_json(self, data, base_name):
        """Save metadata as JSON"""
        json_path = self.cluster_dir / f"{base_name}-metadata.json"
        
        # Convert to JSON-serializable format
        json_data = {
            'metadata': data['metadata'],
            'cluster_summary': {
                'num_clusters': len(data['tracked_clusters']),
                'cluster_ids': [cluster['cluster_id'] for cluster in data['tracked_clusters']],
                'cluster_sizes': [cluster.get('size', 0) for cluster in data['tracked_clusters']]
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def print_cluster_info(self, epoch, iteration):
        """Print current cluster information"""
        clusters = self.dpmm.get_current_cluster_list()
        print(f"Epoch {epoch}, Iteration {iteration}: {len(clusters)} clusters")
        
        for cluster in clusters:
            print(f"  Cluster {cluster['cluster_id']}: "
                  f"mu_norm={torch.norm(cluster['mu']):.3f}, "
                  f"var_sum={torch.sum(cluster['var']):.3f}")
    
    def _purge_invalid_values(self, tensor, name):
        """Remove invalid values from tensor"""
        valid_mask = ~torch.any(torch.isnan(tensor), dim=1)
        valid_mask &= ~torch.any(torch.isinf(tensor), dim=1)
        
        if valid_mask.all():
            return tensor
        
        num_invalid = len(tensor) - valid_mask.sum()
        print(f"Removed {num_invalid} invalid samples from {name}")
        
        return tensor[valid_mask]


def main():
    """Main execution function"""
    
    # DPMM configuration
    dpmm_config = {
        "gamma0": 500,
        "num_lap": 1000,
        "sF": 1e-5
    }
    
    # Paths - adjust these according to your setup
    # dataset_root = "/home/syb/b2d_mini_v2"  # /media/syb/syb_disk_2/b2d_base_v3/carla_dataset
    dataset_root = '/share/home/u19666033/syb/pdm_dataset'

    model_folder = "./log/syb_TFFdeQtd_2stg"  # need choose
    dpmm_load_path = None # need choose 'Give_Way', 'Overtaking', 'Merging', 'Traffic_Sign', 'Emergency_Brake'

    model_path = os.path.join(model_folder, "model_0001.pth")
    config_path = os.path.join(model_folder, "config.json")
        # Load the config saved during training
    with open(config_path, 'rt', encoding='utf-8') as f:
      json_config = f.read()

    loaded_config = jsonpickle.decode(json_config)

    # Generate new config for the case that it has new variables.
    config = GlobalConfig()
    # Overwrite all properties that were set in the saved config.
    config.__dict__.update(loaded_config.__dict__)
    
    # Select ability for dataset
    ability = 'All_Ability'  # need choose. Adjust based on your dataset 'Give_Way', 'Overtaking', 'Merging', 'Traffic_Sign', 'Emergency_Brake'
    ability_list = ['No_Scenario','Give_Way', 'Overtaking', 'Merging', 'Traffic_Sign', 'Emergency_Brake']

    dataset_name = ability  


    output_dir = os.path.join("/home/syb/carla_garage/dpmm_feature/noMoe_wTFFdeQtd", ability)  # need choose
    # dpmm_save_dir = os.path.join(output_dir, "dpmm_model")
    # os.makedirs(dpmm_save_dir)
    output_dir = str(output_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = Ability_CARLA_Data(
        # root=dataset_root,
        # config=config,
        # estimate_class_distributions=config.estimate_class_distributions,
        # estimate_sem_distribution=config.estimate_semantic_distribution,
        # shared_dict=None,
        # validation=False,
        root=dataset_root,
        config=config,
        shared_dict=None,
        rank=0,
        validation=False,
        # ability=ability,
        ability_list=ability_list
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,  # Adjust based on your GPU memory
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(config, model_path, device)
    
    # Extract features (limit batches for testing)
    print("Extracting features...")
    features = feature_extractor.extract_features(
        dataloader, 
        max_num_batches=None  # Adjust based on your needs
    )
    
    if features is None:
        print("No features extracted. Exiting.")
        return
    
    # # Save extracted features for later analysis
    # features_save_path = Path(output_dir) / "extracted_features" / f"{dataset_name}_features.pt"
    # features_save_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save({
    #     'features': features,
    #     'dataset_name': dataset_name,
    #     'config': dpmm_config
    # }, features_save_path)
    # print(f"Saved features to {features_save_path}")
    
    # Train DPMM on features
    print("Training DPMM on extracted features...")
    dpmm_trainer = DpmmFeatureTrainer(dpmm_config, output_dir, load_path=dpmm_load_path)
    
    dpmm_trainer.train_dpmm_on_features(
        features=features,
        dataset_name=dataset_name,
        epochs=1,  # Adjust based on your needs
        iterations_per_epoch=1  # Adjust based on your needs
    )
    
    # Save final DPMM model
    dpmm_path = Path(output_dir) / "dpmm_model"
    dpmm_trainer.dpmm.save_model(str(dpmm_path))
    print(f"Saved DPMM model to {dpmm_path}")
    
    print("DPMM feature training completed!")


if __name__ == "__main__":
    main()