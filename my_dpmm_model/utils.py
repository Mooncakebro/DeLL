from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import inspect
import torch
import numpy as np

import os

import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from torch.utils.data import ConcatDataset, DataLoader


def weighted_kl_divergence(bnp_model, latent_obs, mu_q, log_var_q, kl_method='soft', eps=1e-6):
    '''
    return the KL divergence D_kl(q||p), if bnp_model not exists, then calculate D_kl(q||N(0,I))
    method: soft, using soft assignment to calculate KLD between q(z|s) and DPMixture p(z)
    Input:
        latent_obs: latent encoding from encoder, concat with context encoding
        mu_q: mean vector of latent encoding
        log_var_q: log variance vector of latent encoding
    Output:
        KL divergence value between 2 gaussian distributions KL(q(z|s)||p(z))
    '''
    assert not torch.isnan(mu_q).any(), mu_q
    assert not torch.isnan(log_var_q).any(), log_var_q
    assert not torch.isnan(latent_obs).any(), latent_obs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bnp_model.model is None:
        kld_loss = -0.5 * torch.sum(1 + log_var_q - mu_q.pow(2) - log_var_q.exp(), dim=1)
        kld_loss = torch.mean(kld_loss, dim=0).to(device)
        return kld_loss

    # Convert to variance and std (add epsilon for stability)
    var_q = torch.exp(log_var_q)
    std_q = torch.sqrt(var_q + eps)

    # Build diagonal Gaussian using Independent+Normal
    q_dist = torch.distributions.Independent(
        torch.distributions.Normal(loc=mu_q, scale=std_q),
        reinterpreted_batch_ndims=1
    )

    prob_comps, comps = bnp_model.cluster_assignments(latent_obs)
    B, K = prob_comps.shape
    kl_qz_pz = torch.zeros(B).to(device)
    # Convert prob_comps to tensor if it's numpy array
    if isinstance(prob_comps, np.ndarray):
        prob_comps = torch.from_numpy(prob_comps).float().to(device)

    if kl_method == 'soft':
        for k in range(K):
            # Get component parameters
            mu_k = bnp_model.comp_mu[k].to(device)
            var_k = bnp_model.comp_var[k].to(device)
            std_k = torch.sqrt(var_k + eps)
            
            # Build component distribution
            p_dist_k = torch.distributions.Independent(
                torch.distributions.Normal(loc=mu_k, scale=std_k),
                reinterpreted_batch_ndims=1
            )
            
            kld_k = torch.distributions.kl_divergence(q_dist, p_dist_k)
            kl_qz_pz += prob_comps[:, k] * kld_k
    else:  # hard assignment
        mu_comp = torch.stack([bnp_model.comp_mu[k] for k in comps]).to(device)
        var_comp = torch.stack([bnp_model.comp_var[k] for k in comps]).to(device)
        std_comp = torch.sqrt(var_comp + eps)
        
        p_dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mu_comp, scale=std_comp),
            reinterpreted_batch_ndims=1
        )
        kl_qz_pz = torch.distributions.kl_divergence(q_dist, p_dist)

    return torch.mean(kl_qz_pz).to(device)


def collect_samples_for_tsne(model, dataloaders, device):
    # 确保使用最新权重并设置为评估模式
    model.eval()
    all_z = []
    all_scens = []

    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for dataloader in dataloaders:
            scen_name = dataloader.dataset.scen_name
            collected = 0  # 当前任务已收集样本数

            for batch in dataloader:
                if collected >= len(dataloaders)*10:
                    break  # 超过len(dataloaders)*25个样本则跳过当前任务

                # images = batch['images'].to(device)
                ego_motion = batch['ego_motion'].to(device)
                target_point = batch['target_point'].to(device)
                description = batch['enco_description'].to(device)
                # nav_cmds = batch['ego_nav_cmd'].to(device)

                # 使用当前模型权重进行编码
                # features = model.feature_extractor(images, description, nav_cmds)
                features = torch.cat([ego_motion, target_point, description], dim=1)
                z, _, _ = model.task_encoder(features)

                z_np = z.cpu().numpy()
                num_to_add = min(len(dataloaders)*10 - collected, len(z_np))

                all_z.append(z_np[:num_to_add])
                all_scens.extend([scen_name] * num_to_add)
                collected += num_to_add

    # 恢复模型原始状态（如果是训练中调用）
    model.train()

    return np.vstack(all_z), all_scens


def collect_samples_for_tsne_v2(dataloaders, device):
    # 确保使用最新权重并设置为评估模式
    # model.eval()
    all_z = []
    all_scens = []

    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for dataloader in dataloaders:
            scen_name = dataloader.dataset.scen_name
            collected = 0  # 当前任务已收集样本数

            for batch in dataloader:
                if collected >= 100:
                    break  # 超过100个样本则跳过当前任务

                # images = batch['images'].to(device)
                # descriptions = batch['encoded_description'].to(device)
                # nav_cmds = batch['ego_nav_cmd'].to(device)

                # # 使用当前模型权重进行编码
                # features = model.feature_extractor(images, descriptions, nav_cmds)
                # z, _, _ = model.task_encoder(features)

                z = batch['flatten_trajectory_points'].to(device) 

                z_np = z.cpu().numpy()
                num_to_add = min(100 - collected, len(z_np))

                all_z.append(z_np[:num_to_add])
                all_scens.extend([scen_name] * num_to_add)
                collected += num_to_add

    # 恢复模型原始状态（如果是训练中调用）
    # model.train()

    return np.vstack(all_z), all_scens

def collect_samples_for_tsne_v3(model, dataloaders, device):
    # 确保使用最新权重并设置为评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_z = []
    all_abilities = []

    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for dataloader in dataloaders:
            ability_name = dataloader.dataset.ability
            collected = 0  # 当前任务已收集样本数

            for batch in dataloader:
                if collected >= 100:
                    break  # 超过100个样本则跳过当前任务

                # images = batch['images'].to(device)
                # descriptions = batch['encoded_description'].to(device)
                # nav_cmds = batch['ego_nav_cmd'].to(device)

                # # 使用当前模型权重进行编码
                # features = model.feature_extractor(images, descriptions, nav_cmds)
                # z, _, _ = model.task_encoder(features)

                x = batch['dpmm_data'].to(device) 
                z = model(x)

                z_np = z.cpu().numpy()
                num_to_add = min(100 - collected, len(z_np))

                all_z.append(z_np[:num_to_add])
                all_abilities.extend([ability_name] * num_to_add)
                collected += num_to_add

    # 恢复模型原始状态（如果是训练中调用）
    model.train()

    return np.vstack(all_z), all_scens

def visualize_tsne(z, labels, dataloaders=None, save_path=''):
    tsne = TSNE(n_components=2, random_state=42)
    z_2d = tsne.fit_transform(z)

    plt.figure(figsize=(10, 8))
    unique_scens = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_scens)))

    for scen, color in zip(unique_scens, colors):
        idxs = [i for i, s in enumerate(labels) if s == scen]
        plt.scatter(z_2d[idxs, 0], z_2d[idxs, 1], c=[color], label=scen, alpha=0.7)

    plt.legend()
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    # 保存图像
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"tsne_task_{len(dataloaders)}.png"))
    plt.close()
    print(f"Saved t-SNE visualization to {save_path}")


def plot_losses(dyn_losses, recon_losses, kl_losses, task_id, epoch, save_path=''):
    """Plot and save losses for each epoch."""
    plt.figure(figsize=(10, 5))
    epochs = range(1, epoch + 2)  # Convert 0-based epoch to 1-based indexing
    plt.plot(epochs, dyn_losses, label="Dyn Loss", color="green")
    plt.plot(epochs, recon_losses, label="Recon Loss", color="blue")
    plt.plot(epochs, kl_losses, label="KL Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Task {task_id + 1} Losses Over Epochs")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"task_{task_id + 1}_epoch_{epoch + 1}.png"))
    plt.close()


def reset_optimizer(optimizer):
    """安全地重置优化器内部状态"""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                param_state = optimizer.state[p]
                param_state['step'] = 0
                if 'exp_avg' in param_state:
                    param_state['exp_avg'].zero_()
                if 'exp_avg_sq' in param_state:
                    param_state['exp_avg_sq'].zero_()

def convert_tensor_to_list(data):
    if isinstance(data, dict):
        return {k: convert_tensor_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_tensor_to_list(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.tolist()  # 将tensor转换为list
    else:
        return data
    

def purge_invalid_values(z_samples, tensor_name="z_samples"):
    """
    Purge NaN and Inf values from tensor and provide detailed information.
    
    Args:
        z_samples (torch.Tensor): Input tensor to clean
        tensor_name (str): Name of the tensor for logging purposes
    
    Returns:
        torch.Tensor: Cleaned tensor with NaN and Inf values removed
    """
    print(f"\n{'='*50}")
    print(f"PURGING INVALID VALUES FROM {tensor_name.upper()}")
    print(f"{'='*50}")
    
    # Initial tensor information
    print(f"Original {tensor_name} shape: {z_samples.shape}")
    print(f"Original {tensor_name} dtype: {z_samples.dtype}")
    print(f"Original {tensor_name} numel: {z_samples.numel()}")
    
    if z_samples.numel() == 0:
        print(f"Warning: {tensor_name} is empty!")
        return z_samples
    
    # Check for NaN and Inf values
    has_nan = torch.isnan(z_samples).any().item()
    has_inf = torch.isinf(z_samples).any().item()
    
    print(f"Has NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")
    
    if not (has_nan or has_inf):
        print(f"No invalid values found in {tensor_name}")
        return z_samples
    
    # Detailed NaN analysis
    if has_nan:
        nan_count = torch.isnan(z_samples).sum().item()
        nan_percentage = (nan_count / z_samples.numel()) * 100
        print(f"NaN count: {nan_count} ({nan_percentage:.2f}%)")
        
        # Find NaN positions
        nan_positions = torch.isnan(z_samples).nonzero(as_tuple=False)
        print(f"First 10 NaN positions: {nan_positions[:10] if len(nan_positions) > 10 else nan_positions}")
        
        # For 2D tensors, show which rows/columns have NaN
        if z_samples.dim() == 2:
            rows_with_nan = torch.isnan(z_samples).any(dim=1).sum().item()
            cols_with_nan = torch.isnan(z_samples).any(dim=0).sum().item()
            print(f"Rows with NaN: {rows_with_nan}/{z_samples.shape[0]}")
            print(f"Cols with NaN: {cols_with_nan}/{z_samples.shape[1] if len(z_samples.shape) > 1 else 0}")
    
    # Detailed Inf analysis
    if has_inf:
        inf_count = torch.isinf(z_samples).sum().item()
        inf_percentage = (inf_count / z_samples.numel()) * 100
        print(f"Inf count: {inf_count} ({inf_percentage:.2f}%)")
        
        # Separate positive and negative infinity
        if hasattr(torch, 'isposinf'):
            pos_inf_count = torch.isposinf(z_samples).sum().item()
            neg_inf_count = torch.isneginf(z_samples).sum().item()
        else:
            pos_inf_count = (z_samples == float('inf')).sum().item()
            neg_inf_count = (z_samples == float('-inf')).sum().item()
        
        print(f"  Positive Inf: {pos_inf_count}")
        print(f"  Negative Inf: {neg_inf_count}")
        
        # Find Inf positions
        inf_positions = torch.isinf(z_samples).nonzero(as_tuple=False)
        print(f"First 10 Inf positions: {inf_positions[:10] if len(inf_positions) > 10 else inf_positions}")
        
        # For 2D tensors, show which rows/columns have Inf
        if z_samples.dim() == 2:
            rows_with_inf = torch.isinf(z_samples).any(dim=1).sum().item()
            cols_with_inf = torch.isinf(z_samples).any(dim=0).sum().item()
            print(f"Rows with Inf: {rows_with_inf}/{z_samples.shape[0]}")
            print(f"Cols with Inf: {cols_with_inf}/{z_samples.shape[1] if len(z_samples.shape) > 1 else 0}")
    
    # Statistical summary
    finite_mask = torch.isfinite(z_samples)
    finite_count = finite_mask.sum().item()
    print(f"Finite values: {finite_count}/{z_samples.numel()} ({(finite_count/z_samples.numel())*100:.2f}%)")
    
    if finite_count == 0:
        print(f"Warning: No finite values found in {tensor_name}!")
        return torch.empty(0, dtype=z_samples.dtype, device=z_samples.device)
    
    # Remove invalid values based on tensor dimensionality
    if z_samples.dim() == 1:
        # For 1D tensor, remove individual invalid values
        z_samples_clean = z_samples[finite_mask]
        print(f"1D tensor: Removed {z_samples.numel() - z_samples_clean.numel()} invalid values")
        
    elif z_samples.dim() == 2:
        # For 2D tensor, remove rows with any invalid values
        row_finite_mask = torch.isfinite(z_samples).all(dim=1)
        z_samples_clean = z_samples[row_finite_mask]
        rows_removed = z_samples.shape[0] - z_samples_clean.shape[0]
        print(f"2D tensor: Removed {rows_removed} rows containing invalid values")
        if rows_removed > 0:
            invalid_rows = (~row_finite_mask).nonzero(as_tuple=True)[0]
            print(f"Invalid row indices (first 10): {invalid_rows[:10] if len(invalid_rows) > 10 else invalid_rows}")
    else:
        # For higher dimensional tensors, flatten and remove invalid values
        original_shape = z_samples.shape
        z_samples_flat = z_samples.flatten()
        finite_mask_flat = torch.isfinite(z_samples_flat)
        z_samples_clean_flat = z_samples_flat[finite_mask_flat]
        z_samples_clean = z_samples_clean_flat
        print(f"Higher dimensional tensor: Removed {z_samples_flat.numel() - z_samples_clean.numel()} invalid values")
        print(f"Note: Reshaped to 1D for cleaning")
    
    # Final validation
    final_has_nan = torch.isnan(z_samples_clean).any().item()
    final_has_inf = torch.isinf(z_samples_clean).any().item()
    
    print(f"\nAfter cleaning:")
    print(f"  Cleaned shape: {z_samples_clean.shape}")
    print(f"  Final has NaN: {final_has_nan}")
    print(f"  Final has Inf: {final_has_inf}")
    print(f"  Data preserved: {z_samples_clean.numel()}/{z_samples.numel()} ({(z_samples_clean.numel()/z_samples.numel())*100:.2f}%)")
    
    if z_samples_clean.numel() > 0:
        print(f"  Min value: {z_samples_clean.min().item()}")
        print(f"  Max value: {z_samples_clean.max().item()}")
        print(f"  Mean value: {z_samples_clean.mean().item()}")
    
    print(f"{'='*50}\n")
    
    return z_samples_clean


def evaluate_clustering_internal(data, y_pred, method_name):
    """Internal function to evaluate clustering results"""
    unique_labels = np.unique(y_pred)
    valid_indices = y_pred != -1  # Filter noise points for DBSCAN
    X_valid = data[valid_indices]
    y_valid = y_pred[valid_indices]

    if len(unique_labels) <= 1 or len(np.unique(y_valid)) <= 1:
        print(f"{method_name} produced invalid clusters, skipping metrics")
        return None

    try:
        silhouette = silhouette_score(X_valid, y_valid)  # -1~1 How well-separated clusters are. Higher better
        ch_score = calinski_harabasz_score(X_valid, y_valid)  # 0~+oo Ratio of between-cluster variance to within-cluster variance. Higher better 
        db_score = davies_bouldin_score(X_valid, y_valid)  # 0~+oo Average similarity between each cluster and its most similar cluster. Lower better
        
        metrics = {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(ch_score),
            'davies_bouldin_score': float(db_score),
            'n_clusters': len(np.unique(y_valid)),
            'n_valid_points': len(y_valid),
            'n_total_points': len(y_pred)
        }
        
        return metrics
    except Exception as e:
        print(f"Error calculating metrics for {method_name}: {e}")
        return None

def cluster_and_evaluate(X, dpmm_model=None, output_file='clustering_results.json', 
                        dbscan_eps=None, dbscan_min_samples=None):
    """
    Perform clustering with multiple algorithms and save results to JSON.
    
    Parameters:
    X (array-like): Input data for clustering (trajectory data)
    dpmm_model: Trained DPMM model (optional)
    output_file (str): Path to save results JSON
    dbscan_eps (float): DBSCAN epsilon parameter (auto-calculated if None)
    dbscan_min_samples (int): DBSCAN min_samples parameter (auto-calculated if None)
    
    Returns:
    dict: Clustering results and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Clustering {X.shape[0]} samples with {X.shape[1]} dimensions")
    # Convert X to tensor if DPMM expects it
    if dpmm_model is not None and isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float().to(device)  # Convert to float32 tensor
    else:
        X_tensor = X  # Assume it's already a tensor
    # print_data_info(X)
    # print_data_info(X_tensor)

    results = {
        'data_shape': X.shape,
        'clustering_results': {},
        'evaluation_metrics': {}
    }
    
    # Auto-calculate DBSCAN parameters based on data dimensionality
    if dbscan_eps is None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5).fit(X)  
        distances, _ = nbrs.kneighbors()
        dbscan_eps = np.percentile(distances[:, -1], 50)  # Median distance
        print(f"Adjusted DBSCAN eps: {dbscan_eps:.4f}")

    if dbscan_min_samples is None:
        dbscan_min_samples = min(10, max(5, X.shape[1]//2))  # 5-10 for 40D
        print(f"Auto-calculated DBSCAN min_samples: {dbscan_min_samples}")
    
    # Determine number of clusters for KMeans methods
    n_clusters = None
    if dpmm_model is not None:
        try:
            if hasattr(dpmm_model, 'dpmm') and hasattr(dpmm_model.dpmm, 'info_dict'):
                n_clusters = dpmm_model.dpmm.info_dict["K_history"][-1]
            elif hasattr(dpmm_model, 'info_dict'):
                n_clusters = dpmm_model.info_dict["K_history"][-1]
            
            if n_clusters is not None:
                results['dpmm_info'] = {
                    'final_K': n_clusters,
                    'K_history': dpmm_model.dpmm.info_dict["K_history"] if hasattr(dpmm_model, 'dpmm') else dpmm_model.info_dict["K_history"]
                }
        except Exception as e:
            print(f"Error extracting DPMM info: {e}")
    # n_clusters = 3
    
    # Determine n_clusters for KMeans methods
    if n_clusters is None or n_clusters==1:
        # For trajectory data, estimate based on sample size
        n_clusters = max(3, min(20, X.shape[0] // 20))  # More reasonable for trajectory data
        print(f"No DPMM model provided, using estimated n_clusters: {n_clusters}")
    
    # DPMM (if provided)
    if dpmm_model is not None:
        try:
            if hasattr(dpmm_model, 'cluster_assignments'):
                resp, y_pred_dpmm = dpmm_model.cluster_assignments(X_tensor)  # Use X_tensor here
                # print_data_info(y_pred_dpmm)
                # y_pred_dpmm = y_pred_dpmm.cpu().numpy()  # Convert back to NumPy for evaluation
                y_pred_dpmm = y_pred_dpmm  # Convert back to NumPy for evaluation
                # results['clustering_results']['DPMM'] = y_pred_dpmm.tolist()
                metrics = evaluate_clustering_internal(X, y_pred_dpmm, "DPMM")  # Use original X (NumPy)
                if metrics:
                    results['evaluation_metrics']['DPMM'] = metrics
        except Exception as e:
            print(f"Error with DPMM clustering: {e}")

    # KMeans
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred_kmeans = kmeans.fit_predict(X)
        # results['clustering_results']['KMeans'] = y_pred_kmeans.tolist()
        metrics = evaluate_clustering_internal(X, y_pred_kmeans, "KMeans")
        if metrics:
            results['evaluation_metrics']['KMeans'] = metrics
    except Exception as e:
        print(f"Error with KMeans clustering: {e}")
    
    # KMeans++
    try:
        kmeanspp = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        y_pred_kmeanspp = kmeanspp.fit_predict(X)
        # results['clustering_results']['KMeans++'] = y_pred_kmeanspp.tolist()
        metrics = evaluate_clustering_internal(X, y_pred_kmeanspp, "KMeans++")
        if metrics:
            results['evaluation_metrics']['KMeans++'] = metrics
    except Exception as e:
        print(f"Error with KMeans++ clustering: {e}")
    
    # DBSCAN
    try:
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        y_pred_dbscan = dbscan.fit_predict(X)
        # results['clustering_results']['DBSCAN'] = y_pred_dbscan.tolist()
        metrics = evaluate_clustering_internal(X, y_pred_dbscan, "DBSCAN")
        if metrics:
            results['evaluation_metrics']['DBSCAN'] = metrics
        print(f"DBSCAN found {len(np.unique(y_pred_dbscan[y_pred_dbscan != -1]))} clusters, "
              f"{np.sum(y_pred_dbscan == -1)} noise points")
    except Exception as e:
        print(f"Error with DBSCAN clustering: {e}")
    
    # Save results to JSON
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, default=convert, indent=2, ensure_ascii=False)
        print(f"Clustering eval results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")
    
    return results


def collect_samples_for_cluster_eval(dataloaders, num_per_dataloader=100):
    # 确保使用最新权重并设置为评估模式
    all_x = []
    all_scens = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 禁用梯度计算以节省内存
    with torch.no_grad():
        for dataloader in dataloaders:
            scen_name = dataloader.dataset.scen_name
            collected = 0  # 当前任务已收集样本数

            for batch in dataloader:
                if collected >= num_per_dataloader:
                    break  # 超过100个样本则跳过当前任务

                x = batch['flatten_trajectory_points'].detach()
                # print_data_info(x)

                x_np = x.cpu().numpy()
                # print_data_info(x_np)
                num_to_add = min(100 - collected, len(x_np))

                all_x.append(x_np[:num_to_add])
                all_scens.extend([scen_name] * num_to_add)
                collected += num_to_add


    # print_data_info(np.vstack(all_x))
    return np.vstack(all_x), all_scens


# Example usage:
# X, scens = collect_samples_for_cluster_eval(dataloaders, num_per_dataloader=100)
# results = cluster_and_evaluate(X, model, 'clustering_results.json')

def print_data_info(data):
    # Get the variable name from the caller's frame
    frame = inspect.currentframe()
    try:
        # Search for the variable name in the caller's local variables
        for name, value in frame.f_back.f_locals.items():
            if value is data:
                var_name = name
                break
        else:
            var_name = "unknown"
    finally:
        del frame  # Avoid reference cycles

    # Print info based on data type
    if isinstance(data, np.ndarray):
        print(f"{var_name}: NumPy array | Shape: {data.shape} | Type: {data.dtype} | Device: CPU")
    elif isinstance(data, torch.Tensor):
        print(f"{var_name}: PyTorch tensor | Shape: {data.shape} | Type: {data.dtype} | Device: {data.device}")
    else:
        print(f"{var_name}: Unknown type ({type(data)}) | Shape: {getattr(data, 'shape', 'N/A')} | Device: N/A")


def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def combine_skill_dataloaders(skill_dataloaders):  # return a skill_dataloaders with each skill only have one dataloader
    combined_skill_dataloaders = {}
    for skill, dataloaders in skill_dataloaders.items():
        # 假设每个 DataLoader 的 dataset 是相同的类型
        datasets = [dl.dataset for dl in dataloaders]
        combined_dataset = ConcatDataset(datasets)

        # 创建新的 DataLoader
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=4,
            shuffle=True,      # 可以打乱
            num_workers=0,
        )
        combined_skill_dataloaders[skill] = combined_dataloader
    return combined_skill_dataloaders
