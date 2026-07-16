import os
import bnpy
# from ...bnpy.bnpy.ioutil.ModelReader import load_model_at_prefix
# from bnpy import load_model_at_prefix
from bnpy.ioutil.ModelReader import load_model_at_prefix
import numpy as np
import torch
from itertools import cycle
from bnpy.data.XData import XData
from matplotlib import pylab
import matplotlib.pyplot as plt
from collections import defaultdict
from collections.abc import Mapping, Iterable
import json

FIG_SIZE = (3, 3)
pylab.rcParams['figure.figsize'] = FIG_SIZE



class BNPModel:
    def __init__(self, 
                 save_dir, 
                 gamma0=5.0, 
                 num_lap=100,
                 sF=0.00001,
                 birth_kwargs=None, 
                 merge_kwargs=None):
        super(BNPModel, self).__init__()
        # for DPMM model
        self.model = None
        self.info_dict = None
        self.iterator = cycle(range(2))

        self.save_dir = save_dir
        self.num_lap = num_lap
        self.sF = sF
        if not os.path.exists(os.path.join(self.save_dir, 'birth_debug')):
            os.makedirs(os.path.join(self.save_dir, 'birth_debug'))
        if not os.path.exists(os.path.join(self.save_dir, 'data')):
            os.makedirs(os.path.join(self.save_dir, 'data'))

        self.birth_kwargs = birth_kwargs
        self.merge_kwargs = merge_kwargs
        self.gamma0 = gamma0  # concentration parameter of  the DP process
        self.birth_kwargs = dict(
            # b_startLap=2,
            # b_stopLap=2,
            # b_Kfresh=4,
            # b_minNumAtomsForNewComp=16.0,
            # b_minNumAtomsForTargetComp=16.0,
            # b_minNumAtomsForRetainComp=16.0,
            # b_minPercChangeInNumAtomsToReactivate=0.1,
            # b_debugOutputDir=None,   #os.path.join(self.save_dir, 'birth_debug'),  # for debug
            # b_debugWriteHTML=0,  # for debug
        )
        
        self.merge_kwargs = dict(
            # m_startLap=5,
            # Set limits to number of merges attempted each lap.
            # This value specifies max number of tries for each cluster
            # Setting this very high (to 50) effectively means try all pairs
            m_maxNumPairsContainingComp=50,
            # # Set "reactivation" limits
            # # So that each cluster is eligible again after 10 passes thru dataset
            # # Or when it's size changes by 400%
            # m_nLapToReactivate=1,
            # Specify how to rank pairs (determines order in which merges are tried)
            # 'obsmodel_elbo' means rank pairs by improvement to observation model ELBO
            m_pair_ranking_procedure='obsmodel_elbo',
            # # 'total_size' and 'descending' means try largest combined clusters first
            # # m_pair_ranking_procedure='total_size',
            # m_pair_ranking_direction='descending',
        )

        self.comp_mu = None
        self.comp_var = None

        # New attributes for tracking clusters
        self.cluster_history = []  # List of all clusters ever created
        self.current_clusters = {}  # Current mapping {component_index: {'id': int, 'mu': torch.tensor 'var': torch.tensor}}
        self.next_available_id = 0  # To assign unique IDs to new clusters

        self.components = []

    def show_clusters_over_time(self, task_output_path=None, query_laps=[0, 1, 2, 5, 10, None], nrows=2, data=None):
        task_output_path = self.info_dict['task_output_path']
        ncols = int(np.ceil(len(query_laps) // float(nrows)))
        fig_handle, ax_handle_list = pylab.subplots(
            figsize=(FIG_SIZE[0] * ncols, FIG_SIZE[1] * nrows),
            nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        for plot_id, lap_val in enumerate(query_laps):
            cur_model, lap_val = bnpy.load_model_at_lap(task_output_path, lap_val)
            # Plot the current model
            cur_ax_handle = ax_handle_list.flatten()[plot_id]
            bnpy.viz.PlotComps.plotCompsFromHModel(
                cur_model, Data=data, ax_handle=cur_ax_handle)
            # cur_ax_handle.set_xticks([-2, -1, 0, 1, 2])
            # cur_ax_handle.set_yticks([-2, -1, 0, 1, 2])
            cur_ax_handle.set_xlabel("lap: %d" % lap_val)
            # cur_ax_handle.set_xlim([-2, 2])
            # cur_ax_handle.set_ylim([-2, 2])
        pylab.tight_layout()
        plt.show()  # ⬅️ 这里会暂停程序直到你关闭图像窗口

    def fit(self, z):
        '''
        fit the model, input z should be torch.tensor format
        '''
        z = XData(z.detach().cpu().numpy())
        if not self.model:
            print("=== Initialing DPMM model ===")
            self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',  # 'DiagGauss'
                                                  output_path=os.path.join(self.save_dir,
                                                                           str(next(self.iterator))),
                                                  initname='randexamples',
                                                  K=1, gamma0=self.gamma0,
                                                  sF=self.sF, ECovMat='eye',
                                                  moves='birth,merge', nBatch=1, nLap=self.num_lap,
                                                  **dict(
                                                      sum(map(list, [self.birth_kwargs.items(),
                                                                     self.merge_kwargs.items()]), []))
                                                  )
            # print(f'model:\n {self.model}.')
            # print(f'info_dict:\n {self.info_dict}.')
            # print(f'model and info_dict type: {type(self.model)} and {type(self.info_dict)}.')
            print("=== DPMM model initialized ===")
        else:
            self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',  # 'DiagGauss'
                                                  output_path=os.path.join(self.save_dir,
                                                                           str(next(self.iterator))),
                                                  initname=self.info_dict['task_output_path'],
                                                  K=self.info_dict['K_history'][-1], gamma0=self.gamma0,
                                                  sF=self.sF, ECovMat='eye',
                                                  moves='birth,merge', nBatch=1, nLap=self.num_lap,
                                                  **dict(
                                                      sum(map(list, [self.birth_kwargs.items(),
                                                                     self.merge_kwargs.items()]), []))
                                                  )
            # print(f'info_dict:\n {self.info_dict}.')
            # print(f'model and info_dict: {self.model} and {self.info_dict}.')
            # print(f'model and info_dict type: {type(self.model)} and {type(self.info_dict)}.')    
        self.calc_cluster_component_params()

        # Get current tracked clusters
        # tracked_clusters = self.get_current_cluster_list()
        # sorted_tracked_clusters = sorted(tracked_clusters, key=lambda x: x['cluster_id'])
        # for item in sorted_tracked_clusters:
        #     print(f"Cluster ID: {item['cluster_id']}, Mean: {item['mu']}, Variance: {item['var']}")


    def plot_clusters(self, z, suffix=""):
        # save best model for debugging
        cur_model, lap_val = bnpy.load_model_at_lap(self.info_dict['task_output_path'], None)
        bnpy.viz.PlotComps.plotCompsFromHModel(cur_model, Data=z)
        pylab.savefig(os.path.join(self.save_dir+'/birth_debug', "dpmm_" + suffix + ".png"))

    def calc_cluster_component_params(self):
        # Compute current component parameters
        self.comp_mu = [
            torch.tensor(self.model.obsModel.get_mean_for_comp(i), dtype=torch.float32)
            for i in range(self.model.obsModel.K)
        ]
        self.comp_var = [
            torch.tensor(
                np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0),
                dtype=torch.float32
            )
            for i in range(self.model.obsModel.K)
        ]

        #print the components (not matching id, all are cpu numpy array)
        print("-"*15)
        print(f'K = {self.model.obsModel.K}')
        self.components = []
        for i in range(self.model.obsModel.K):
            print(f'{i}th component:')
            print(f'mu:')
            print(self.comp_mu[i].cpu().numpy())
            print(f'var:')
            print(self.comp_var[i].cpu().numpy())
            self.components.append({'k':i, 'mu':self.comp_mu[i].tolist(), 'var':self.comp_var[i].tolist()})
        print("-"*15)

        # First run: initialize all clusters
        if not self.current_clusters:
            self.current_clusters = {}
            for idx, (mu, var) in enumerate(zip(self.comp_mu, self.comp_var)):
                cluster_id = self.next_available_id
                self.current_clusters[idx] = {'id': cluster_id, 'mu': mu, 'var': var}
                self.cluster_history.append({
                    'id': cluster_id,
                    'mu': mu,
                    'var': var,
                    'active': True
                })
                self.next_available_id += 1
            return

        # Find best matches between new components and old clusters
        matches = []  # (kl_divergence, new_idx, old_id)
        
        for new_idx, (mu_new, var_new) in enumerate(zip(self.comp_mu, self.comp_var)):
            for old_idx, old_data in self.current_clusters.items():
                kl = self.kl_divergence_diagonal_gaussian(
                    old_data['mu'], 
                    mu_new, 
                    old_data['var'], 
                    var_new
                ).item()
                matches.append((kl, new_idx, old_data['id']))
        
        # Sort matches by KL divergence (best matches first)
        matches.sort(key=lambda x: x[0])
        
        # Greedy assignment
        new_current_clusters = {}
        used_old_ids = set()
        assigned_new_idxs = set()
        kl_threshold = 1.0  # Adjust based on your data scale
        
        for kl, new_idx, old_id in matches:
            # Skip if already assigned or KL too high
            if new_idx in assigned_new_idxs or old_id in used_old_ids or kl > kl_threshold:
                continue
            
            # Assign this old ID to the new component
            new_current_clusters[new_idx] = {
                'id': old_id,
                'mu': self.comp_mu[new_idx],
                'var': self.comp_var[new_idx]
            }
            assigned_new_idxs.add(new_idx)
            used_old_ids.add(old_id)
        
        # Handle new components without good matches
        for new_idx in range(len(self.comp_mu)):
            if new_idx in assigned_new_idxs:
                continue
                
            # Create new cluster
            cluster_id = self.next_available_id
            new_current_clusters[new_idx] = {
                'id': cluster_id,
                'mu': self.comp_mu[new_idx],
                'var': self.comp_var[new_idx]
            }
            self.cluster_history.append({
                'id': cluster_id,
                'mu': self.comp_mu[new_idx],
                'var': self.comp_var[new_idx],
                'active': True
            })
            self.next_available_id += 1
        
        # Update active status for old clusters
        active_ids = {data['id'] for data in new_current_clusters.values()}
        for cluster in self.cluster_history:
            cluster['active'] = cluster['id'] in active_ids
        
        self.current_clusters = new_current_clusters

    def get_current_cluster_list(self):
        """
        Return current clusters as a list of dicts with cluster IDs and parameters.
        Format: [{'cluster_id': 0, 'mu': numpy array, 'var': numpy array}, ...]
        """
        return [
            {
                'cluster_id': data['id'],
                'mu': data['mu'],
                'var': data['var']
            } for data in self.current_clusters.values()
        ]

    def get_all_cluster_history(self):
        """
        Return full history of all clusters including active/inactive status.
        """
        return self.cluster_history

    def kl_divergence_diagonal_gaussian(self, mu_1, mu_2, var_1, var_2):
        """
        var_1: sigma_1 squared
        var_2: sigma_2 squared
        """
        # cov_1 = torch.diag_embed(var_1)
        # dist_1 = torch.distributions.MultivariateNormal(loc=mu_1, covariance_matrix=cov_1)
        # cov_2 = torch.diag_embed(var_2)
        # dist_2 = torch.distributions.MultivariateNormal(loc=mu_2, covariance_matrix=cov_2)

        # Add small epsilon to variance for numerical stability
        eps = 1e-6
        std_1 = torch.sqrt(var_1 + eps)
        std_2 = torch.sqrt(var_2 + eps)

        # Define independent univariate normals and reinterpret as multivariate
        dist_1 = torch.distributions.Independent(
            torch.distributions.Normal(loc=mu_1, scale=std_1),
            reinterpreted_batch_ndims=1)
        # Define independent univariate normals and reinterpret as multivariate
        dist_2 = torch.distributions.Independent(
            torch.distributions.Normal(loc=mu_2, scale=std_2),
            reinterpreted_batch_ndims=1)

        return torch.distributions.kl_divergence(dist_1, dist_2)

    def cluster_assignments(self, z):
        z = XData(z.detach().cpu().numpy())
        LP = self.model.calc_local_params(z)
        # Here, resp is a 2D array of size N x K.
        # Each entry resp[n, k] gives the probability
        # that data atom n is assigned to cluster k under
        # the posterior.
        resp = LP['resp']
        # To convert to hard assignments
        # Here, Z is a 1D array of size N, where entry Z[n] is an integer in the set {0, 1, 2, … K-1, K}.
        Z = resp.argmax(axis=1)
        return resp, Z

    # def sample_component(self, num_samples: int, component: int):
    #     """
    #     Samples from a dpmm cluster and return the corresponding
    #     image space map.
    #     :param num_samples: (Int) Number of samples
    #     :return: (Tensor)
    #     """
    #     mu = self.comp_mu[component]
    #     cov = torch.diag_embed(self.comp_var[component])
    #     dist = torch.distributions.MultivariateNormal(loc=mu,
    #                                                   covariance_matrix=cov)
    #     z = dist.sample_n(num_samples)
    #     return z

    # def sample_all(self, num_samples: int):
    #     """
    #     Sample a total of (roughly) num_samples samples from all cluster components
    #     """
    #     # E_proba_k=model.allocmodel.get_active_comp_probs()
    #     num_comps = len(self.comp_mu)     # number of active components
    #     latent_dim = len(self.comp_mu[0])     # dimension of the latent variable
    #     num_per_comp = int(num_samples/num_comps)
    #     z = torch.zeros(num_comps * num_per_comp, latent_dim)
    #     for k in range(0, num_comps):
    #         z[k * num_per_comp:(k + 1) * num_per_comp, :] = \
    #             self.sample_component(num_per_comp, k)
    #     return z


    def sample_component(self, num_samples: int, component: int):
        """
        Samples from a dpmm cluster and return the corresponding
        image space map.
        
        Args:
            num_samples: Number of samples
            component: Index of component to sample from
            
        Returns:
            Tensor of samples with shape (num_samples, latent_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        mu = self.comp_mu[component].to(device)
        var = self.comp_var[component].to(device)
        
        # # Create diagonal covariance matrix
        # cov = torch.diag_embed(var)
        
        # # Use new sample() API instead of deprecated sample_n()
        # dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)

        # Add small epsilon to variance for numerical stability
        eps = 1e-6
        std = torch.sqrt(var + eps)

        # Define independent univariate normals and reinterpret as multivariate
        dist = torch.distributions.Independent(
            torch.distributions.Normal(loc=mu, scale=std),
            reinterpreted_batch_ndims=1)

        z = dist.sample((num_samples,))  # Note the tuple shape
        
        return z

    def sample_all(self, num_samples: int):
        """
        Sample a total of (roughly) num_samples samples from all cluster components
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            Tensor of samples with shape (total_samples, latent_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        num_comps = len(self.comp_mu)       # Number of active components
        latent_dim = len(self.comp_mu[0])   # Dimension of latent variable
        
        if num_comps == 0:
            return torch.empty((0, latent_dim)).to(device)
        
        # Calculate samples per component (at least 1 per component)
        samples_per_comp = max(1, num_samples // num_comps)
        
        # Sample from each component
        samples = []
        for k in range(num_comps):
            samples.append(self.sample_component(samples_per_comp, k))
        
        # Concatenate all samples
        z = torch.cat(samples, dim=0)
        
        # If we got more samples than requested, randomly subsample
        if len(z) > num_samples:
            idx = torch.randperm(len(z))[:num_samples]
            z = z[idx]
        
        return z.to(device)  # Ensure output is on correct device
    
    def save_model_and_info_dict_for_eval(self):
        pass


    #################################################
    ############## For Evaluation ###############
    #################################################
    # def manage_latent_representation(self, z, env_idx, env_name_list:list, prefix:str='', save_info_dict:bool=False):
    #     '''
    #     calculate the latent z and their corresponding clusters, save the values for later evaluation
    #     saved value:
    #         z: latent encoding from VAE Encoder
    #         env_idx: corresponding env idx of each z (true label)
    #         env_name: corresponding env name of each z
    #         cluster_label: corresponding cluster No. that z should be assigned to
    #         cluster_param: corresponding cluster parameters (mu & var of diagGauss) of each z
    #     '''
        
    #     comps_mu_list = []
    #     comps_var_list = []
    #     # cluster label
    #     _, cluster_label = self.cluster_assignments(z)
    #     # get clusters param
    #     comp_mu = [self.model.obsModel.get_mean_for_comp(i) for i in np.arange(0, self.model.obsModel.K)]
    #     comp_var = [np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0) for i in np.arange(0, self.model.obsModel.K)]
    #     for i in cluster_label:
    #         comps_mu_list.append(comp_mu[i])
    #         comps_var_list.append(comp_var[i])
    #     # summarize data into dict
    #     data = dict(
    #         z=z.detach().cpu().numpy(),
    #         env_idx=env_idx.detach().cpu().numpy(),
    #         env_name=env_name_list,
    #         cluster_label=cluster_label,
    #         cluster_mu=comps_mu_list,
    #         cluster_var=comps_var_list,
    #     )
    #     # save the file
    #     np.savez(self.save_dir+'/data'+'/latent_samples_{}.npz'.format(prefix), **data)
    #     if save_info_dict and self.info_dict is not None:
    #         self.save_info_dict()


    # def load_model(self, abs_path, model_type:str='Best'):
    #     if os.path.exists(abs_path) and os.path.exists(abs_path+'/data/info_dict.npy'):

    #         self.info_dict = np.load(abs_path+'/data/info_dict.npy', allow_pickle=True)
    #         z_files = os.listdir(abs_path+'/data/')
    #         z_files = [x for x in z_files if x.startswith('latent_samples')]
    #         if z_files == []:
    #             print('no latent samples exists at {}'.format(abs_path+'/data/'))
    #             return

    #         z_files_split =  [x.split('_')[2] for x in z_files]
    #         max_index = -1
    #         max_value = -1
    #         for i, element in enumerate(z_files_split):
    #             if element.startswith('step'):
    #                 _, number = element.split('step')
    #                 number = int(number)
    #                 if number > max_value:
    #                     max_index = i
    #                     max_value = number
    #         z_file_name = z_files[max_index]
    #         z = np.load(abs_path+'/data/'+z_file_name)['z']
    #         z = XData(z)
    #         self.model, self.info_dict = bnpy.run(z, 'DPMixtureModel', 'DiagGauss', 'memoVB',
    #                                               output_path=os.path.join(self.save_dir,
    #                                                                        str(next(self.iterator))),
    #                                               initname=self.info_dict['task_output_path'],
    #                                               K=self.info_dict['K_history'][-1], gamma0=self.gamma0,
    #                                               sF=self.sF, ECovMat='eye',
    #                                               moves='birth,merge', nBatch=5, nLap=self.num_lap,
    #                                               **dict(
    #                                                   sum(map(list, [self.birth_kwargs.items(),
    #                                                                  self.merge_kwargs.items()]), []))
    #                                               )
    #         self.calc_cluster_component_params()
    #         print('load {} bnp_model from {}'.format(model_type, abs_path))
    #     else:
    #         print(f'invalid bnp_modle load path {abs_path}')

    
    
    # def save_info_dict(self):
    #     info = dict(
    #         task_output_path = self.info_dict['task_output_path'],
    #         K_history=[self.info_dict['K_history'][-1]],
    #     )
    #     np.save(self.save_dir+'/data'+'/info_dict.npy', info)

    # def save_comps_parameters(self):
    #     '''
    #     save the model param in form of *npy
    #     '''
    #     self.comp_mu = [self.model.obsModel.get_mean_for_comp(i)
    #                     for i in np.arange(0, self.model.obsModel.K)]
    #     self.comp_var = [np.sum(self.model.obsModel.get_covar_mat_for_comp(i), axis=0) # save diag value in a 1-dim array
    #                      for i in np.arange(0, self.model.obsModel.K)]
    #     data = dict(
    #         comp_mu=self.comp_mu,
    #         comp_var=self.comp_var
    #     )
    #     np.save(self.save_dir+'/data'+'/comp_params.npy', data)

    def save_model(self, save_path):
        # save info_dict, parameters (gamma0, sF, etc.). model already been saved into .mat file under results/exp-time/dpmm_model
                # Create directory if needed
        os.makedirs(save_path, exist_ok=True)
        
        # Save cluster tracking state
        state = {
            'cluster_history': self.cluster_history,
            'current_clusters': self.current_clusters,
            'next_available_id': self.next_available_id,
            'components': self.components,
            'gamma0': self.gamma0,
            'num_lap': self.num_lap,
            'sF': self.sF,
            'birth_kwargs': self.birth_kwargs,
            'merge_kwargs': self.merge_kwargs,
            # 'save_dir': self.save_dir
        }
        torch.save(state, os.path.join(save_path, 'bnp_model_state.pth'))

        # Save info_dict
        if self.info_dict:
            sanitized_info = self._sanitize_for_json(self.info_dict)
            with open(os.path.join(save_path, 'info_dict.json'), 'w') as f:
                json.dump(sanitized_info, f, indent=2, ensure_ascii=False)
        
        print(f'dpmm saved to {save_path}.')

    def load_model(self, load_path):
        # Load cluster tracking state
        state_path = os.path.join(load_path, 'bnp_model_state.pth')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.cluster_history = state['cluster_history']
            self.current_clusters = state['current_clusters']
            self.next_available_id = state['next_available_id']
            self.components = state['components']
            self.gamma0 = state['gamma0']
            self.num_lap = state['num_lap']
            self.sF = state['sF']
            self.birth_kwargs = state['birth_kwargs']
            self.merge_kwargs = state['merge_kwargs']
            # self.save_dir = new_save_dir or state['save_dir']

        # Load info_dict
        info_path = os.path.join(load_path, 'info_dict.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.info_dict = json.load(f)
        
        # Load bnpy model
        if os.path.exists(load_path):
            try:
                self.model = load_model_at_prefix(self.info_dict['task_output_path'], prefix='Best')
            except Exception as e:
                print(f"Error loading bnpy model: {e}")
                self.model = None

        print(f'dpmm loaded from {load_path}.')
    
    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects for JSON serialization"""
        # Handle dictionaries - convert keys and sanitize values
        if isinstance(obj, Mapping):
            sanitized = {}
            for k, v in obj.items():
                # Convert NumPy/non-string keys to native types
                if isinstance(k, (np.integer, np.int64, np.int32)):
                    key = int(k)
                elif isinstance(k, (np.floating, np.float64, np.float32)):
                    key = float(k)
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    key = str(k)
                else:
                    key = k
                    
                sanitized[key] = self._sanitize_for_json(v)
            return sanitized
        
        # Handle lists/tuples - sanitize each element
        elif isinstance(obj, Iterable) and not isinstance(obj, str):
            return [self._sanitize_for_json(item) for item in obj]
        
        # Handle NumPy types
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle other non-serializable objects
        elif not isinstance(obj, (str, int, float, bool, type(None))):
            return str(obj)  # Convert to descriptive string
        
        # Base types pass through
        return obj




