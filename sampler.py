# Copyright 2023 Massimiliano Patacchiola
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np

class Sampler():
    def __init__(self, env_name, data_list, portion=(0.0,1.0), tot_steps=1000):
        """ Sampler class used to obtain offline trajectories.
        
            env_name: the environment used. One of Hopper-v3, HalfCheetah-v3, Walker2d-v3
            data_list: list of npz files addresses containing the offline trajectories
            portion: tuple containing two values in the range [0.0, 1.0] representing
                the portion of the trajectories used by the sampler. Useful to define train/test splits.
                e.g. a train sampler takes the portion [0.0, 0.5], the test sampler [0.5, 1.0].
            tot_steps: (not used) maximum length of the trajectories, keep it to 1000.
            
            Usage example:
                from sampler import Sampler
                files_list = ['./dataset/sac-hopper-massdec_25_leggeom.npz', './dataset/sac-hopper-massdec_50_leggeom.npz',
                              './dataset/sac-hopper-massdec_25_footgeom.npz', './dataset/sac-hopper-massdec_50_footgeom.npz']
                train_sampler = Sampler(env_name='Hopper-v3', data_list=files_list, portion=(0.0,0.25))
                x, y = train_sampler.sample(tot_shots=5, idx=datset_idx)
        """
        self.tot_steps = tot_steps
        assert(portion[0] < portion[1])
        if(len(data_list) == 0):
            print(f"[ERROR] data_list is empty!")
            quit()            
        
        if(env_name == "Hopper-v3"):
            tot_inputs = 11; tot_outputs = 3
        elif(env_name == "HalfCheetah-v3"):
            tot_inputs = 17; tot_outputs = 6
        elif(env_name == "Walker2d-v3"):
            tot_inputs = 17; tot_outputs = 6
        else:
            print(f"[ERROR] The dataset {env_name} is not supported!")
            quit()
    
        env_name_raw = env_name.lower().replace("-","").replace("v3","")    
        self.dataset_list = list()
        self.tot_data = 0
        for filename in data_list:
            dataset = np.load(filename)["data"]
            start_portion = int(dataset.shape[0] * portion[0])
            stop_portion = int(dataset.shape[0] * portion[1])
            x = torch.from_numpy(dataset[start_portion:stop_portion,0:tot_inputs]).to(torch.float32)
            y = torch.from_numpy(dataset[start_portion:stop_portion,tot_inputs:tot_inputs+tot_outputs]).to(torch.float32)
            self.tot_data += x.shape[0]
            self.dataset_list.append((x,y))

    def __len__(self):
        return len(self.dataset_list)

    def sample(self, tot_shots, idx=None, replace=False):
        """ Sample a trajectory.
        
            tot_shots: the number of trajectories to sample.
            idx: the index of the data-file to sample, default None (random index)
            replace: if True sampling is with replacement (could sample the same index)
            
            return: two torch tensors containing input/outputs (states/actions)
        """
        #1. Randomly select a data-file if idx is not given
        if(idx is None):
            idx = np.random.randint(len(self.dataset_list))
        # Pick the dataset from the index
        dataset = self.dataset_list[idx]
        #2. Randomly select tot_shots trajectories
        x_list = list()
        y_list = list()
        tot_pairs = int(dataset[0].shape[0])
        start_indices = np.arange(0, tot_pairs, self.tot_steps)
        sampled_start_indices = np.random.choice(start_indices, size=tot_shots, replace=replace)
        for start_index in sampled_start_indices:
            stop_index = start_index + self.tot_steps
            x_list.append(dataset[0][start_index:stop_index,:].unsqueeze(0))
            y_list.append(dataset[1][start_index:stop_index,:].unsqueeze(0))
        #3. Concatenate and return
        return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0) # -> [tot_shots, tot_steps, dim]
