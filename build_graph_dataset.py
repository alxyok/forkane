# MIT License

# Copyright (c) 2021 alxyok

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import config
from metaflow import FlowSpec, step, current
import os
import os.path as osp

class BuildGraphDatasetFlow(FlowSpec):
    
    @step
    def start(self):
        """
        The start step:
        1) Read the list of files to build the dataset from.
        2) Build the connectivity matrix shared by all data row.
        """
        
        import torch_geometric as pyg
        import utils
        import yaml
        
        os.makedirs(config.processed_dir, exist_ok=True)
        
        if config.purge:
            for file in os.listdir(config.processed_dir):
                try:
                    os.remove(os.path.join(config.processed_dir, file))
                except:
                    pass
        
        self.x_dim = 65
        self.y_dim = 33
        self.z_dim = 33
        
        self.grid_shape = (self.x_dim, self.y_dim, self.z_dim)
        self.edge_index = utils.grid_3d_connectivity_matrix(self.grid_shape)
        
        with open(osp.join(config.root_path, "filenames.yaml"), "r") as f:
            self.filenames = yaml.safe_load(f)
            
        self.next(self.build_graph, foreach="filenames")
                  
                  
    @step
    def build_graph(self):
        """
        Build and save the graphs, in PyTorch format, in parallel by branching the flow. If data augmentation is enabled, this step will generated random crops (of size kernel_size) of the original tensor for a random number of times. If disabled, it will return a single graph containing all nodes in the original data.
        """
        
        import h5py
        import numpy as np
        import torch
        import torch_geometric as pyg
        
        def build_crop(data, idx=0):
            
            i_size, j_size, k_size = data[0].shape
        
            coordinates = list()
            for k in range(k_size):
                for j in range(j_size):
                    for i in range(i_size):
                        coordinates.append([float(i), float(j), float(k)])
            coordinates = np.stack((coordinates))
            
            x0, x1, y = data
            x = np.hstack((x0.reshape((-1, 1)), x1.reshape((-1, 1))))
            y = np.reshape(y, (-1, 1))
            
            if config.add_coordinates:
                x = np.hstack((x, coordinates))
                y = np.hstack((y, coordinates))
            
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)

            index = torch.tensor(self.edge_index, dtype=torch.long)

            graph = pyg.data.Data(x=x, edge_index=index, y=y)

            processed_path = osp.join(config.processed_dir, f'data-{current.task_id}-{idx}.pt')
            torch.save(graph, processed_path)
        
        
        raw_path = osp.join(config.raw_dir, 'npy', f'{self.input}.npy')
        
        grid_shape = (3,) + self.grid_shape
        data = np.memmap(raw_path, dtype='float32', mode='r', shape=grid_shape)
        x0, x1, y = data[0, ...], data[1, ...], data[2, ...]
        
        if config.augment_data:
            min_ = config.min_num_crops
            max_ = config.max_num_crops
            for random_step in np.arange(np.random.randint(min_, max_)):
                
                xi = np.random.randint(0, self.x_dim - config.kernel_size)
                yi = np.random.randint(0, self.y_dim - config.kernel_size)
                zi = np.random.randint(0, self.z_dim - config.kernel_size)

                x0_ = x0[xi:xi + config.kernel_size, yi:yi + config.kernel_size, zi:zi + config.kernel_size]
                x1_ = x1[xi:xi + config.kernel_size, yi:yi + config.kernel_size, zi:zi + config.kernel_size]
                y_ = y[xi:xi + config.kernel_size, yi:yi + config.kernel_size, zi:zi + config.kernel_size]
                
                build_crop((x0_, x1_, y_), idx=random_step)
        
        else:
            build_crop((x0, x1, y))
        
        self.next(self.join)
        
        
    @step
    def join(self, inputs):
        """
        Join the parallel branches.
        """
        
        self.next(self.end)
        
        
    @step
    def end(self):
        """
        End the flow.
        """
        
        pass
            
if __name__ == '__main__':
    
    BuildGraphDatasetFlow()