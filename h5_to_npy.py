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
from metaflow import FlowSpec, step
import os.path as osp

class ConvertToNumpyFlow(FlowSpec):
    
    @step
    def start(self):
        """
        Read the list of files to convert.
        """
        
        import yaml
        
        with open(osp.join(config.root_path, "filenames.yaml"), "r") as f:
            self.filenames = yaml.safe_load(f)
            
        self.next(self.convert, foreach="filenames")
                  
                  
    @step
    def convert(self):
        """
        Convert HDF5 files to Numpy files.
        """
        
        import h5py
        import numpy as np
        
        h5_path = osp.join(config.raw_dir, 'h5', f'{self.input}.h5')
        with h5py.File(h5_path, 'r') as file:
            x0 = file["/grad_filt_8"][:]
            x1 = file["/filt_grad_8"][:]
            y = file["/filt_8"][:]
        
        data = np.stack((x0, x1, y))
        
        filebase = self.input.split('.')[0]
        np_path = osp.join(config.raw_dir, 'npy', f'{filebase}.npy')
        np.save(np_path, data)
        
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
    
    ConvertToNumpyFlow()