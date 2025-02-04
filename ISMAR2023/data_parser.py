import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset(Dataset):
    def __init__(self, dtype=None, gpu_id=None, **kwargs):
        folder_name = 'data/train'+str(gpu_id)            
        self.sequences = len(os.listdir(osp.join(folder_name, 'Local')))        
        self.local_paths = [osp.join(folder_name, 'Local/')+str(i)+'.npy' for i in range(self.sequences)]
        self.remote_paths = [osp.join(folder_name, 'Remote/')+str(i)+'.npy' for i in range(self.sequences)]
        
    def __len__(self):
        return len(self.local_paths)

    def __getitem__(self, idx):        
        local = torch.from_numpy(np.load(self.local_paths[idx]))
        remote = torch.from_numpy(np.load(self.remote_paths[idx]))        
        return {'local': local, 'remote': remote}        

if __name__ == '__main__':
    batch_size = 16
    shuffle = True
    train_data_set = Dataset(train_data=False)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    train_set_size = len(train_data_set)
    n_batches_train = train_set_size // batch_size
    
    print('# of training example: {}, # of batches {}'.format(train_set_size, n_batches_train))
    print('# of local path='+str(len(train_data_set.local_paths)))
    print('# of remote path='+str(len(train_data_set.remote_paths)))
    
    #region Tensor Example
    '''
    x = torch.rand(100, 93)
    x = x.unfold(0, 32, 16)
    x = torch.transpose(x, 1, 2)
    x = x.reshape(-1, x.size(2))
    print(x.shape)
    
    x = torch.arange(1.0, 101)
    x = x.unfold(0, 32, 16)
    x = x.reshape(-1)
    print(x.shape)  
    '''
    #endregion    