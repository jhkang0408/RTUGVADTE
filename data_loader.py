import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Dataset(Dataset):
    def __init__(self, dtype=None, network="", directory_id=None, **kwargs):
        self.folder_name = 'data/train'+'_'+network+'_'+str(directory_id)         
        self.sequences = len(os.listdir(self.folder_name))        
        self.paths = [self.folder_name+'/'+str(i)+'.npy' for i in range(self.sequences)]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        data = torch.from_numpy(np.load(self.paths[idx]))        
        return {'data': data}

if __name__ == '__main__':
    batch_size = 16
    shuffle = True
    train_data_set = Dataset(gpu_id=0)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    train_set_size = len(train_data_set)
    n_batches_train = train_set_size // batch_size
    
    print('# of training example: {}, # of batches {}'.format(train_set_size, n_batches_train))
    print('# of path='+str(len(train_data_set.paths)))
    
    print((train_data_set.__getitem__(0)['data']).shape)