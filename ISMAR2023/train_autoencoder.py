import os
import os.path as osp
import gc
import time
import datetime 
import numpy as np
from tqdm import tqdm 

import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from rotation import rotation_matrix_to_angle_axis, rad2deg
from model import ConvolutionalAutoEncoder

class Dataset_AutoEncoder(Dataset):
    def __init__(self, handedness, train_data, repertories, clip_length, gpu_id):
        self.gpu_id = gpu_id
        self.handedness = handedness
        self.train_data = train_data
        if train_data:
            self.folder_name = 'data/train(autoencoder)'+str(gpu_id)
        else:
            self.folder_name = 'data/valid(autoencoder)'+str(gpu_id)          
        self.count = 0
        self.paths = []      
        
        self.repertories = repertories
        self.L= clip_length
        self.raw_data_dir = 'data(raw)/'+'train_autoencoder/'+self.handedness+'/'
            
    def split(self):
        if self.train_data == True:
            for repertorie in self.repertories:
                folder_raw = self.raw_data_dir + repertorie['repertorie'] + '/' + 'raw'
                folder_split = self.raw_data_dir  + repertorie['repertorie'] + '/' + 'split'
                for f in os.listdir(folder_split):    
                    os.remove(os.path.join(folder_split, f))                    
                input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
                sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
                for i in range(1, int(np.max(sequences))+1):
                    np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')        
        else:
            data_dir_list = []
            if self.handedness == 'right':
                data_dir_list = ['data(raw)/valid_autoencoder/161/', 'data(raw)/valid_autoencoder/172/', 'data(raw)/valid_autoencoder/180/']
            elif self.handedness == 'left':
                data_dir_list = ['data(raw)/valid_autoencoder/173/', 'data(raw)/valid_autoencoder/174/']
                
            for data_dir in data_dir_list:
                for repertorie in self.repertories:
                    folder_raw = data_dir + repertorie['repertorie'] + '/' + 'raw'
                    folder_split = data_dir  + repertorie['repertorie'] + '/' + 'split'
                    for f in os.listdir(folder_split):    
                        os.remove(os.path.join(folder_split, f))                    
                    input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
                    sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
                    for i in range(1, int(np.max(sequences))+1):
                        np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')   
    
    def delete(self):        
        for f in os.listdir(self.folder_name):    
            os.remove(os.path.join(self.folder_name, f))  
                
    def generate(self):
        if self.train_data == True:
            for repertorie in self.repertories:
                folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                        
                for i in range(len(os.listdir(folder_split))):                
                    data = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                    data = torch.tensor(data, dtype=torch.float32)                 
                    data = data[:,18:36]
                    np.save(osp.join(self.folder_name, str(self.count)), data[-self.L:,:].numpy())
                    self.count = self.count + 1
                    
                    data = torch.transpose(data.unfold(0, self.L, int(self.L/2)), 1, 2)
                    for k in range(data.shape[0]):
                        np.save(osp.join(self.folder_name, str(self.count)), data[k].numpy())
                        self.count = self.count + 1                  
        else:
            data_dir_list = []
            if self.handedness == 'right':
                data_dir_list = ['data(raw)/valid_autoencoder/161/', 'data(raw)/valid_autoencoder/172/', 'data(raw)/valid_autoencoder/180/']
            elif self.handedness == 'left':
                data_dir_list = ['data(raw)/valid_autoencoder/173/', 'data(raw)/valid_autoencoder/174/'] 
                
            for data_dir in data_dir_list:
                for repertorie in self.repertories:
                    folder_split = data_dir + repertorie['repertorie'] + '/' + 'split'                        
                    for i in range(len(os.listdir(folder_split))):                
                        data = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                        data = torch.tensor(data, dtype=torch.float32)               
                        data = data[:,15:33]
                        np.save(osp.join(self.folder_name, str(self.count)), data[-self.L:,:].numpy())
                        self.count = self.count + 1
                        
                        data = torch.transpose(data.unfold(0, self.L, int(self.L/2)), 1, 2)
                        for k in range(data.shape[0]):
                            np.save(osp.join(self.folder_name, str(self.count)), data[k].numpy())
                            self.count = self.count + 1                
        self.paths = [osp.join(self.folder_name, str(i)+'.npy') for i in range(self.count)] 
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        X = torch.from_numpy(np.load(self.paths[idx]))
        return {'X': X}

def sixd2matrot(pose_6d):
    rot_vec_1 = F.normalize(pose_6d[:,:,:,0:3], dim=-1) # forward
    rot_vec_2 = F.normalize(pose_6d[:,:,:,3:6], dim=-1) # up
    rot_vec_3 = torch.cross(rot_vec_2, rot_vec_1) # right
    pose_matrot = torch.stack([rot_vec_3,rot_vec_2,rot_vec_1], dim=-1) # (right,up,forward)
    return pose_matrot

def matrot2aa(pose_matrot):
    homogen_matrot = F.pad(pose_matrot, [0, 1])
    pose = rotation_matrix_to_angle_axis(homogen_matrot) # Nx3x3 -> Nx3
    return pose

def sixd2aa(pose_6d):
    N = pose_6d.shape[0]
    L = pose_6d.shape[1]
    J = int(pose_6d.shape[2]/6)
    pose_6d = torch.reshape(pose_6d, (N,L,J,6))            
    pose_matrot = sixd2matrot(pose_6d) # (N, L, J, 6) -> (N, L, J, 3, 3)
    pose_aa = torch.reshape(matrot2aa(pose_matrot.reshape(-1,3,3)), (N,L,J,3)) # (N, L, J, 3, 3) -> (N, L, J, 3) -> (N, L, J, 3)
    return rad2deg(pose_aa)    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--handedness', default='right', help='Handedness')
parser.add_argument('--numEpoch', type=int, default=600, help='epoch size for training')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size for training')
parser.add_argument('--hiddenSize', type=int, default=32, help='Hidden Size')
parser.add_argument('--kernelSize', type=int, default=15, help='Hidden Size')
parser.add_argument('--L', type=int, default=30, help='Clip Length')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache() 
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')   
    
    # Training setup.
    GPU_ID = args.gpu_id
    SHUFFLE = args.shuffle
    HANDEDNESS = args.handedness
    NUM_EPOCHS = args.numEpoch
    BATCH_SIZE = args.batchSize    
    L = args.L
    LR = args.lr  # learning rate
            
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')        

    INPUT_SIZE=18
    OUTPUT_SIZE=18
    HIDDEN_SIZE=args.hiddenSize
    KERNEL_SIZE=args.kernelSize
    model = ConvolutionalAutoEncoder(INPUT_SIZE, HIDDEN_SIZE, KERNEL_SIZE)
    model = model.to(device)
    
    save_dir = "AutoEncoder,"+str(HANDEDNESS)+","+str(HIDDEN_SIZE)+","+str(KERNEL_SIZE)+","+str(NUM_EPOCHS)+","+str(BATCH_SIZE)+","+str(LR)+","+nowDatetime
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                   {'repertorie': 'Pointing at two targets', 'num': 210},
                   {'repertorie': 'Pointing at single target with gaze shift', 'num': 180},
                   {'repertorie': 'Pointing at single target with explanation', 'num': 126},
                   {'repertorie': 'Transition (Pointing)', 'num': 30},
                   {'repertorie': 'Touching single target', 'num': 30},
                   {'repertorie': 'Touching single target with gaze shift', 'num': 210},
                   {'repertorie': 'Touching single target with explanation', 'num': 126},
                   {'repertorie': 'Touching two targets', 'num': 210},
                   {'repertorie': 'Transition (Touching)', 'num': 27}]
    
    train_data_set = Dataset_AutoEncoder(HANDEDNESS, True, REPERTORIES, L, GPU_ID) 
    train_data_set.delete()
    train_data_set.split()
    train_data_set.generate()    
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=16, pin_memory=True, drop_last=True)
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), len(train_data_set) // BATCH_SIZE))

    valid_data_set = Dataset_AutoEncoder(HANDEDNESS, False, REPERTORIES, L, GPU_ID) 
    valid_data_set.delete()
    valid_data_set.split()
    valid_data_set.generate()
    valid_data_loader = DataLoader(valid_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    print('# of valid example: {}, # of batches {}'.format(len(valid_data_set), len(valid_data_set) // BATCH_SIZE)) 
            
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.999)
    train_start = time.time()
    for epoch in range(1, NUM_EPOCHS+1): 
        # Train
        epoch_start = time.time()                
        model.train()    
        epoch_loss = 0    
        epoch_pe = 0
        epoch_re = 0
        pbar = tqdm(train_data_loader)
        for iter, X in enumerate(pbar):            
            X = X['X'].to(device)            
            X_hat = model(X, False)
            loss = F.l1_loss(X_hat, X)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()                            
            pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(train_data_loader)))
            pbar.set_postfix({"loss":((loss))})  
            epoch_loss += loss.item()

            # position error
            X_pos_hat = torch.reshape(torch.cat((X_hat[:,:,6:9], X_hat[:,:,15:18]), -1), (BATCH_SIZE, L, 2, 3))  
            X_pos = torch.reshape(torch.cat((X[:,:,6:9], X[:,:,15:18]), -1), (BATCH_SIZE, L, 2, 3))  
            pe = torch.mean(torch.sqrt(torch.sum(torch.square(X_pos-X_pos_hat),axis=-1)))*100
            epoch_pe += pe
            # rotation error
            X_rot_hat = sixd2aa(torch.cat((X_hat[:,:,0:6], X_hat[:,:,9:15]), -1))
            X_rot = sixd2aa(torch.cat((X[:,:,0:6], X[:,:,9:15]), -1))
            re = torch.mean(torch.sum(torch.absolute(X_rot-X_rot_hat),axis=-1))
            epoch_re += re
        
        scheduler.step()
        train_time = time.time() - epoch_start
        epoch_loss /= (len(train_data_loader))
        epoch_pe /= (len(train_data_loader))
        epoch_re /= (len(train_data_loader))
        log = "Train Time = %.1f, [Epoch %d/%d] [Train Loss: %.7f] [Position Error: %.3f] [Rotation Error: %.3f]" % (train_time, epoch, NUM_EPOCHS, epoch_loss, epoch_pe, epoch_re)
        print(log)     
        
        # Valid
        epoch_start = time.time()       
        model.eval()
        epoch_loss = 0
        epoch_pe = 0
        epoch_re = 0
        pbar = tqdm(valid_data_loader)
        with torch.no_grad():
            for iter, X in enumerate(pbar):            
                X = X['X'].to(device)            
                X_hat = model(X, False)
                loss = F.l1_loss(X_hat, X)                         
                pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(valid_data_loader)))
                pbar.set_postfix({"loss":((loss))})  
                epoch_loss += loss.item()
                
                # position error
                X_pos_hat = torch.reshape(torch.cat((X_hat[:,:,6:9], X_hat[:,:,15:18]), -1), (BATCH_SIZE, L, 2, 3))  
                X_pos = torch.reshape(torch.cat((X[:,:,6:9], X[:,:,15:18]), -1), (BATCH_SIZE, L, 2, 3))  
                pe = torch.mean(torch.sqrt(torch.sum(torch.square(X_pos-X_pos_hat),axis=-1)))*100
                epoch_pe += pe
                # rotation error
                X_rot_hat = sixd2aa(torch.cat((X_hat[:,:,0:6], X_hat[:,:,9:15]), -1))
                X_rot = sixd2aa(torch.cat((X[:,:,0:6], X[:,:,9:15]), -1))
                re = torch.mean(torch.sum(torch.absolute(X_rot-X_rot_hat),axis=-1))
                epoch_re += re       
                
        valid_time = time.time() - epoch_start
        epoch_loss /= (len(valid_data_loader))
        epoch_pe /= (len(valid_data_loader))
        epoch_re /= (len(valid_data_loader))
        log = "Valid Time = %.1f, [Epoch %d/%d] [Valid Loss: %.7f] [Position Error: %.3f] [Rotation Error: %.3f]" % (valid_time, epoch, NUM_EPOCHS, epoch_loss, epoch_pe, epoch_re)
        print(log)                           
        # Model Save
        torch.save(model.state_dict(), os.path.join(save_dir, "AutoEncoder"+str(epoch)+".pt"))
        
    print("Total Time= %.2f" %(time.time()-train_start))      