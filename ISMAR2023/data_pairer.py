import os
import time
import torch
import os.path as osp
import numpy as np
import itertools as it
from scipy import interpolate

class data_pairer:
    def __init__(self, handedness, repertories, clip_length, window_size, gpu_id):
        self.repertories = repertories
        self.L= clip_length
        self.W = window_size
        self.raw_data_dir = 'data(raw)/'+'train/'+handedness+'/'  
        self.paired_motion_dir = 'data/train'+str(gpu_id)+'/'             
        self.count = 0
        self.count_easy = 0
        
    def split(self):
        for repertorie in self.repertories:
            folder_raw = self.raw_data_dir + repertorie['repertorie'] + '/' + 'raw'
            folder_split = self.raw_data_dir  + repertorie['repertorie'] + '/' + 'split'
            for f in os.listdir(folder_split):
                os.remove(os.path.join(folder_split, f))
            input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
            sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
            for i in range(1, int(np.max(sequences))+1):
                np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')        
    
    def delete(self):
        for repertorie in self.repertories:
            folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'     
            for f in os.listdir(folder_split):
                os.remove(os.path.join(folder_split, f))              
        for d in [self.paired_motion_dir+'Local', self.paired_motion_dir+'Remote']:    
            for f in os.listdir(d):    
                os.remove(os.path.join(d, f))     
                 
    def __len__(self):        
        print('# of Local Path='+str(len(os.listdir(self.paired_motion_dir+'/Local'))))
        print('# of Remote Path='+str(len(os.listdir(self.paired_motion_dir+'/Remote'))))
        
    def generate_pair_order(self):              
        idx=np.zeros((210*210,2))
        i=0
        for combi in it.product(range(1,210+1),repeat=2):
            idx[i,:]=np.asarray(combi)
            i+=1
        np.random.shuffle(idx)
        np.save(self.paired_motion_dir+'Hard(Pointing at two targets)', idx) 
        
        idx=np.zeros((210*210,2))
        i=0
        for combi in it.product(range(1,210+1),repeat=2):
            idx[i,:]=np.asarray(combi)
            i+=1
        np.random.shuffle(idx)
        np.save(self.paired_motion_dir+'Hard(Touching two targets)', idx)        
        
    def interpolate(self, folder_split_L, folder_split_R, idx_L, idx_R, Start, End, Scale):
        local = np.loadtxt(osp.join(folder_split_L, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(folder_split_R, 'Data'+str(idx_R)+'.txt')).astype(np.float32)
        
        key_idx = local.shape[1]-1
        local_keys = np.append(np.where(local[:,key_idx]==1), local.shape[0]-1)
        remote_keys = np.append(np.where(remote[:,key_idx]==1), remote.shape[0]-1)                
        local_re = np.empty((0, local.shape[1]), np.float32)
        remote_re = np.empty((0, local.shape[1]), np.float32)
        
        for k in range(Start, End-1): # Phase
            local_transition_len = local_keys[k+1]-local_keys[k]
            remote_transition_len = remote_keys[k+1]-remote_keys[k]
            
            if local_transition_len>remote_transition_len:
                # upsample remote
                x = np.linspace(0, 1, remote_transition_len)
                y = remote[remote_keys[k]:remote_keys[k+1], :]
                f = interpolate.PchipInterpolator(x, y, axis=0)   
                xnew = np.linspace(0, 1, local_transition_len)        
                ynew = f(xnew) # upsampled remote
                local_re = np.concatenate((local_re, local[local_keys[k]:local_keys[k+1], :]), axis=0) 
                remote_re = np.concatenate((remote_re, ynew), axis=0)
            else:
                # upsample local
                x = np.linspace(0, 1, local_transition_len)
                y = local[local_keys[k]:local_keys[k+1], :] 
                f = interpolate.PchipInterpolator(x, y, axis=0)   
                xnew = np.linspace(0, 1, remote_transition_len)
                ynew = f(xnew) # upsampled locala
                local_re = np.concatenate((local_re, ynew), axis=0) 
                remote_re = np.concatenate((remote_re, remote[remote_keys[k]:remote_keys[k+1], :]), axis=0)             
        ############################################################################################################
        local_re = torch.tensor(local_re, dtype=torch.float32)        
        local_re[:,14] = Scale * local_re[:,14]
        local_re = local_re[:,0:15]
        
        remote_re = torch.tensor(remote_re, dtype=torch.float32)         
        remote_re[:,14] = Scale * remote_re[:,14]      
        remote_re[:,21:24] = Scale * remote_re[:,21:24]
        remote_re[:,30:36] = Scale * remote_re[:,30:36]
        remote_re = torch.cat((remote_re[:,0:15],
                               torch.unsqueeze(torch.tensor(Scale).repeat(remote_re.shape[0]), -1), 
                               remote_re[:,15:36]), -1)
        ############################################################################################################
        # Save
        np.save(osp.join(self.paired_motion_dir, 'Local/'+str(self.count)), local_re[-self.L:,:].numpy())
        np.save(osp.join(self.paired_motion_dir, 'Remote/'+str(self.count)), remote_re[-self.L:,:].numpy())        
        self.count = self.count + 1  
        
        local_re = torch.transpose(local_re.unfold(0, self.L, self.L-self.W), 1, 2)
        remote_re = torch.transpose(remote_re.unfold(0, self.L, self.L-self.W), 1, 2)
        for k in range(local_re.shape[0]):
            np.save(osp.join(self.paired_motion_dir, 'Local/'+str(self.count)), local_re[k].numpy())
            np.save(osp.join(self.paired_motion_dir, 'Remote/'+str(self.count)), remote_re[k].numpy())
            self.count = self.count + 1
    
    def interpolate_evaluation_DIP(self, folder_split_L, folder_split_R, idx_L, idx_R):
        local = np.loadtxt(osp.join(folder_split_L, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(folder_split_R, 'Data'+str(idx_R)+'.txt')).astype(np.float32)
        
        key_idx = local.shape[1]-1
        local_keys = np.append(np.where(local[:,key_idx]==1), local.shape[0]-1)
        remote_keys = np.append(np.where(remote[:,key_idx]==1), remote.shape[0]-1)                
        local_re = np.empty((0, local.shape[1]), np.float32)
        remote_re = np.empty((0, local.shape[1]), np.float32)
            
        key_DIP = 0
        for k in range(local_keys.shape[0]-1): # Phase
            local_transition_len = local_keys[k+1]-local_keys[k]
            remote_transition_len = remote_keys[k+1]-remote_keys[k]
            
            if local_transition_len>remote_transition_len:
                # upsample remote
                x = np.linspace(0, 1, remote_transition_len)
                y = remote[remote_keys[k]:remote_keys[k+1], :]
                f = interpolate.PchipInterpolator(x, y, axis=0)
                xnew = np.linspace(0, 1, local_transition_len)        
                ynew = f(xnew) # upsampled remote
                local_re = np.concatenate((local_re, local[local_keys[k]:local_keys[k+1], :]), axis=0) 
                remote_re = np.concatenate((remote_re, ynew), axis=0)
                if k==0:
                    key_DIP=local_transition_len
            else:
                # upsample local
                x = np.linspace(0, 1, local_transition_len)
                y = local[local_keys[k]:local_keys[k+1], :] 
                f = interpolate.PchipInterpolator(x, y, axis=0)   
                xnew = np.linspace(0, 1, remote_transition_len)
                ynew = f(xnew) # upsampled locala
                local_re = np.concatenate((local_re, ynew), axis=0) 
                remote_re = np.concatenate((remote_re, remote[remote_keys[k]:remote_keys[k+1], :]), axis=0)                       
                if k==0:
                    key_DIP=remote_transition_len                
        if local_re.shape != remote_re.shape:
            print("Error Occured in Pairing") 
        
        local_re = torch.tensor(local_re, dtype=torch.float32)
        remote_re = torch.tensor(remote_re, dtype=torch.float32)   
        return local_re[key_DIP-(self.W+1)+10:key_DIP+10,:], remote_re[key_DIP-(self.W+1)+10:key_DIP+10,:]      
    
    def interpolate_evaluation_MN(self, folder_split_L, folder_split_R, idx_L, idx_R):
        local = np.loadtxt(osp.join(folder_split_L, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(folder_split_R, 'Data'+str(idx_R)+'.txt')).astype(np.float32)
        
        key_idx = local.shape[1]-1
        local_keys = np.append(np.where(local[:,key_idx]==1), local.shape[0]-1)
        remote_keys = np.append(np.where(remote[:,key_idx]==1), remote.shape[0]-1)                
        local_re = np.empty((0, local.shape[1]), np.float32)
        remote_re = np.empty((0, local.shape[1]), np.float32)
        
        for k in range(local_keys.shape[0]-1): # Phase
            local_transition_len = local_keys[k+1]-local_keys[k]
            remote_transition_len = remote_keys[k+1]-remote_keys[k]
            
            if local_transition_len>remote_transition_len:
                # upsample remote
                x = np.linspace(0, 1, remote_transition_len)
                y = remote[remote_keys[k]:remote_keys[k+1], :]
                f = interpolate.PchipInterpolator(x, y, axis=0)   
                xnew = np.linspace(0, 1, local_transition_len)        
                ynew = f(xnew) # upsampled remote
                local_re = np.concatenate((local_re, local[local_keys[k]:local_keys[k+1], :]), axis=0) 
                remote_re = np.concatenate((remote_re, ynew), axis=0)
            else:
                # upsample local
                x = np.linspace(0, 1, local_transition_len)
                y = local[local_keys[k]:local_keys[k+1], :] 
                f = interpolate.PchipInterpolator(x, y, axis=0)   
                xnew = np.linspace(0, 1, remote_transition_len)
                ynew = f(xnew) # upsampled locala
                local_re = np.concatenate((local_re, ynew), axis=0) 
                remote_re = np.concatenate((remote_re, remote[remote_keys[k]:remote_keys[k+1], :]), axis=0)                       
        if local_re.shape != remote_re.shape:
            print("Error Occured in Pairing")
        
        local_re = torch.tensor(local_re, dtype=torch.float32)
        remote_re = torch.tensor(remote_re, dtype=torch.float32)         
        return local_re, remote_re
    
    def generate_paired_motion(self, epoch, level):
        start = time.time()        
        self.count = 0
        for d in [self.paired_motion_dir+'Local', self.paired_motion_dir+'Remote']:    
            for f in range(len(os.listdir(d))):    
                os.remove(os.path.join(d, str(f+self.count)+'.npy'))                         
        ####################################################################################################################################    
        if level == 'Easy':        
            print('Curriculum Learning (Easy) pairing...')
            
            for repertorie in self.repertories:
                if repertorie['repertorie']=='Pointing at single target' or repertorie['repertorie']=='Touching single target':
                    folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                        
                                        
                    for j in range(repertorie['num']):                    
                        for i in range(repertorie['num']):
                            if repertorie['repertorie'].split(' ')[0] == 'Touching':
                                for scale in [0.90, 1.00, 1.10]:
                                    self.interpolate(folder_split, folder_split, i+1, j+1, 0, 7, scale)
                            else:
                                self.interpolate(folder_split, folder_split, i+1, j+1, 0, 7, 1)                        
        ####################################################################################################################################  
        if level == 'Hard':
            print('Curriculum Learning (Hard) pairing...')
            
            for repertorie in self.repertories:
                if repertorie['repertorie']=='Pointing at two targets' or repertorie['repertorie']=='Touching two targets':
                    folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'            
                    idx = np.load(self.paired_motion_dir+'Hard('+repertorie['repertorie']+').npy').astype('uint8')
                    From = epoch * repertorie['num']
                    To = (epoch+1) * repertorie['num']
                    random_idx = idx[From:To,:]
                    for i in range(random_idx.shape[0]):
                        # print(str(folder_split)+","+str(random_idx[i,:][0])+","+str(random_idx[i,:][1]))
                        if repertorie['repertorie'].split(' ')[0] == 'Touching':
                            for scale in [0.90, 1.00, 1.10]:
                                self.interpolate(folder_split, folder_split, random_idx[i,:][0], random_idx[i,:][1], 0, 9, scale)
                        else:
                            self.interpolate(folder_split, folder_split, random_idx[i,:][0], random_idx[i,:][1], 0, 9, 1)                        

        end = time.time() - start
        print("Curriculum Learning (" + level +") pairing time="+str(end))    
####################################################################################################################################
            
if __name__ == '__main__':
    
    REPERTORIES = [{'repertorie': 'Pointing at two targets', 'num': 210},
                   {'repertorie': 'Touching two targets', 'num': 210},
                   {'repertorie': 'Pointing at single target', 'num': 27},
                   {'repertorie': 'Touching single target', 'num': 30}]
    CLIP_LENGTH = 30
    WINDOW_SIZE = 20
    GPU_ID = 0
    DATA_PAIRER = data_pairer('right', REPERTORIES, CLIP_LENGTH, WINDOW_SIZE, GPU_ID)    
    DATA_PAIRER.split()
    DATA_PAIRER = data_pairer('left', REPERTORIES, CLIP_LENGTH, WINDOW_SIZE, GPU_ID)    
    DATA_PAIRER.split()    
    # DATA_PAIRER.delete()     
    # DATA_PAIRER.generate_pair_order()

    # start = time.time()
    # DATA_PAIRER.generate_paired_motion(0, 'Easy')
    # DATA_PAIRER.generate_paired_motion(0, 'Hard')  
    # end = time.time() - start
    # print(end)    
    # DATA_PAIRER.__len__()            