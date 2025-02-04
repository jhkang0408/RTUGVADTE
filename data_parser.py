import os
import time
import torch
import os.path as osp
import numpy as np
from scipy import interpolate 

class data_parser:
    def __init__(self, handedness, repertories, window_size, clip_length, network, directory_id):
        self.repertories = repertories        
        self.WINDOW_SIZE = window_size
        self.CLIP_LENGTH = clip_length
        self.NETWORK = network
        self.raw_data_dir = 'data(raw)/'+'train/'+handedness+'/'
        self.raw_data_test_pairing_dir = 'data(raw)/'+'test_pairing/'
        self.data_dir = 'data/train'+'_'+network+'_'+str(directory_id)+'/'             
        self.count = 0

    def __len__(self):
        print('# of Path='+str(len(os.listdir(self.data_dir)))) 
        
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
        for f in os.listdir(self.data_dir):    
            os.remove(os.path.join(self.data_dir, f))
                               
    def devide(self, repertorie_name, folder_split, idx, fps, power):
        data = np.loadtxt(osp.join(folder_split, 'Data'+str(idx)+'.txt')).astype(np.float32)                                
        f = interpolate.PchipInterpolator(np.linspace(0, 1, data.shape[0]), data, axis=0)        
        data = f(np.linspace(0, 1, int(data.shape[0]*fps/60)))                
        if power!=0:
            idx_0 = np.nonzero(data[:,12])[0][0]-1
            idx_1 = np.where(data[:,12] == 1)[0][0]+1
            idx_2 = np.where(data[:,12] == 1)[0][-1]-1
            idx_3 = np.where(data[idx_2:,12] == 0)[0][0]+idx_2+1
            f1 = interpolate.PchipInterpolator(np.linspace(0, 1, idx_1-idx_0), data[idx_0:idx_1,:], axis=0)
            f2 = interpolate.PchipInterpolator(np.linspace(0, 1, idx_3-idx_2), data[idx_2:idx_3,:], axis=0)             
            if power==1:
                data_new1 = f1(np.sin(np.linspace(0, 1, idx_1-idx_0)*np.pi/2))
                data_new2 = f2(np.sin(np.linspace(0, 1, idx_3-idx_2)*np.pi/2))
            elif power==2:                
                data_new1 = f1(np.sin(np.linspace(0, 1, idx_1-idx_0)*np.pi/2)**2)
                data_new2 = f2(np.sin(np.linspace(0, 1, idx_3-idx_2)*np.pi/2)**2)                
            elif power==3:
                data_new1 = f1(np.sin(np.linspace(0, 1, idx_1-idx_0)*np.pi/2)**0.5)
                data_new2 = f2(np.sin(np.linspace(0, 1, idx_3-idx_2)*np.pi/2)**0.5)                
            elif power==4:
                data_new1 = f1(1+np.sin(np.linspace(0, 1, idx_1-idx_0)*np.pi/2+3/2*np.pi))
                data_new2 = f2(1+np.sin(np.linspace(0, 1, idx_3-idx_2)*np.pi/2+3/2*np.pi))                
            elif power==5:
                data_new1 = f1((1+np.sin(np.linspace(0, 1, idx_1-idx_0)*np.pi/2+3/2*np.pi))**2)
                data_new2 = f2((1+np.sin(np.linspace(0, 1, idx_3-idx_2)*np.pi/2+3/2*np.pi))**2)                
            data = np.concatenate((data[:idx_0, :], data_new1[:, :], data[idx_1-1:idx_2+1, :], data_new2[:, :], data[idx_3:, :]), axis=0)
        data = torch.tensor(data, dtype=torch.float32)
        
        if self.NETWORK == "MPNet":
            Data = data[:,:14]
            np.save(osp.join(self.data_dir, str(self.count)), Data[-self.CLIP_LENGTH:, :].numpy())
            self.count = self.count + 1
            Data_unfold = torch.transpose(Data.unfold(0, self.CLIP_LENGTH, (self.CLIP_LENGTH-self.WINDOW_SIZE)//2), 1, 2)

        elif self.NETWORK == "UGNet":
            motion = data[:,14:38]
            # Add "(Pointing at / Touching) single target with explanation"
            if repertorie_name=='Transition (Pointing)' or\
               repertorie_name=='Transition (Touching)' or\
               repertorie_name=='Pointing at single target with explanation' or\
               repertorie_name=='Touching single target with explanation':
                delta = motion[1:,:] - motion[:-1,:]
                gaze_progression = data[1:,11]
                deicticgesture_progression = data[1:,12]
                delta[:,:12] = (gaze_progression>0).float().unsqueeze(-1) * delta[:,:12]
                delta[:,12:] = (deicticgesture_progression>0).float().unsqueeze(-1) * delta[:,12:]
                Data = torch.cat((motion[:-1,:], data[1:,6:14], motion[:-1,:]+delta), -1)
            else:
                Data = torch.cat((motion[:-1,:], data[1:,6:14], motion[1:,:]), -1)
                
            np.save(osp.join(self.data_dir, str(self.count)), Data[-self.CLIP_LENGTH:, :].numpy())
            self.count = self.count + 1
            Data_unfold = torch.transpose(Data.unfold(0, self.CLIP_LENGTH, (self.CLIP_LENGTH-self.WINDOW_SIZE)//2), 1, 2)                                    
        
        for k in range(Data_unfold.shape[0]):
            np.save(osp.join(self.data_dir, str(self.count)), Data_unfold[k].numpy())
            self.count = self.count + 1
            
    def interpolate_evaluation_DIP(self, test_pairing_folder_split, folder_split, idx_L, idx_R):
        local = np.loadtxt(osp.join(test_pairing_folder_split, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(test_pairing_folder_split, 'Data'+str(idx_R)+'.txt')).astype(np.float32)                
        key_idx = local.shape[1]-1
        local_keys = np.append(np.where(local[:,key_idx]==1), local.shape[0]-1)
        remote_keys = np.append(np.where(remote[:,key_idx]==1), remote.shape[0]-1)                
                
        local = np.loadtxt(osp.join(folder_split, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(folder_split, 'Data'+str(idx_R)+'.txt')).astype(np.float32)        
        local_re = np.empty((0, local.shape[1]), np.float32)
        remote_re = np.empty((0, local.shape[1]), np.float32)
            
        key_DIP = 0
        # print(local_keys.shape[0])
        for k in range(local_keys.shape[0]-1): # progression
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
        first_frame = local_re[0].unsqueeze(0)
        repeated_frames = first_frame.repeat(30, 1)
        local_re = torch.cat((repeated_frames, local_re), dim=0)
        
        remote_re = torch.tensor(remote_re, dtype=torch.float32)         
        first_frame = remote_re[0].unsqueeze(0)
        repeated_frames = first_frame.repeat(30, 1)
        remote_re = torch.cat((repeated_frames, remote_re), dim=0)
        
        return local_re[:key_DIP+40,:], remote_re[:key_DIP+40,:]
            
    def interpolate_evaluation_MN(self, test_pairing_folder_split, folder_split, idx_L, idx_R):
        local = np.loadtxt(osp.join(test_pairing_folder_split, 'Data'+str(idx_L)+'.txt')).astype(np.float32)
        remote = np.loadtxt(osp.join(test_pairing_folder_split, 'Data'+str(idx_R)+'.txt')).astype(np.float32)                
        key_idx = local.shape[1]-1
        local_keys = np.append(np.where(local[:,key_idx]==1), local.shape[0]-1)
        remote_keys = np.append(np.where(remote[:,key_idx]==1), remote.shape[0]-1)
        
        local = np.loadtxt(osp.join(folder_split, 'Data'+str(idx_L)+'.txt')).astype(np.float32)        
        remote = np.loadtxt(osp.join(folder_split, 'Data'+str(idx_R)+'.txt')).astype(np.float32)        
        local_re = np.empty((0, local.shape[1]), np.float32)
        remote_re = np.empty((0, local.shape[1]), np.float32) 
        
        for k in range(local_keys.shape[0]-1): # progression
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
                ynew = f(xnew) # upsampled local
                local_re = np.concatenate((local_re, ynew), axis=0) 
                remote_re = np.concatenate((remote_re, remote[remote_keys[k]:remote_keys[k+1], :]), axis=0)                       
        if local_re.shape != remote_re.shape:
            print("Error Occured in Pairing")
        
        local_re = torch.tensor(local_re, dtype=torch.float32)
        first_frame = local_re[0].unsqueeze(0)
        repeated_frames = first_frame.repeat(30, 1)
        local_re = torch.cat((repeated_frames, local_re), dim=0)
        
        remote_re = torch.tensor(remote_re, dtype=torch.float32)         
        first_frame = remote_re[0].unsqueeze(0)
        repeated_frames = first_frame.repeat(30, 1)
        remote_re = torch.cat((repeated_frames, remote_re), dim=0)
        
        return local_re, remote_re 
            
    def generate(self, augmentation):
        start = time.time()
        self.count = 0
        for repertorie in self.repertories:
            folder_split = self.raw_data_dir + repertorie['repertorie'] + '/' + 'split'
            for i in range(repertorie['num']):
                if augmentation == True and (repertorie['repertorie']=='Pointing at two targets' or repertorie['repertorie']=='Touching two targets'):
                    for f in [15, 30, 60]:
                        self.devide(repertorie['repertorie'], folder_split, i+1, f, 0)
                elif augmentation == True and (repertorie['repertorie']=='Pointing at single target' or repertorie['repertorie']=='Touching single target'):
                    for p in [1, 2, 3, 4, 5]:
                        for f in [15, 30, 60]:
                            self.devide(repertorie['repertorie'], folder_split, i+1, f, p)
                else:
                    self.devide(repertorie['repertorie'], folder_split, i+1, 30, 0)
                    
        end = time.time() - start
        print("data generation time="+str(end))
####################################################################################################################################
            
if __name__ == '__main__':
 
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
    WINDOW_SIZE = 30
    CLIP_LENGTH = 40
    DIRECTORY_ID = 0
    DATA_PARSER = data_parser('right' , REPERTORIES, WINDOW_SIZE, CLIP_LENGTH, "MPNet", DIRECTORY_ID)
    # DATA_PARSER = data_parser('right', REPERTORIES, WINDOW_SIZE, CLIP_LENGTH, "UGNet", DIRECTORY_ID)
    
    # DATA_PARSER.delete()     
    # DATA_PARSER.split()    
    # DATA_PARSER.generate()
    # DATA_PARSER.__len__()
    
    # Prepaire Test MPNet Accuracy
    for subject in ['161', '172', '173', '174', '180']:
        DATA_PARSER = data_parser('', REPERTORIES, 30, 45, '', 0)
        DATA_PARSER.raw_data_dir = 'data(raw)/test_MPNet/' + subject + '/'            
        DATA_PARSER.split()
        
        for repertorie in REPERTORIES:
            folder_raw = DATA_PARSER.raw_data_dir + repertorie['repertorie'] + '/' + 'raw'
            folder_split = DATA_PARSER.raw_data_dir  + repertorie['repertorie'] + '/' + 'split'
            for f in os.listdir(folder_split):
                os.remove(os.path.join(folder_split, f))
            input_data = np.loadtxt(osp.join(folder_raw, 'Data.txt')).astype(np.float32)
            sequences = np.loadtxt(osp.join(folder_raw, 'Sequences.txt')).astype(np.float32)
            for i in range(1, int(np.max(sequences))+1):
                np.savetxt(osp.join(folder_split, 'Data'+str(i)+'.txt'), input_data[np.where(sequences==i)], fmt='%1.4e')      
        