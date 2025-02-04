import os
import os.path as osp
import gc
import math
import random
import numpy as np
import scipy.linalg as linalg
import datetime 

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model import MPNet_MoE, MPNet_MLP, MPNet_Transformer
from model import UGNet_MoE, UGNet_Transformer, UGNet_MLP, UGNet_Diffusion
from model import ConvolutionalAutoEncoder

from data_parser import data_parser
from ISMAR2023.rotation import rotation_matrix_to_angle_axis, rad2deg

class Runner(object):                                              
    def __init__(self, directory_id, device, MPNet, UGNet, scale):               
        self.device = device
        self.MPNet = MPNet.to(self.device)
        self.UGNet = UGNet.to(self.device)
        self.scale = scale
        self.iter = 0
        
        self.Horizontal_Deviation = 0
        self.Vertical_Deviation = 0
        self.Position_Deviation = 0  
        
        self.Frechet_Motion_Distance = 0
        self.Generated_Motion_Dir = 'data/'+'test(Generated)_'+str(directory_id)+'/'
        self.Real_Motion_Dir = 'data/'+'test(Real)_'+str(directory_id)+'/'
        
        self.Total_Accuracy = 0
        self.Gaze_Accuracy = 0
        self.DeicticGesture_Accuracy = 0
        self.Idle_Accuracy = 0 
        
        self.L = 30
        self.count = 0                     
                
    def sixd2matrot(self, pose_6d):
        rot_vec_1 = F.normalize(pose_6d[:,0:3], dim=-1)
        rot_vec_2 = F.normalize(pose_6d[:,3:6], dim=-1)
        rot_vec_3 = torch.cross(rot_vec_2, rot_vec_1)
        pose_matrot = torch.stack([rot_vec_3, rot_vec_2, rot_vec_1],dim=-1)
        return pose_matrot
    
    def SignedAngle(self, From, To, Axis):
        unsignedAngle = torch.arccos(torch.clip(torch.dot(From, To), -1.0, 1.0))
        cross_x = From[1] * To[2] - From[2] * To[1]
        cross_y = From[2] * To[0] - From[0] * To[2]
        cross_z = From[0] * To[1] - From[1] * To[0]
        sign = torch.sign(Axis[0] * cross_x + Axis[1] * cross_y + Axis[2] * cross_z)
        return unsignedAngle * sign;     
        
    def compute_Embedding(self, autoencoder, dataloader):
        embeddings = []        
        for data in dataloader:            
            embedding = autoencoder(data['data'].to(self.device), True)  
            embeddings.append(embedding.squeeze())                    
        return torch.stack(embeddings).detach().cpu().numpy()
        
    def calculate_FMD(self, real_embeddings, generated_embeddings):
        # calculate mean and covariance statistics
        mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
        mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        self.Frechet_Motion_Distance = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)    
                        
    def run_DIP(self, local, remote): # Deictic Intention Preservation
        local = local.to(self.device)
        remote = remote.to(self.device)
        
        local = local.unsqueeze(0) # (1, T, :)
        remote = remote.unsqueeze(0) # (1, T, :)
                        
        T = local.shape[1]-WINDOW_SIZE
        
        Idle_Threshold = 0.15
        Gaze_Threshold = 0.25
        DeicticGesture_Threshold = 0.15
        
        Head_Target_Changing = False
        Head_Target_Changing_Count = 0
        Hand_Target_Changing = False
        Hand_Target_Changing_Count = 0
        
        Head_Blending_Count = 0
        Hand_Blending_Count = 0
        Head_Blending_Window_Length = 20
        Hand_Blending_Window_Length = 20
        
        Progression = torch.zeros(1,WINDOW_SIZE,3).to(self.device)
        Progression[:,:,-1] = 1
        for t in range(T):
            Input_MPNet = torch.cat((local[:, t:WINDOW_SIZE+t, :5], local[:, t:WINDOW_SIZE+t, 6:12]), -1)                    
            progression = self.MPNet(Input_MPNet)  
            Progression_temp = Progression.clone()
            Progression[:,:-1,:] = Progression_temp[:,1:,:]
            Progression[0,-1,:] = progression.squeeze(1)            
            
            Gaze_Progression = Progression[0,-1,0]
            DeicticGesture_Progression = Progression[0,-1,1]
            Idle_Progression = Progression[0,-1,2]
            
            Input_UGNet = torch.cat((remote[:, t:WINDOW_SIZE+t, 12:36], remote[:, t:WINDOW_SIZE+t, 7:12], Progression), -1)            

            if UGNet_MODEL != 'Diffusion':
                output_data = self.UGNet(Input_UGNet)
            else:
                output_data = self.UGNet.sample_ddpm(Input_UGNet) 
                
            # Check Head Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 7:9], remote[:, WINDOW_SIZE+t-2, 7:9]) == False:
                Head_Target_Changing == True
                Head_Target_Changing_Count = 0
                
            if Head_Target_Changing == True:
                if Head_Target_Changing_Count < 45:
                    Head_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Head_Target_Changing == False
                    Head_Target_Changing_Count = 0              
            
            # Check Hand Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 9:12], remote[:, WINDOW_SIZE+t-2, 9:12]) == False:
                Hand_Target_Changing == True
                Hand_Target_Changing_Count = 0
            
            if Hand_Target_Changing == True:
                if Hand_Target_Changing_Count < 45:
                    Hand_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Hand_Target_Changing == False
                    Hand_Target_Changing_Count = 0 
            
            # Head Blending
            if Idle_Progression < Idle_Threshold and Gaze_Progression < Gaze_Threshold:
                if not Head_Target_Changing:
                    if Head_Blending_Count < Head_Blending_Window_Length:
                        Head_Blending_Count += 1
            else:
                if Head_Blending_Count > 0:
                    Head_Blending_Count -= 1    
            Head_Blending = Head_Blending_Count / Head_Blending_Window_Length
            
            # Hand Blending
            if Idle_Progression < Idle_Threshold and DeicticGesture_Progression < DeicticGesture_Threshold:
                if not Hand_Target_Changing:
                    if Hand_Blending_Count < Hand_Blending_Window_Length:
                        Hand_Blending_Count += 1
            else:
                if Hand_Blending_Count > 0:
                    Hand_Blending_Count -= 1           
            Hand_Blending = Hand_Blending_Count / Hand_Blending_Window_Length
            
            ## autoregression
            # Gaze Progression
            remote[:, WINDOW_SIZE+t, 12:15] = torch.nn.functional.normalize(torch.lerp(output_data[:, 0:3], local[:, WINDOW_SIZE+t, 12:15], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 15:18] = torch.nn.functional.normalize(torch.lerp(output_data[:, 3:6], local[:, WINDOW_SIZE+t, 15:18], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 18:21] = torch.nn.functional.normalize(torch.lerp(output_data[:, 6:9], local[:, WINDOW_SIZE+t, 18:21], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 21:24] = torch.lerp(output_data[:, 9:12], local[:, WINDOW_SIZE+t, 21:24], Head_Blending)            
            # Deictic Gesture Progression     
            remote[:, WINDOW_SIZE+t, 24:27] = torch.nn.functional.normalize(torch.lerp(output_data[:, 12:15], local[:, WINDOW_SIZE+t, 24:27], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 27:30] = torch.nn.functional.normalize(torch.lerp(output_data[:, 15:18], local[:, WINDOW_SIZE+t, 27:30], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 30:33] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 18:21], local[:, WINDOW_SIZE+t, 30:33], Hand_Blending)
            remote[:, WINDOW_SIZE+t, 33:36] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 21:24], local[:, WINDOW_SIZE+t, 33:36], Hand_Blending)            
        
        Head_rot = torch.squeeze(self.sixd2matrot(output_data[:,3:9]))
        Head_pos = output_data[0,9:12]        
        RightHand_rot_local = torch.squeeze(self.sixd2matrot(output_data[:,12:18]))
        RightHand_pos_local = torch.squeeze(output_data[0,18:21])                
        IndexFingerTip_pos_RightHand = torch.matmul(RightHand_rot_local, output_data[0,21:24])+RightHand_pos_local        
        IndexFingerTip_pos = torch.matmul(Head_rot, IndexFingerTip_pos_RightHand)+Head_pos        
        
        Target_pos = remote[0,-1,36:39]
        EFR = F.normalize(IndexFingerTip_pos-Head_pos, dim=0)        
        ET = F.normalize(Target_pos-Head_pos, dim=0)
        
        # 3. Evaluation
        EFR_Horizon = F.normalize(EFR * torch.tensor([1,0,1]).to(self.device), dim=0).cpu().numpy()
        ET_Horizon = F.normalize(ET * torch.tensor([1,0,1]).to(self.device), dim=0).cpu().numpy()
        EFR_Vertical = F.normalize(EFR * torch.tensor([0,1,1]).to(self.device), dim=0).cpu().numpy()        
        ET_Vertical = F.normalize(ET * torch.tensor([0,1,1]).to(self.device), dim=0).cpu().numpy()        
        
        Horizontal_Deviation = np.rad2deg(np.arccos(np.clip(np.dot(EFR_Horizon, ET_Horizon), -1.0, 1.0)))
        Vertical_Deviation = np.rad2deg(np.arccos(np.clip(np.dot(EFR_Vertical, ET_Vertical), -1.0, 1.0)))
        Position_Deviation = np.linalg.norm(IndexFingerTip_pos.cpu().numpy()-Target_pos.cpu().numpy())
        
        self.Horizontal_Deviation += Horizontal_Deviation
        self.Vertical_Deviation += Vertical_Deviation
        self.Position_Deviation += Position_Deviation
        self.iter += 1 
        
    def result_DIP(self, touch):
        if touch==False:
            log = "Pointing, " + str(self.iter).zfill(5) + ", [HD: %.3f (°)] [VD: %.3f (°)]" % (self.Horizontal_Deviation/self.iter, self.Vertical_Deviation/self.iter)
            with open('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')'+"/DIP"+str(SUBJECT)+"(Pointing).txt", "w") as file:
                file.write(log)
        else:
            log = "Touching, " + str(self.iter).zfill(5) + ", [PD: %.3f (cm)] " % (self.Position_Deviation/self.iter*100)
            with open('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')'+"/DIP"+str(SUBJECT)+"(Touching).txt", "w") as file:
                file.write(log)            
        print(log)  
        
    def run_MN(self, local, remote): # Movement Naturalness
        local = local.to(self.device)
        remote = remote.to(self.device)
        
        local = local.unsqueeze(0) # (1, T, :)
        remote = remote.unsqueeze(0) # (1, T, :)
                
        generated_list = []
        T = local.shape[1]-WINDOW_SIZE
        
        Idle_Threshold = 0.15
        Gaze_Threshold = 0.25
        DeicticGesture_Threshold = 0.15
        
        Head_Target_Changing = False
        Head_Target_Changing_Count = 0
        Hand_Target_Changing = False
        Hand_Target_Changing_Count = 0
        
        Head_Blending_Count = 0
        Hand_Blending_Count = 0
        Head_Blending_Window_Length = 20
        Hand_Blending_Window_Length = 20
        
        Progression = torch.zeros(1, WINDOW_SIZE, 3).to(self.device)
        Progression[:,:,-1] = 1
        for t in range(T):
            Input_MPNet = torch.cat((local[:, t:WINDOW_SIZE+t, :5], local[:, t:WINDOW_SIZE+t, 6:12]), -1)                    
            progression = self.MPNet(Input_MPNet)  
            Progression_temp = Progression.clone()
            Progression[:,:-1,:] = Progression_temp[:,1:,:]
            Progression[0,-1,:] = progression.squeeze(1)            
            
            Gaze_Progression = Progression[0,-1,0]
            DeicticGesture_Progression = Progression[0,-1,1]
            Idle_Progression = Progression[0,-1,2]
            
            Input_UGNet = torch.cat((remote[:, t:WINDOW_SIZE+t, 12:36], remote[:, t:WINDOW_SIZE+t, 7:12], Progression), -1)            

            if UGNet_MODEL != 'Diffusion':
                output_data = self.UGNet(Input_UGNet)
            else:
                output_data = self.UGNet.sample_ddpm(Input_UGNet) 
                
            # Save
            generated_list.append(torch.squeeze(output_data[0,3:21]))
        
            # Check Head Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 7:9], remote[:, WINDOW_SIZE+t-2, 7:9]) == False:
                Head_Target_Changing == True
                Head_Target_Changing_Count = 0
                
            if Head_Target_Changing == True:
                if Head_Target_Changing_Count < 45:
                    Head_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Head_Target_Changing == False
                    Head_Target_Changing_Count = 0              
            
            # Check Hand Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 9:12], remote[:, WINDOW_SIZE+t-2, 9:12]) == False:
                Hand_Target_Changing == True
                Hand_Target_Changing_Count = 0
            
            if Hand_Target_Changing == True:
                if Hand_Target_Changing_Count < 45:
                    Hand_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Hand_Target_Changing == False
                    Hand_Target_Changing_Count = 0 
            
            # Head Blending
            if Idle_Progression < Idle_Threshold and Gaze_Progression < Gaze_Threshold:
                if not Head_Target_Changing:
                    if Head_Blending_Count < Head_Blending_Window_Length:
                        Head_Blending_Count += 1
            else:
                if Head_Blending_Count > 0:
                    Head_Blending_Count -= 1    
            Head_Blending = Head_Blending_Count / Head_Blending_Window_Length
            
            # Hand Blending
            if Idle_Progression < Idle_Threshold and DeicticGesture_Progression < DeicticGesture_Threshold:
                if not Hand_Target_Changing:
                    if Hand_Blending_Count < Hand_Blending_Window_Length:
                        Hand_Blending_Count += 1
            else:
                if Hand_Blending_Count > 0:
                    Hand_Blending_Count -= 1           
            Hand_Blending = Hand_Blending_Count / Hand_Blending_Window_Length
            
            ## autoregression
            # Gaze Progression
            remote[:, WINDOW_SIZE+t, 12:15] = torch.nn.functional.normalize(torch.lerp(output_data[:, 0:3], local[:, WINDOW_SIZE+t, 12:15], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 15:18] = torch.nn.functional.normalize(torch.lerp(output_data[:, 3:6], local[:, WINDOW_SIZE+t, 15:18], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 18:21] = torch.nn.functional.normalize(torch.lerp(output_data[:, 6:9], local[:, WINDOW_SIZE+t, 18:21], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 21:24] = torch.lerp(output_data[:, 9:12], local[:, WINDOW_SIZE+t, 21:24], Head_Blending)            
            # Deictic Gesture Progression     
            remote[:, WINDOW_SIZE+t, 24:27] = torch.nn.functional.normalize(torch.lerp(output_data[:, 12:15], local[:, WINDOW_SIZE+t, 24:27], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 27:30] = torch.nn.functional.normalize(torch.lerp(output_data[:, 15:18], local[:, WINDOW_SIZE+t, 27:30], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 30:33] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 18:21], local[:, WINDOW_SIZE+t, 30:33], Hand_Blending)
            remote[:, WINDOW_SIZE+t, 33:36] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 21:24], local[:, WINDOW_SIZE+t, 33:36], Hand_Blending)
            
        real = local[0,WINDOW_SIZE:,15:33]
        generated = torch.stack(generated_list)
        
        # Save
        np.save(osp.join(self.Real_Motion_Dir, str(self.count)), real[-self.L:,:].cpu().numpy())        
        np.save(osp.join(self.Generated_Motion_Dir, str(self.count)), generated[-self.L:,:].cpu().numpy())        
        self.count = self.count + 1
        
        real = torch.transpose(real.unfold(0, self.L, int(self.L/2)), 1, 2)
        generated = torch.transpose(generated.unfold(0, self.L, int(self.L/2)), 1, 2)
        for i in range(real.shape[0]):
            np.save(osp.join(self.Real_Motion_Dir, str(self.count)), real[i].cpu().numpy())
            np.save(osp.join(self.Generated_Motion_Dir, str(self.count)), generated[i].cpu().numpy())            
            self.count = self.count + 1  

    def result_MN(self):
        log = "[FMD: %.3f]" % (self.Frechet_Motion_Distance)
        print(log)  

    def run_missingtracking(self, local, remote):
        local = local.to(self.device)
        remote = remote.to(self.device)
        
        local = local.unsqueeze(0) # (1, T, :)
        remote = remote.unsqueeze(0) # (1, T, :)
                
        generated_list = []
        T = local.shape[1]-WINDOW_SIZE
        
        Idle_Threshold = 0.15
        Gaze_Threshold = 0.25
        DeicticGesture_Threshold = 0.15
        
        Head_Target_Changing = False
        Head_Target_Changing_Count = 0
        Hand_Target_Changing = False
        Hand_Target_Changing_Count = 0
        
        Head_Blending_Count = 0
        Hand_Blending_Count = 0
        Head_Blending_Window_Length = 20
        Hand_Blending_Window_Length = 20
        
        Progression = torch.zeros(1, WINDOW_SIZE, 3).to(self.device)
        Progression[:,:,-1] = 1
        for t in range(T):
            Input_MPNet = torch.cat((local[:, t:WINDOW_SIZE+t, :5], local[:, t:WINDOW_SIZE+t, 6:12]), -1)                    
            progression = self.MPNet(Input_MPNet)  
            Progression_temp = Progression.clone()
            Progression[:,:-1,:] = Progression_temp[:,1:,:]
            Progression[0,-1,:] = progression.squeeze(1)            
            
            Gaze_Progression = Progression[0,-1,0]
            DeicticGesture_Progression = Progression[0,-1,1]
            Idle_Progression = Progression[0,-1,2]
            
            Input_UGNet = torch.cat((remote[:, t:WINDOW_SIZE+t, 12:36], remote[:, t:WINDOW_SIZE+t, 7:12], Progression), -1)            

            if UGNet_MODEL != 'Diffusion':
                output_data = self.UGNet(Input_UGNet)
            else:
                output_data = self.UGNet.sample_ddpm(Input_UGNet) 
                
            # Save
            generated_list.append(torch.squeeze(output_data[0,3:21]))
        
            # Check Head Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 7:9], remote[:, WINDOW_SIZE+t-2, 7:9]) == False:
                Head_Target_Changing == True
                Head_Target_Changing_Count = 0
                
            if Head_Target_Changing == True:
                if Head_Target_Changing_Count < 45:
                    Head_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Head_Target_Changing == False
                    Head_Target_Changing_Count = 0              
            
            # Check Hand Target Change
            if torch.allclose(remote[:, WINDOW_SIZE+t-1, 9:12], remote[:, WINDOW_SIZE+t-2, 9:12]) == False:
                Hand_Target_Changing == True
                Hand_Target_Changing_Count = 0
            
            if Hand_Target_Changing == True:
                if Hand_Target_Changing_Count < 45:
                    Hand_Target_Changing_Count += 1
                elif Head_Target_Changing_Count >= 45:
                    Hand_Target_Changing == False
                    Hand_Target_Changing_Count = 0 
            
            # Head Blending
            if Idle_Progression < Idle_Threshold and Gaze_Progression < Gaze_Threshold:
                if not Head_Target_Changing:
                    if Head_Blending_Count < Head_Blending_Window_Length:
                        Head_Blending_Count += 1
            else:
                if Head_Blending_Count > 0:
                    Head_Blending_Count -= 1    
            Head_Blending = Head_Blending_Count / Head_Blending_Window_Length
            
            # Hand Blending
            if Idle_Progression < Idle_Threshold and DeicticGesture_Progression < DeicticGesture_Threshold:
                if not Hand_Target_Changing:
                    if Hand_Blending_Count < Hand_Blending_Window_Length:
                        Hand_Blending_Count += 1
            else:
                if Hand_Blending_Count > 0:
                    Hand_Blending_Count -= 1           
            Hand_Blending = Hand_Blending_Count / Hand_Blending_Window_Length
            
            ## autoregression
            # Gaze Progression
            remote[:, WINDOW_SIZE+t, 12:15] = torch.nn.functional.normalize(torch.lerp(output_data[:, 0:3], local[:, WINDOW_SIZE+t, 12:15], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 15:18] = torch.nn.functional.normalize(torch.lerp(output_data[:, 3:6], local[:, WINDOW_SIZE+t, 15:18], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 18:21] = torch.nn.functional.normalize(torch.lerp(output_data[:, 6:9], local[:, WINDOW_SIZE+t, 18:21], Head_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 21:24] = torch.lerp(output_data[:, 9:12], local[:, WINDOW_SIZE+t, 21:24], Head_Blending)            
            # Deictic Gesture Progression     
            remote[:, WINDOW_SIZE+t, 24:27] = torch.nn.functional.normalize(torch.lerp(output_data[:, 12:15], local[:, WINDOW_SIZE+t, 24:27], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 27:30] = torch.nn.functional.normalize(torch.lerp(output_data[:, 15:18], local[:, WINDOW_SIZE+t, 27:30], Hand_Blending), p=2, dim=1)
            remote[:, WINDOW_SIZE+t, 30:33] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 18:21], local[:, WINDOW_SIZE+t, 30:33], Hand_Blending)
            remote[:, WINDOW_SIZE+t, 33:36] = torch.lerp(torch.lerp(torch.tensor([1/self.scale]), torch.tensor([1.0]), torch.tensor(Hand_Blending)).to(self.device) * output_data[:, 21:24], local[:, WINDOW_SIZE+t, 33:36], Hand_Blending)
            
        generated = torch.stack(generated_list)
        
        return generated

    def missingtracking(self, local, num_frame_drop):
       ''' 
          # num_frame_drop
           6: 0.2s
          12: 0.4s
          18: 0.6s
          24: 0.8s
          30: 1.0s
       '''
       local_noisy = local.clone()  # Create copy to avoid modifying original
       
       for i in range(30, local.shape[0], 30):  # Every 30 frames
           if random.random() < 0.3:  # 30% probability
               local_noisy[i:i+num_frame_drop] = local_noisy[i-1:i]  # Repeat previous frame's values
               
       return local_noisy

    def calculate_motion_metrics(self, clean_motion, noisy_motion):
        FPS = 30 
        
        # Position error (head + hand) in meters
        head_pos_error = torch.mean(torch.norm(clean_motion[:,6:9] - noisy_motion[:,6:9], dim=1))
        hand_pos_error = torch.mean(torch.norm(clean_motion[:,15:18] - noisy_motion[:,15:18], dim=1))
        position_error = (head_pos_error + hand_pos_error) / 2.0
        
        # Rotation error (head + hand) in degrees
        # Head
        clean_head_matrot = self.sixd2matrot(clean_motion[:,0:6])
        noisy_head_matrot = self.sixd2matrot(noisy_motion[:,0:6])
        clean_head_matrot = torch.cat([clean_head_matrot, torch.zeros(clean_head_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        noisy_head_matrot = torch.cat([noisy_head_matrot, torch.zeros(noisy_head_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        clean_head_rot = rad2deg(rotation_matrix_to_angle_axis(clean_head_matrot))
        noisy_head_rot = rad2deg(rotation_matrix_to_angle_axis(noisy_head_matrot))
        head_rot_error = torch.mean(torch.norm(clean_head_rot - noisy_head_rot, dim=1))
        
        # Hand
        clean_hand_matrot = self.sixd2matrot(clean_motion[:,9:15])
        noisy_hand_matrot = self.sixd2matrot(noisy_motion[:,9:15])
        clean_hand_matrot = torch.cat([clean_hand_matrot, torch.zeros(clean_hand_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        noisy_hand_matrot = torch.cat([noisy_hand_matrot, torch.zeros(noisy_hand_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        clean_hand_rot = rad2deg(rotation_matrix_to_angle_axis(clean_hand_matrot))
        noisy_hand_rot = rad2deg(rotation_matrix_to_angle_axis(noisy_hand_matrot))
        hand_rot_error = torch.mean(torch.norm(clean_hand_rot - noisy_hand_rot, dim=1))
        
        rot_error = (head_rot_error + hand_rot_error) / 2.0
        
        # Linear velocity error in meters/sec
        clean_head_vel = FPS * (clean_motion[1:,6:9] - clean_motion[:-1,6:9])
        noisy_head_vel = FPS * (noisy_motion[1:,6:9] - noisy_motion[:-1,6:9])
        head_vel_error = torch.mean(torch.norm(clean_head_vel - noisy_head_vel, dim=1))
        
        clean_hand_vel = FPS * (clean_motion[1:,15:18] - clean_motion[:-1,15:18])
        noisy_hand_vel = FPS * (noisy_motion[1:,15:18] - noisy_motion[:-1,15:18]) 
        hand_vel_error = torch.mean(torch.norm(clean_hand_vel - noisy_hand_vel, dim=1))
            
        vel_error = (head_vel_error + hand_vel_error) / 2.0
        
        # Angular velocity error in degrees/sec
        clean_head_matrot = torch.matmul(
            self.sixd2matrot(clean_motion[1:,0:6]), 
            self.sixd2matrot(clean_motion[:-1,0:6]).transpose(-2,-1)
        )
        noisy_head_matrot = torch.matmul(
            self.sixd2matrot(noisy_motion[1:,0:6]),
            self.sixd2matrot(noisy_motion[:-1,0:6]).transpose(-2,-1)
        )
        clean_head_matrot = torch.cat([clean_head_matrot, torch.zeros(clean_head_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        noisy_head_matrot = torch.cat([noisy_head_matrot, torch.zeros(noisy_head_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        clean_head_angvel = rad2deg(FPS * rotation_matrix_to_angle_axis(clean_head_matrot))
        noisy_head_angvel = rad2deg(FPS * rotation_matrix_to_angle_axis(noisy_head_matrot))
        head_angvel_error = torch.mean(torch.norm(clean_head_angvel - noisy_head_angvel, dim=1))
        
        clean_hand_matrot = torch.matmul(
            self.sixd2matrot(clean_motion[1:,9:15]),
            self.sixd2matrot(clean_motion[:-1,9:15]).transpose(-2,-1)
        )
        noisy_hand_matrot = torch.matmul(
            self.sixd2matrot(noisy_motion[1:,9:15]),
            self.sixd2matrot(noisy_motion[:-1,9:15]).transpose(-2,-1)
        )
        clean_hand_matrot = torch.cat([clean_hand_matrot, torch.zeros(clean_hand_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        noisy_hand_matrot = torch.cat([noisy_hand_matrot, torch.zeros(noisy_hand_matrot.shape[0], 3, 1).to(self.device)], dim=2)
        clean_hand_angvel = rad2deg(FPS * rotation_matrix_to_angle_axis(clean_hand_matrot))
        noisy_hand_angvel = rad2deg(FPS * rotation_matrix_to_angle_axis(noisy_hand_matrot))
        hand_angvel_error = torch.mean(torch.norm(clean_hand_angvel - noisy_hand_angvel, dim=1))
        
        ang_vel_error = (head_angvel_error + hand_angvel_error) / 2.0
    
        return position_error, rot_error, vel_error, ang_vel_error
    
    def run_Responsiveness(self, initial_pose, initial_target_feature, changed_target_feature, changed_target_position):    
        initial_pose = initial_pose.to(self.device)
        initial_target_feature = initial_target_feature.to(self.device)
        changed_target_feature = changed_target_feature.to(self.device)
        changed_target_position = changed_target_position.to(self.device)
    
        T = 30
        Progression = torch.zeros(1,WINDOW_SIZE,3).to(self.device)
        Progression[:,:,0] = 1
        Progression[:,:,1] = 1
        Progression[:,:,2] = 0
                                
        UGNet_Pose = initial_pose.repeat(WINDOW_SIZE, 1).unsqueeze(0)
        UGNet_Target = initial_target_feature.repeat(WINDOW_SIZE, 1).unsqueeze(0)
        UGNet_Target[0,-1,:] = changed_target_feature 
        
        for t in range(T):
            Input_UGNet = torch.cat((UGNet_Pose, UGNet_Target, Progression), -1)            

            if UGNet_MODEL != 'Diffusion':
                output_data = self.UGNet(Input_UGNet)
            else:
                output_data = self.UGNet.sample_ddpm(Input_UGNet) 

            # autoregression
            UGNet_Pose[:,-1,:] = output_data
            UGNet_Target_temp = UGNet_Target.clone()
            UGNet_Target[:,:-1,:] = UGNet_Target_temp[:,1:,:]
            UGNet_Target[0,-1,:] = changed_target_feature
            
            Head_rot = torch.squeeze(self.sixd2matrot(output_data[:,3:9]))
            Head_pos = output_data[0,9:12]        
            RightHand_rot_local = torch.squeeze(self.sixd2matrot(output_data[:,12:18]))
            RightHand_pos_local = torch.squeeze(output_data[0,18:21])                
            IndexFingerTip_pos_RightHand = torch.matmul(RightHand_rot_local, output_data[0,21:24])+RightHand_pos_local        
            IndexFingerTip_pos = torch.matmul(Head_rot, IndexFingerTip_pos_RightHand)+Head_pos        
            
            Target_pos = changed_target_position
            EFR = F.normalize(IndexFingerTip_pos-Head_pos, dim=0)        
            ET = F.normalize(Target_pos-Head_pos, dim=0)
            
            # 3. Evaluation
            EFR_Horizon = F.normalize(EFR * torch.tensor([1,0,1]).to(self.device), dim=0).cpu().numpy()
            ET_Horizon = F.normalize(ET * torch.tensor([1,0,1]).to(self.device), dim=0).cpu().numpy()
            EFR_Vertical = F.normalize(EFR * torch.tensor([0,1,1]).to(self.device), dim=0).cpu().numpy()        
            ET_Vertical = F.normalize(ET * torch.tensor([0,1,1]).to(self.device), dim=0).cpu().numpy()        
            
            Horizontal_Deviation = np.rad2deg(np.arccos(np.clip(np.dot(EFR_Horizon, ET_Horizon), -1.0, 1.0)))
            Vertical_Deviation = np.rad2deg(np.arccos(np.clip(np.dot(EFR_Vertical, ET_Vertical), -1.0, 1.0)))
            Position_Deviation = np.linalg.norm(IndexFingerTip_pos.cpu().numpy()-Target_pos.cpu().numpy())
            
            self.Horizontal_Deviation += Horizontal_Deviation
            self.Vertical_Deviation += Vertical_Deviation
            self.Position_Deviation += Position_Deviation
        
        self.iter += 1
               
    def result_Responsiveness(self, touch):
        if touch==False:
            log = "Pointing, " + str(self.iter).zfill(5) + ", [HD: %.3f (°)] [VD: %.3f (°)]" % (self.Horizontal_Deviation/self.iter/30, self.Vertical_Deviation/self.iter/30)
            with open('Quantitative Evaluation(Responsiveness)'+'(UGNet_'+UGNet_MODEL+')'+"/Responsiveness(Pointing).txt", "w") as file:
                file.write(log)
        else:
            log = "Touching, " + str(self.iter).zfill(5) + ", [PD: %.3f (cm)] " % (self.Position_Deviation/self.iter*100/30)
            with open('Quantitative Evaluation(Responsiveness)'+'(UGNet_'+UGNet_MODEL+')'+"/Responsiveness(Touching).txt", "w") as file:
                file.write(log)
        print(log)   
        
class Dataset(Dataset):
    def __init__(self, folder_name=None, **kwargs):
        self.sequences = len(os.listdir(folder_name))        
        self.paths = [folder_name+str(i)+'.npy' for i in range(self.sequences)]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        data = torch.from_numpy(np.load(self.paths[idx]))        
        return {'data': data}
                    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default='161', help='Directory')
parser.add_argument('--mpnet_model', default='MoE', help='MPNet Model')
parser.add_argument('--ugnet_model', default='MoE', help='UGNet Model')
parser.add_argument('--motion_category', default='AC', help='Motion Category')
parser.add_argument('--metric', default='DIP', help='Metric')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--directory_id', type=int, default=0, help='DIRECTORY ID')
args = parser.parse_args()

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')

    SUBJECT = args.subject
    MPNet_MODEL = args.mpnet_model
    UGNet_MODEL = args.ugnet_model
    MOTION_CATEGORY = args.motion_category
    METRIC = args.metric
    GPU_ID = args.gpu_id
    DIRECTORY_ID = args.directory_id
    
    DIR = 'data(raw)/test/' + SUBJECT + '/'
            
    SCALE=1
    if SUBJECT == '161':
        SCALE = 0.9471
    elif SUBJECT == '172':
        SCALE = 1.0118
    elif SUBJECT == '173':
        SCALE = 1.0176
    elif SUBJECT == '174':    
        SCALE = 1.0235        
    elif SUBJECT == '180':
        SCALE = 1.0588
        
    if SUBJECT == '161' or SUBJECT == '172' or SUBJECT == '180':
        if MOTION_CATEGORY == 'AC':     
            # MPNet
            if MPNet_MODEL == 'MoE':    
                MPNet_path = "model(trained)/right/MPNet(MoE)(AC)80.pt"
            elif MPNet_MODEL == 'MLP':    
                MPNet_path = "model(trained)/right/MPNet(MLP)(AC)80.pt"
            elif MPNet_MODEL == 'Transformer':    
                MPNet_path = "model(trained)/right/MPNet(Transformer)(AC)80.pt" 
            # UGNet    
            if UGNet_MODEL == 'MoE':    
                UGNet_path = "model(trained)/right/UGNet(MoE)(AC)120.pt"                             
            elif UGNet_MODEL == 'MLP':    
                UGNet_path = "model(trained)/right/UGNet(MLP)(AC)120.pt"                                                         
            elif UGNet_MODEL == 'Transformer':    
                UGNet_path = "model(trained)/right/UGNet(Transformer)(AC)120.pt"                             
            elif UGNet_MODEL == 'Diffusion':    
                UGNet_path = "model(trained)/right/UGNet(Diffusion)(AC)120.pt"                             
                
        elif MOTION_CATEGORY == 'DG':
            MPNet_path = "model(trained)/right/MPNet(MoE)(DG)80.pt"
            UGNet_path = "model(trained)/right/UGNet(MoE)(DG)120.pt"            
            
    elif SUBJECT == '173' or SUBJECT == '174':
        if MOTION_CATEGORY == 'AC':
            # MPNet
            if MPNet_MODEL == 'MoE':    
                MPNet_path = "model(trained)/left/MPNet(MoE)(AC)80.pt"
            elif MPNet_MODEL == 'MLP':    
                MPNet_path = "model(trained)/left/MPNet(MLP)(AC)80.pt"
            elif MPNet_MODEL == 'Transformer':    
                MPNet_path = "model(trained)/left/MPNet(Transformer)(AC)80.pt" 
            # UGNet    
            if UGNet_MODEL == 'MoE':    
                UGNet_path = "model(trained)/left/UGNet(MoE)(AC)120.pt"                             
            elif UGNet_MODEL == 'MLP':    
                UGNet_path = "model(trained)/left/UGNet(MLP)(AC)120.pt"     
            elif UGNet_MODEL == 'Transformer':    
                UGNet_path = "model(trained)/left/UGNet(Transformer)(AC)120.pt"                             
            elif UGNet_MODEL == 'Diffusion':    
                UGNet_path = "model(trained)/left/UGNet(Diffusion)(AC)120.pt"    
                
        elif  MOTION_CATEGORY == 'DG':
            MPNet_path = "model(trained)/left/MPNet(MoE)(DG)80.pt"
            UGNet_path = "model(trained)/left/UGNet(MoE)(DG)120.pt"

    WINDOW_SIZE = 30
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # MPNet Load    
    if MPNet_MODEL == 'MoE':    
        MPNet = MPNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=6+5, hidden_size=24, output_size=3, use_cuda=True)
    elif MPNet_MODEL == 'MLP':    
        MPNet = MPNet_MLP(input_size=6+5, hidden_size=24, output_size=3, window_size=WINDOW_SIZE, use_cuda=True)       
    elif MPNet_MODEL == 'Transformer':    
        MPNet = MPNet_Transformer(input_size=6+5, hidden_size=16, output_size=3, use_cuda=True)               
    MPNet.load_state_dict(torch.load(MPNet_path, map_location=device))
    MPNet.eval()

    # UGNet Load
    if UGNet_MODEL == 'MoE':    
        UGNet = UGNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=24+(5+3), output_size=24, window_size=WINDOW_SIZE, use_cuda=True)
    elif UGNet_MODEL == 'MLP':
        UGNet = UGNet_MLP(rng=np.random.RandomState(23456), num_experts=9, input_size=24+(5+3), output_size=24, window_size=WINDOW_SIZE, use_cuda=True)        
    elif UGNet_MODEL == 'Transformer':    
        UGNet = UGNet_Transformer(input_size=24+(5+3), output_size=24, use_cuda=True)
    elif UGNet_MODEL == 'Diffusion':    
        UGNet = UGNet_Diffusion(T = 30)
    UGNet.load_state_dict(torch.load(UGNet_path, map_location=device))
    UGNet.eval()

    if METRIC == 'DIP': # Deictic Intention Preservation
        if not os.path.exists('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')'):
            os.makedirs('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')')        
            
        REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                       {'repertorie': 'Touching single target', 'num': 30}]        
        DATA_PARSER = data_parser('', REPERTORIES, WINDOW_SIZE, 45, '', 0)
        DATA_PARSER.raw_data_dir = DIR            
        DATA_PARSER.raw_data_test_pairing_dir = 'data(raw)/'+'test_pairing/' + SUBJECT + '/' 
                    
        runner = Runner(directory_id=DIRECTORY_ID, device=device, MPNet=MPNet, UGNet=UGNet, scale=SCALE)                             
        with torch.no_grad():
            for repertorie in DATA_PARSER.repertories:
                folder_split = DATA_PARSER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                  
                test_pairing_folder_split = DATA_PARSER.raw_data_test_pairing_dir + repertorie['repertorie'] + '/' + 'split'                
                for j in range(repertorie['num']):
                    for i in range(repertorie['num']):
                        local, remote = DATA_PARSER.interpolate_evaluation_DIP(test_pairing_folder_split, folder_split, i+1, j+1)                    
                        runner.run_DIP(local, remote)      
                if repertorie['repertorie'].split(' ')[0] == 'Touching':
                    runner.result_DIP(True)
                else:
                    runner.result_DIP(False)
                runner.iter = 0
                runner.Horizontal_Deviation = 0
                runner.Vertical_Deviation = 0
                runner.Position_Deviation = 0          
        
    elif METRIC == 'MN': # Frechet Motion Distance (FMD), Movement Natrualness
        if not os.path.exists('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')'):
            os.makedirs('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')')        
            
        REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                       {'repertorie': 'Pointing at two targets', 'num': 8},
                       {'repertorie': 'Pointing at single target with gaze shift', 'num': 8},
                       {'repertorie': 'Pointing at single target with explanation', 'num': 8},
                       {'repertorie': 'Transition (Pointing)', 'num': 27},
                       {'repertorie': 'Touching single target', 'num': 27},
                       {'repertorie': 'Touching single target with gaze shift', 'num': 8},
                       {'repertorie': 'Touching single target with explanation', 'num': 8},
                       {'repertorie': 'Touching two targets', 'num': 8},
                       {'repertorie': 'Transition (Touching)', 'num': 27}]        
                
        Frechet_Motion_Distance_List = []            
        for trial in range(10):
            DATA_PARSER = data_parser('', REPERTORIES, WINDOW_SIZE, 45, '', 0)
            DATA_PARSER.raw_data_dir = DIR            
            DATA_PARSER.raw_data_test_pairing_dir = 'data(raw)/'+'test_pairing/' + SUBJECT + '/'
                                  
            runner = Runner(directory_id=DIRECTORY_ID, device=device, MPNet=MPNet, UGNet=UGNet, scale=SCALE)                             
            for d in [runner.Generated_Motion_Dir, runner.Real_Motion_Dir]:    
                for f in os.listdir(d):
                    os.remove(os.path.join(d, f))
            
            print("1. Inference and save generated motion")
            with torch.no_grad():
                for repertorie in DATA_PARSER.repertories:
                    folder_split = DATA_PARSER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'
                    test_pairing_folder_split = DATA_PARSER.raw_data_test_pairing_dir + repertorie['repertorie'] + '/' + 'split'                
                    for i in range(repertorie['num']):
                        local, remote = DATA_PARSER.interpolate_evaluation_MN(test_pairing_folder_split, folder_split, i+1, random.randint(1, repertorie['num']))
                        runner.run_MN(local, remote)           
            
            print("2. Compute embeddings (Auto Encoder)")        
            autoencoder = ConvolutionalAutoEncoder(input_size=18, hidden_size=256, output_size=18, kernel_size=15)
            autoencoder = autoencoder.to(device)
            if SUBJECT == '161' or SUBJECT == '172' or SUBJECT == '180':
                autoencoder_path = "model(trained)/right/AutoEncoder600.pt"
            elif SUBJECT == '173' or SUBJECT == '174':
                autoencoder_path = "model(trained)/left/AutoEncoder600.pt"
            autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
            autoencoder = autoencoder.to(device)
            autoencoder.eval()
            
            real_motion_set = Dataset(folder_name=runner.Real_Motion_Dir) 
            real_motion_set_loader = DataLoader(real_motion_set, batch_size=1, shuffle=False, drop_last=False)            
            real_embeddings = runner.compute_Embedding(autoencoder, real_motion_set_loader)
            generated_motion_set = Dataset(folder_name=runner.Generated_Motion_Dir) 
            generated_motion_set_loader = DataLoader(generated_motion_set, batch_size=1, shuffle=False, drop_last=False)            
            generated_embeddings = runner.compute_Embedding(autoencoder, generated_motion_set_loader)            
            
            print("3. Calculate FMD")
            runner.calculate_FMD(real_embeddings, generated_embeddings)
            runner.result_MN()      
            Frechet_Motion_Distance_List.append(runner.Frechet_Motion_Distance)
                  
        with open('Quantitative Evaluation('+MOTION_CATEGORY+')'+'(MPNet_'+MPNet_MODEL+')'+'(UGNet_'+UGNet_MODEL+')'+"/MN"+str(SUBJECT)+".txt", "w") as file:
            for Frechet_Motion_Distance in Frechet_Motion_Distance_List:
                file.write(str(Frechet_Motion_Distance)+"\n")   

    elif METRIC == 'MT': # Missing Tracking (FrameSkip)
       if not os.path.exists('Quantitative Evaluation(Missing Tracking)'):
           os.makedirs('Quantitative Evaluation(Missing Tracking)')
           
       REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                      {'repertorie': 'Pointing at two targets', 'num': 8},
                      {'repertorie': 'Pointing at single target with gaze shift', 'num': 8},
                      {'repertorie': 'Pointing at single target with explanation', 'num': 8},
                      {'repertorie': 'Transition (Pointing)', 'num': 27},
                      {'repertorie': 'Touching single target', 'num': 27},
                      {'repertorie': 'Touching single target with gaze shift', 'num': 8},
                      {'repertorie': 'Touching single target with explanation', 'num': 8},
                      {'repertorie': 'Touching two targets', 'num': 8},
                      {'repertorie': 'Transition (Touching)', 'num': 27}]        
               
       num_frame_drop_list = [6, 12, 18, 24, 30]  # 0.2s, 0.4s, 0.6s, 0.8s, 1.0s
       results_file = f'Quantitative Evaluation(Missing Tracking)/MT{SUBJECT}.txt'
        
       with open(results_file, "w") as f:
           f.write("Missing Tracking Analysis Results\n\n")       
           
       for num_frame_drop in num_frame_drop_list:
           print(f"\nAnalyzing {num_frame_drop/30:.1f}s frame drop")
           DATA_PARSER = data_parser('', REPERTORIES, WINDOW_SIZE, 45, '', 0)
           DATA_PARSER.raw_data_dir = DIR            
           DATA_PARSER.raw_data_test_pairing_dir = 'data(raw)/'+'test_pairing/' + SUBJECT + '/'
           runner = Runner(directory_id=DIRECTORY_ID, device=device, MPNet=MPNet, UGNet=UGNet, scale=SCALE)            
           
           print(f"Testing with frame drop: {num_frame_drop}")
           with torch.no_grad():
               metrics_list = []
               for repertorie in DATA_PARSER.repertories:
                   folder_split = DATA_PARSER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'
                   test_pairing_folder_split = DATA_PARSER.raw_data_test_pairing_dir + repertorie['repertorie'] + '/' + 'split'                
                   for j in range(repertorie['num']):
                       for i in range(repertorie['num']):
                           local, remote = DATA_PARSER.interpolate_evaluation_MN(test_pairing_folder_split, folder_split, i+1, j+1)                    
                           local_noisy = runner.missingtracking(local, num_frame_drop)
                           clean_motion = runner.run_missingtracking(local, remote)                   
                           noisy_motion = runner.run_missingtracking(local_noisy, remote)
                           
                           position_error, rot_error, vel_error, ang_vel_error = runner.calculate_motion_metrics(clean_motion, noisy_motion)
                           position_error *= 100  # m to cm
                           vel_error *= 100  # m/s to cm/s
                           
                           metrics = np.array([position_error.cpu().numpy(), rot_error.cpu().numpy(), vel_error.cpu().numpy(), ang_vel_error.cpu().numpy()])
                           metrics_list.append(metrics)
                           
           metrics = np.mean(metrics_list, axis=0)
           print(f"\nFrame Drop Duration: {num_frame_drop/30:.1f}s ({num_frame_drop} frames)")
           print("-" * 50)
           print(f"Position Error: {metrics[0]:.3f}cm")
           print(f"Rotation Error: {metrics[1]:.3f}deg")
           print(f"Linear Velocity Error: {metrics[2]:.3f}cm/s")
           print(f"Angular Velocity Error: {metrics[3]:.3f}deg/s\n")
           
           with open(results_file, "a") as f:
               f.write(f"\nFrame Drop Duration: {num_frame_drop/30:.1f}s ({num_frame_drop} frames)\n")
               f.write("-" * 50 + "\n")
               f.write(f"Position Error: {metrics[0]:.3f}cm\n")
               f.write(f"Rotation Error: {metrics[1]:.3f}deg\n")
               f.write(f"Linear Velocity Error: {metrics[2]:.3f}cm/s\n")
               f.write(f"Angular Velocity Error: {metrics[3]:.3f}deg/s\n\n")
                
    elif METRIC == 'Responsiveness': # Dynamic Responsiveness (DR) 
        if not os.path.exists('Quantitative Evaluation(Responsiveness)'+'(UGNet_'+UGNet_MODEL+')'):
            os.makedirs('Quantitative Evaluation(Responsiveness)'+'(UGNet_'+UGNet_MODEL+')')        
        
        REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 14},
                       {'repertorie': 'Touching single target', 'num': 29}]
        
        pointing_initial_pose = torch.tensor([-1.8990e-02, 8.4800e-03, 9.9978e-01, 3.1110e-02, 9.9000e-04, 9.9952e-01, -4.7570e-02, 9.9887e-01, 4.9000e-04, 1.6800e-03, -3.9000e-04, 1.3860e-02, -9.7802e-01, 6.1410e-02, -1.9927e-01, 1.9061e-01, 6.5082e-01, -7.3491e-01, 4.1870e-02, -7.2110e-02, 4.1893e-01, 1.3511e-01, -5.5810e-02, 1.7470e-02])
        pointing_initial_target_feature = torch.tensor([-5.5100e-03, 1.5714e+00, -5.5100e-03, 1.5714e+00, 1.0000e+00])
        pointing_changed_target_feature = torch.tensor([[1.2350e+00, 7.8449e-01, 1.2350e+00, 7.8449e-01, 1.0000e+00],
                                                        [6.0649e-01, 1.1881e+00, 6.0649e-01, 1.1881e+00, 1.0000e+00],
                                                        [-8.1100e-03, 1.2493e+00, -8.1100e-03, 1.2493e+00, 1.0000e+00],
                                                        [-6.1784e-01, 1.1887e+00, -6.1784e-01, 1.1887e+00, 1.0000e+00],
                                                        [-1.2276e+00, 7.9224e-01, -1.2276e+00, 7.9224e-01, 1.0000e+00],
                                                        [1.1942e+00, 1.5712e+00, 1.1942e+00, 1.5712e+00, 1.0000e+00],
                                                        [6.3254e-01, 1.5710e+00, 6.3254e-01, 1.5710e+00, 1.0000e+00],
                                                        [-6.0768e-01, 1.5710e+00, -6.0768e-01, 1.5710e+00, 1.0000e+00],
                                                        [-1.2365e+00, 1.5708e+00, -1.2365e+00, 1.5708e+00, 1.0000e+00],
                                                        [1.1834e+00, 2.2935e+00, 1.1834e+00, 2.2935e+00, 1.0000e+00],
                                                        [6.2643e-01, 1.9598e+00, 6.2643e-01, 1.9598e+00, 1.0000e+00],
                                                        [-1.6200e-03, 1.8928e+00, -1.6200e-03, 1.8928e+00, 1.0000e+00],
                                                        [-6.1699e-01, 1.9555e+00, -6.1699e-01, 1.9555e+00, 1.0000e+00],
                                                        [-1.2323e+00, 2.3540e+00, -1.2323e+00, 2.3540e+00, 1.0000e+00]])
        pointing_changed_target_position = torch.tensor([[-1.43013, 0.50007, 0.49916],
                                                         [-0.86069, 0.49952, 1.24072],
                                                         [0.01220, 0.50090, 1.50376],
                                                         [0.88048, 0.49782, 1.23897],
                                                         [1.41698, 0.49948, 0.50637],
                                                         [-1.37875, -0.00024, 0.54530],
                                                         [-0.90666, -0.00030, 1.23689],
                                                         [0.86342, -0.00021, 1.24148],
                                                         [1.41825, 0.00001, 0.49265],
                                                         [-1.38251, -0.49742, 0.56412],
                                                         [-0.88287, -0.50004, 1.22001],
                                                         [0.00244, -0.50072, 1.50072],
                                                         [0.87253, -0.49805, 1.22999],
                                                         [1.42513, -0.49949, 0.50172]])
        
        touching_initial_pose = torch.tensor([-8.0500e-02, -1.7869e-01, 9.8061e-01, 3.6170e-02, -3.9586e-01, 9.1760e-01, -4.5230e-02, 9.1661e-01, 3.9722e-01, 1.6990e-02, -5.9930e-02, 4.9860e-02, -9.0291e-01, 9.7980e-02, -4.1852e-01, 3.9135e-01, 5.9011e-01, -7.0613e-01, 2.8850e-02, -1.7900e-01, 3.0297e-01, 1.4588e-01, -4.1850e-02, 2.0850e-02])        
        touching_initial_target_feature = torch.tensor([-6.9000e-04, 2.2163e+00, -6.9000e-04, 2.2163e+00, 4.9716e-01])

        touching_changed_target_feature = torch.tensor([[1.2735e+00, 1.5212e+00, 1.2735e+00, 1.5212e+00, 4.1215e-01],
                                                        [6.3665e-01, 1.5642e+00, 6.3665e-01, 1.5642e+00, 4.1641e-01],
                                                        [-2.6980e-02, 1.5697e+00, -2.6980e-02, 1.5697e+00, 4.0458e-01],
                                                        [-6.1739e-01, 1.5666e+00, -6.1739e-01, 1.5666e+00, 3.9731e-01],
                                                        [-1.2589e+00, 1.5722e+00, -1.2589e+00, 1.5722e+00, 3.9758e-01],
                                                        [1.1935e+00, 2.6910e+00, 1.1935e+00, 2.6910e+00, 4.9759e-01],
                                                        [6.3275e-01, 2.3157e+00, 6.3275e-01, 2.3157e+00, 5.0237e-01],
                                                        [-6.3575e-01, 2.3277e+00, -6.3575e-01, 2.3277e+00, 4.9291e-01],
                                                        [-1.2635e+00, 2.7487e+00, -1.2635e+00, 2.7487e+00, 5.0482e-01],
                                                        [1.1577e+00, 2.8848e+00, 1.1577e+00, 2.8848e+00, 7.2241e-01],
                                                        [6.2961e-01, 2.6496e+00, 6.2961e-01, 2.6496e+00, 7.1837e-01],
                                                        [-1.7040e-02, 2.5603e+00, -1.7040e-02, 2.5603e+00, 7.1624e-01],
                                                        [-6.8532e-01, 2.6729e+00, -6.8532e-01, 2.6729e+00, 7.1278e-01],
                                                        [-1.2566e+00, 2.9442e+00, -1.2566e+00, 2.9442e+00, 7.1013e-01],
                                                        [1.2205e+00, 1.5683e+00, 1.2205e+00, 1.5683e+00, 5.1076e-01],
                                                        [6.3835e-01, 1.5682e+00, 6.3835e-01, 1.5682e+00, 4.8972e-01],
                                                        [3.9200e-03, 1.5721e+00, 3.9200e-03, 1.5721e+00, 4.8847e-01],
                                                        [-6.2215e-01, 1.5688e+00, -6.2215e-01, 1.5688e+00, 4.9597e-01],
                                                        [-1.2663e+00, 1.5616e+00, -1.2663e+00, 1.5616e+00, 5.0052e-01],
                                                        [1.2340e+00, 2.6347e+00, 1.2340e+00, 2.6347e+00, 5.8662e-01],
                                                        [6.3902e-01, 2.2235e+00, 6.3902e-01, 2.2235e+00, 5.7324e-01],
                                                        [-2.0140e-02, 2.1165e+00, -2.0140e-02, 2.1165e+00, 5.8331e-01],
                                                        [-6.5745e-01, 2.2216e+00, -6.5745e-01, 2.2216e+00, 5.7953e-01],
                                                        [-1.2154e+00, 2.6158e+00, -1.2154e+00, 2.6158e+00, 5.8231e-01],
                                                        [1.1858e+00, 2.8409e+00, 1.1858e+00, 2.8409e+00, 7.8477e-01],
                                                        [5.9104e-01, 2.5426e+00, 5.9104e-01, 2.5426e+00, 7.7569e-01],
                                                        [9.8900e-03, 2.4587e+00, 9.8900e-03, 2.4587e+00, 7.7379e-01],
                                                        [-6.1132e-01, 2.5451e+00, -6.1132e-01, 2.5451e+00, 7.7764e-01],
                                                        [-1.2538e+00, 2.8915e+00, -1.2538e+00, 2.8915e+00, 7.7248e-01]])
        touching_changed_target_position = torch.tensor([[-0.39403, 0.00599, 0.12071],
                                                         [-0.24755, 0.00220, 0.33482],
                                                         [0.01091, 0.00044, 0.40444],
                                                         [0.23001, 0.00136, 0.32396],
                                                         [0.37840, -0.00017, 0.12199],
                                                         [-0.36801, -0.30148, 0.14585],
                                                         [-0.23840, -0.29977, 0.32509],
                                                         [0.23302, -0.29825, 0.31577],
                                                         [0.38864, -0.29765, 0.12334],
                                                         [-0.36217, -0.60457, 0.15875],
                                                         [-0.23376, -0.59871, 0.32087],
                                                         [0.00670, -0.59860, 0.39325],
                                                         [0.24691, -0.59654, 0.30204],
                                                         [0.36689, -0.59620, 0.11923],
                                                         [-0.47975, 0.00043, 0.17526],
                                                         [-0.29181, 0.00102, 0.39329],
                                                         [-0.00192, -0.00063, 0.48847],
                                                         [0.28904, 0.00080, 0.40304],
                                                         [0.47749, 0.00138, 0.15007],
                                                         [-0.47579, -0.30000, 0.16658],
                                                         [-0.29140, -0.29982, 0.39217],
                                                         [0.01004, -0.30272, 0.49851],
                                                         [0.30332, -0.29915, 0.39289],
                                                         [0.46819, -0.29947, 0.17377],
                                                         [-0.46304, -0.60518, 0.18765],
                                                         [-0.27450, -0.59918, 0.40905],
                                                         [-0.00483, -0.60028, 0.48826],
                                                         [0.28489, -0.59862, 0.40647],
                                                         [0.46520, -0.59751, 0.15261]]) 
        
        runner = Runner(directory_id=DIRECTORY_ID, device=device, MPNet=MPNet, UGNet=UGNet, scale=1)                             
        with torch.no_grad():
            for repertorie in REPERTORIES:
                for i in range(repertorie['num']):
                    if repertorie['repertorie'].split(' ')[0] == 'Touching':
                        runner.run_Responsiveness(touching_initial_pose, touching_initial_target_feature, touching_changed_target_feature[i], touching_changed_target_position[i])
                    else :
                        runner.run_Responsiveness(pointing_initial_pose, pointing_initial_target_feature, pointing_changed_target_feature[i], pointing_changed_target_position[i])                    
                        
                if repertorie['repertorie'].split(' ')[0] == 'Touching':
                    runner.result_Responsiveness(True)
                else:
                    runner.result_Responsiveness(False)
                    
                runner.iter = 0
                runner.Horizontal_Deviation = 0
                runner.Vertical_Deviation = 0
                runner.Position_Deviation = 0  