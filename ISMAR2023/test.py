import os
import os.path as osp
import gc
import random
import numpy as np
import scipy.linalg as linalg
import datetime 
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import ISMAR2023, ConvolutionalAutoEncoder
from data_pairer import data_pairer

class Runner(object):                                              
    def __init__(self, gpu_id, device, model, scale):               
        self.device = device
        self.model = model.to(self.device)
        self.scale = scale
        self.iter = 0        
        self.Horizontal_Deviation = 0
        self.Vertical_Deviation = 0
        self.Position_Deviation = 0
        self.Frechet_Motion_Distance = 0
        self.Generated_Motion_Dir = 'data/'+'test(Generated)_'+str(gpu_id)+'/'
        self.Real_Motion_Dir = 'data/'+'test(Real)_'+str(gpu_id)+'/'
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
                        
    def run_DIP(self, local, remote, touch): # Deictic Intention Preservation
        local = local.to(self.device)
        remote = remote.to(self.device)
            
        local = local.unsqueeze(0) # (1, T, :)
        remote = remote.unsqueeze(0) # (1, T, :)
        
        Local = local[:,-W:,0:15] # (1, 15)      
        Remote = remote[:,-W-1:-1,0:15] # (1, 15)      
        if touch==False: 
            Local = torch.cat((Local, torch.tensor(1).repeat(Local.shape[1]).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                             
            Remote = torch.cat((Remote, torch.tensor(1).repeat(Remote.shape[1]).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                             
        else:
            Local = torch.cat((Local, torch.tensor(self.scale).repeat(Local.shape[1]).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                             
            Remote = torch.cat((Remote, torch.tensor(self.scale).repeat(Remote.shape[1]).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                                     
            
        output_data = self.model(Local, Remote)
        
        if touch==False:
            output_data[0,6:9] = self.scale * output_data[0,6:9]
            output_data[0,15:21] = self.scale * output_data[0,15:21]

        # 1. EFRC (Prediction), InderFingerTip (Local -> Global)
        Head_pos = output_data[0,6:9]
        RightHand_rot = torch.squeeze(self.sixd2matrot(output_data[:,9:15]))
        RightHand_pos = torch.squeeze(output_data[0,15:18])
        IndexFingerTip_pos = torch.matmul(RightHand_rot, output_data[0,-3:])+RightHand_pos        
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
            with open("Quantitative Evaluation/DIP"+str(SUBJECT)+"(Pointing).txt", "w") as file:
                file.write(log)
        else:
            log = "Touching, " + str(self.iter).zfill(5) + ", [PD: %.3f (cm)] " % (self.Position_Deviation/self.iter*100)
            with open("Quantitative Evaluation/DIP"+str(SUBJECT)+"(Touching).txt", "w") as file:
                file.write(log)            
        print(log) 
        
    def run_MN(self, local, remote, touch): # Movement Naturalness
        local = local.to(self.device)
        remote = remote.to(self.device)
        
        local = local.unsqueeze(0) # (1, T, :)
        remote = remote.unsqueeze(0) # (1, T, :)
                
        generated_list = []
        T = local.shape[1]-W
        
        for i in range(T):              
            if touch==False: 
                Local = torch.cat((local[:, i+1:W+i+1, 0:15], torch.tensor(1).repeat(W).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                             
                Remote = torch.cat((remote[:, i+0:W+i+0, 0:15], torch.tensor(1).repeat(W).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                                         
            else:
                Local = torch.cat((local[:, i+1:W+i+1, 0:15], torch.tensor(self.scale).repeat(W).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                             
                Remote = torch.cat((remote[:, i+0:W+i+0, 0:15], torch.tensor(self.scale).repeat(W).unsqueeze(0).unsqueeze(-1).to(self.device)), -1)                                         
            
            output_data = self.model(Local, Remote) # (1,21)
            if touch==False:
                output_data[0,6:9] = self.scale * output_data[0,6:9]
                output_data[0,15:21] = self.scale * output_data[0,15:21]
            
            # Get Ray
            Head_pos = output_data[0,6:9]
            RightHand_rot = torch.squeeze(self.sixd2matrot(output_data[:,9:15]))
            RightHand_pos = torch.squeeze(output_data[0,15:18])
            IndexFingerTip_pos = torch.matmul(RightHand_rot, output_data[0,-3:])+RightHand_pos        
            Target_pos = remote[0,W+i+0,36:39]
            H2T = Target_pos - Head_pos
            HF = output_data[0,:3]
            H2I = IndexFingerTip_pos - Head_pos
            
            # Auto-regression
            # (H2T,HF)
            remote[0, W+i, 0] = self.SignedAngle(F.normalize(H2T * torch.tensor([1,0,1]).to(self.device), dim=0), F.normalize(HF * torch.tensor([1,0,1]).to(self.device), dim=0), torch.tensor([0,1,0]).to(self.device))
            remote[0, W+i, 1] = self.SignedAngle(F.normalize(H2T * torch.tensor([0,1,1]).to(self.device), dim=0), F.normalize(HF * torch.tensor([0,1,1]).to(self.device), dim=0), torch.tensor([1,0,0]).to(self.device))
            # (H2T,H2I) 
            remote[0, W+i, 2] = self.SignedAngle(F.normalize(H2T * torch.tensor([1,0,1]).to(self.device), dim=0), F.normalize(H2I * torch.tensor([1,0,1]).to(self.device), dim=0), torch.tensor([0,1,0]).to(self.device))
            remote[0, W+i, 3] = self.SignedAngle(F.normalize(H2T * torch.tensor([0,1,1]).to(self.device), dim=0), F.normalize(H2I * torch.tensor([0,1,1]).to(self.device), dim=0), torch.tensor([1,0,0]).to(self.device))            
            # (HF,H2I) 
            remote[0, W+i, 4] = self.SignedAngle(F.normalize(HF * torch.tensor([1,0,1]).to(self.device), dim=0), F.normalize(H2I * torch.tensor([1,0,1]).to(self.device), dim=0), torch.tensor([0,1,0]).to(self.device))
            remote[0, W+i, 5] = self.SignedAngle(F.normalize(HF * torch.tensor([0,1,1]).to(self.device), dim=0), F.normalize(H2I * torch.tensor([0,1,1]).to(self.device), dim=0), torch.tensor([1,0,0]).to(self.device))
            # H2T
            remote[0, W+i, 6] = torch.arccos(torch.clip(torch.dot(F.normalize(H2T * torch.tensor([1,0,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([1,0,0], dtype=torch.float32).to(self.device)), -1.0, 1.0))
            remote[0, W+i, 7] = torch.arccos(torch.clip(torch.dot(F.normalize(H2T * torch.tensor([0,1,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([0,-1,0], dtype=torch.float32).to(self.device)), -1.0, 1.0)) 
            # H2I
            remote[0, W+i, 8] = torch.arccos(torch.clip(torch.dot(F.normalize(H2I * torch.tensor([1,0,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([1,0,0], dtype=torch.float32).to(self.device)), -1.0, 1.0))
            remote[0, W+i, 9] = torch.arccos(torch.clip(torch.dot(F.normalize(H2I * torch.tensor([0,1,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([0,-1,0], dtype=torch.float32).to(self.device)), -1.0, 1.0))  
            # HF
            remote[0, W+i,10] = torch.arccos(torch.clip(torch.dot(F.normalize(HF * torch.tensor([1,0,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([1,0,0], dtype=torch.float32).to(self.device)), -1.0, 1.0))
            remote[0, W+i,11] = torch.arccos(torch.clip(torch.dot(F.normalize(HF * torch.tensor([0,1,1], dtype=torch.float32).to(self.device), dim=0), torch.tensor([0,-1,0], dtype=torch.float32).to(self.device)), -1.0, 1.0))  
            
            Head_rot = torch.squeeze(self.sixd2matrot(output_data[:,0:6]))
            Head_rot_inv = torch.inverse(Head_rot)
            Head_pos = output_data[0,6:9]                        
            
            RightHand_forward_wrt_head = torch.matmul(Head_rot_inv, output_data[0,9:12])
            RightHand_up_wrt_head = torch.matmul(Head_rot_inv, output_data[0,12:15])
            RightHand_position_wrt_head = torch.matmul(Head_rot_inv, output_data[0,15:18]) - torch.matmul(Head_rot_inv, Head_pos)
            
            output_data[0,9:12] = RightHand_forward_wrt_head
            output_data[0,12:15] = RightHand_up_wrt_head
            output_data[0,15:18] = RightHand_position_wrt_head
            
            # Save            
            generated_list.append(torch.squeeze(output_data[0,0:18]))
            
        real = local[0,W:,15:33]
        generated = torch.stack(generated_list)
        # Save
        np.save(osp.join(self.Real_Motion_Dir, str(self.count)), real[-self.L:,:].cpu().numpy())        
        np.save(osp.join(self.Generated_Motion_Dir, str(self.count)), generated[-self.L:,:].cpu().numpy())        
        self.count = self.count + 1
        
        real = torch.transpose(real.unfold(0, self.L, int(self.L/2)), 1, 2)
        generated = torch.transpose(generated.unfold(0, self.L, int(self.L/2)), 1, 2)
        for j in range(real.shape[0]):
            np.save(osp.join(self.Real_Motion_Dir, str(self.count)), real[j].cpu().numpy())
            np.save(osp.join(self.Generated_Motion_Dir, str(self.count)), generated[j].cpu().numpy())            
            self.count = self.count + 1  

    def result_MN(self):
        log = "[FMD: %.3f]" % (self.Frechet_Motion_Distance)
        print(log)  

class Dataset(Dataset):
    def __init__(self, dtype=None, folder_name=None, gpu_id=None, **kwargs):
        self.sequences = len(os.listdir(folder_name))        
        self.paths = [folder_name+str(i)+'.npy' for i in range(self.sequences)]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):        
        data = torch.from_numpy(np.load(self.paths[idx]))        
        return {'data': data} 
                    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default='173', help='Directory')
parser.add_argument('--metric', default='DIP', help='Metric')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
args = parser.parse_args()

# python test.py --subject '161' --metric 'MN'; python test.py --subject '172' --metric 'MN'; python test.py --subject '180' --metric 'MN'; python test.py --subject '173' --metric 'MN'; python test.py --subject '174' --metric 'MN';

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')
    
    SUBJECT = args.subject
    METRIC = args.metric    
    GPU_ID = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    W = 20
    DIR = 'data(raw)/test/' + SUBJECT + '/'
        
    if not os.path.exists('Quantitative Evaluation'):
        os.makedirs('Quantitative Evaluation')  
        
    SCALE=1
    if SUBJECT == '161':
        SCALE = 0.9471
    elif SUBJECT == '172':
        SCALE = 1.0118
    elif SUBJECT == '180':
        SCALE = 1.0588
    elif SUBJECT == '173':
        SCALE = 1.0176
    elif SUBJECT == '174':    
        SCALE = 1.0235
        
    if SUBJECT == '161' or SUBJECT == '172' or SUBJECT == '180':
        path = "model(trained)/right/ISMAR2023(Hard)629.pt"
    elif SUBJECT == '173' or SUBJECT == '174':
        path = "model(trained)/left/ISMAR2023(Hard)629.pt"
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')    
    model = ISMAR2023(rng=np.random.RandomState(23456), num_experts=6, input_size=16, output_size=21, use_cuda=True, runtime=False)        
    model.load_state_dict(torch.load(path, map_location=device))    
    model.eval()
    
    if METRIC == 'DIP':
        REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                       {'repertorie': 'Touching single target', 'num': 30}]        
        DATA_PAIRER = data_pairer('', REPERTORIES, 30, 20, GPU_ID) 
        DATA_PAIRER.raw_data_dir = DIR
        DATA_PAIRER.split()        
        runner = Runner(gpu_id=GPU_ID, device=device, model=model, scale=SCALE)        
        with torch.no_grad():
            for repertorie in DATA_PAIRER.repertories:
                folder_split = DATA_PAIRER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                  
                for j in range(repertorie['num']):
                    for i in range(repertorie['num']):
                        local, remote = DATA_PAIRER.interpolate_evaluation_DIP(folder_split, folder_split, i+1, j+1)                    
                        if repertorie['repertorie'].split(' ')[0] == 'Touching':
                            runner.run_DIP(local, remote, True)    
                        else:
                            runner.run_DIP(local, remote, False)             
                if repertorie['repertorie'].split(' ')[0] == 'Touching':
                    runner.result_DIP(True)
                else:
                    runner.result_DIP(False)
                runner.iter = 0
                runner.Horizontal_Deviation = 0
                runner.Vertical_Deviation = 0
                runner.Position_Deviation = 0          
        DATA_PAIRER.delete()

    elif METRIC == 'MN':
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
            DATA_PAIRER = data_pairer('', REPERTORIES, 30, 20, GPU_ID)
            DATA_PAIRER.raw_data_dir = DIR
            DATA_PAIRER.split()
            
            runner = Runner(gpu_id=GPU_ID, device=device, model=model, scale=SCALE)                     
            for d in [runner.Generated_Motion_Dir, runner.Real_Motion_Dir]:    
                for f in os.listdir(d):    
                    os.remove(os.path.join(d, f))
        
            print("1. Inference and save generated motion")
            with torch.no_grad():
                for repertorie in DATA_PAIRER.repertories:
                    folder_split = DATA_PAIRER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                  
                    for i in range(repertorie['num']):
                        local, remote = DATA_PAIRER.interpolate_evaluation_MN(folder_split, folder_split, i+1, random.randint(1, repertorie['num']))
                        if repertorie['repertorie'].split(' ')[0] == 'Touch' or repertorie['repertorie'] == 'Transition (Touching)':
                            runner.run_MN(local, remote, True)
                        else:
                            runner.run_MN(local, remote, False)                        
            
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
                    
            generated_motion_set = Dataset(folder_name=runner.Generated_Motion_Dir, gpu_id=GPU_ID) 
            generated_motion_set_loader = DataLoader(generated_motion_set, batch_size=1, shuffle=False, drop_last=False)
            real_motion_set = Dataset(folder_name=runner.Real_Motion_Dir, gpu_id=GPU_ID) 
            real_motion_set_loader = DataLoader(real_motion_set, batch_size=1, shuffle=False, drop_last=False)
            
            generated_embeddings = runner.compute_Embedding(autoencoder, generated_motion_set_loader)
            real_embeddings = runner.compute_Embedding(autoencoder, real_motion_set_loader)
            
            print("3. Calculate FID")
            runner.calculate_FMD(real_embeddings, generated_embeddings)
            runner.result_MN()                    
            Frechet_Motion_Distance_List.append(runner.Frechet_Motion_Distance)
                  
        with open("Quantitative Evaluation/MN"+str(SUBJECT)+".txt", "w") as file:
            for Frechet_Motion_Distance in Frechet_Motion_Distance_List:
                file.write(str(Frechet_Motion_Distance)+"\n")   