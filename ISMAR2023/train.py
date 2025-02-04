import os
import gc
import time
import datetime 
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import ISMAR2023, Discriminator
from data_pairer import data_pairer
from data_parser import Dataset

class Runner(object):                                              
    def __init__(self, device, model, discriminator_shorterm, discriminator_longterm, lr):               
        self.device = device
        self.model = model.to(self.device)
        self.discriminator_shorterm = discriminator_shorterm.to(self.device)        
        self.discriminator_longterm = discriminator_longterm.to(self.device)
        self.optimizer_D = torch.optim.SGD(list(self.discriminator_shorterm.parameters())+list(self.discriminator_longterm.parameters()), lr=lr, momentum=0.8)
        self.optimizer_G = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.8)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma= 0.999)
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma= 0.999)       
        self.learning_rate = lr
        self.iter = 0
        
    def matrot2sixd(self, pose_matrot):        
        return torch.cat([pose_matrot[:,:,:,:,2], pose_matrot[:,:,:,:,1]], -1) # (forward, up)

    def sixd2matrot(self, pose_6d):
        rot_vec_1 = F.normalize(pose_6d[:,:,:,0:3], dim=-1) # forward
        rot_vec_2 = F.normalize(pose_6d[:,:,:,3:6], dim=-1) # up
        rot_vec_3 = torch.cross(rot_vec_2, rot_vec_1) # right        
        return torch.stack([rot_vec_3,rot_vec_2,rot_vec_1], dim=-1) # (right, up, forward)
        
    def linear_velocity(self, positions):
        return positions[:,1:,:]-positions[:,:-1,:]
        
    def angular_velocity(self, rotations):
        N = rotations.shape[0]
        L = rotations.shape[1]
        J = int(rotations.shape[2]/6)
        rotations = torch.reshape(rotations, (N,L,J,6))                
        return self.matrot2sixd(torch.matmul(torch.inverse(self.sixd2matrot(rotations[:,:-1,:,:])), self.sixd2matrot(rotations[:,1:,:,:])))  
                   
    def forward(self, local, remote):
        Y = remote[:, W:L, -OUTPUT_SIZE:]
        Y_hat_list = []
        for i in range(T):                
            Y_hat_list.append(self.model(local[:, i+1:W+i+1, :INPUT_SIZE], remote[:, i+0:W+i+0, :INPUT_SIZE]))
        Y_hat = torch.stack(Y_hat_list, dim=1)
        return Y, Y_hat
        
    def cal_loss_dis(self, remote, Y_hat):                       
        fake = torch.cat((remote[:,:W,-OUTPUT_SIZE:], Y_hat.detach()), 1)
        fake_pos = torch.cat((fake[:,:,6:9], fake[:,:,15:18]), -1)
        fake_rot = torch.cat((fake[:,:,0:6], fake[:,:,9:15]), -1)
        fake_lin = self.linear_velocity(fake_pos)
        fake_ang = self.angular_velocity(fake_rot)
        fake_ang = torch.reshape(fake_ang, (fake_ang.size(0), fake_ang.size(1), -1))
        fake = torch.cat((fake_pos[:,1:,:], fake_rot[:,1:,:], fake_lin, fake_ang), -1)
        
        real = remote[:,:,-OUTPUT_SIZE:]
        real_pos = torch.cat((real[:,:,6:9], real[:,:,15:18]), -1)
        real_rot = torch.cat((real[:,:,0:6], real[:,:,9:15]), -1)
        real_lin = self.linear_velocity(real_pos)
        real_ang = self.angular_velocity(real_rot)
        real_ang = torch.reshape(real_ang, (real_ang.size(0), real_ang.size(1), -1))
        real = torch.cat((real_pos[:,1:,:], real_rot[:,1:,:], real_lin, real_ang), -1)      
        
        loss_dis = lambda_gan * (torch.mean((self.discriminator_shorterm(real[:,-(T+W_s-1):,:]) - 1) ** 2) + \
                                 torch.mean((self.discriminator_shorterm(fake[:,-(T+W_s-1):,:]) - 0) ** 2) + \
                                 torch.mean((self.discriminator_longterm(real[:,-(T+W_l-1):,:]) - 1) ** 2) + \
                                 torch.mean((self.discriminator_longterm(fake[:,-(T+W_l-1):,:]) - 0) ** 2)) if lambda_gan !=0 else 0
        return loss_dis
    
    def cal_loss_gen(self, remote, Y_hat, Y):        
        fake = torch.cat((remote[:,:W,-OUTPUT_SIZE:], Y_hat), 1)
        fake_pos = torch.cat((fake[:,:,6:9], fake[:,:,15:18]), -1)
        fake_rot = torch.cat((fake[:,:,0:6], fake[:,:,9:15]), -1)
        fake_lin = self.linear_velocity(fake_pos)
        fake_ang = self.angular_velocity(fake_rot)
        fake_ang = torch.reshape(fake_ang, (fake_ang.size(0), fake_ang.size(1), -1))
        fake = torch.cat((fake_pos[:,1:,:], fake_rot[:,1:,:], fake_lin, fake_ang), -1)        
                
        loss_rec = F.l1_loss(Y_hat, Y, reduction='sum') / BATCH_SIZE / T
        loss_gan = lambda_gan * (torch.mean((self.discriminator_shorterm(fake[:,-(T+W_s-1):,:]) - 1) ** 2) +\
                                 torch.mean((self.discriminator_longterm(fake[:,-(T+W_l-1):,:]) - 1) ** 2)) if lambda_gan !=0 else 0                            
        loss = loss_rec + loss_gan
        return loss_rec, loss_gan, loss
    
    def run(self, dataloader, epoch, mode='TRAIN'):
        if mode == 'TRAIN':
            self.model.train()
            self.discriminator_shorterm.train()
            self.discriminator_longterm.train()
            
        epoch_loss_dis = 0
        epoch_loss_rec = 0
        epoch_loss_gan = 0        
        epoch_loss = 0
        
        pbar = tqdm(dataloader)
        for iter, data in enumerate(pbar):
            
            local, remote  = data['local'].to(self.device), data['remote'].to(self.device)                       
            
            Y, Y_hat = self.forward(local, remote)       
                                 
            ############################################################################################################################################
            # ---------------------
            #  Train Discriminator
            # ---------------------                    
            loss_dis = self.cal_loss_dis(remote, Y_hat)            
            if mode=='TRAIN' and lambda_gan!=0:
                loss_dis.backward()
                self.optimizer_D.step()
                self.optimizer_D.zero_grad()

            # -----------------
            #  Train Generator
            # -----------------
            loss_rec, loss_gan, loss = self.cal_loss_gen(remote, Y_hat, Y)                             
            if mode=='TRAIN':
                loss.backward()
                self.optimizer_G.step()
                self.optimizer_G.zero_grad()
            ############################################################################################################################################
            
            pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(dataloader)))
            pbar.set_postfix({"loss_rec":((loss_rec.item()))})                 

            if iter % 50 == 0:                                                
                if mode == 'TRAIN':                    
                    writer.add_scalar("Train (Reconstruction)", loss_rec.item(), self.iter)
                    writer.add_scalar("Train (Generator)", loss_gan.item() if lambda_gan !=0 else 0, self.iter)                                                
                    writer.add_scalar("Train (Discriminator)", loss_dis.item() if lambda_gan !=0 else 0, self.iter)
                    self.iter += 1
                    
            epoch_loss_dis = epoch_loss_dis + loss_dis.item() if lambda_gan !=0 else 0                        
            epoch_loss_rec += loss_rec.item()
            epoch_loss_gan = epoch_loss_gan + loss_gan.item() if lambda_gan !=0 else 0            
            epoch_loss += loss.item()            
        
        epoch_loss_dis = epoch_loss_dis / (len(dataloader)) if lambda_gan !=0 else 0
        epoch_loss_rec /= (len(dataloader))
        epoch_loss_gan = epoch_loss_gan / (len(dataloader)) if lambda_gan !=0 else 0                         
        epoch_loss /= (len(dataloader))
            
        return epoch_loss, epoch_loss_rec, epoch_loss_gan, epoch_loss_dis
    
def Curriculum_Learning(pairer, level, gpu_id, epochs):
    pairer.generate_paired_motion(0, level)
    train_data_set = Dataset(gpu_id=gpu_id)
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=16, pin_memory=True, drop_last=True)
    NUM_BATCHES = len(train_data_set) // BATCH_SIZE
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), NUM_BATCHES))
        
    train_start = time.time()
    for epoch in range(0, epochs):
        # Train
        epoch_start = time.time()        
        total_train_loss, total_train_loss_rec, total_train_loss_gan, total_train_loss_dis = runner.run(train_data_loader, epoch, 'TRAIN')
        training_time = time.time() - epoch_start
        log = "Train Time = %.1f, [Epoch %d/%d] [Train Loss: %.7f] [Reconstruction Loss: %.7f] [GAN Loss: %.7f] [Dis Loss: %.7f]" % (training_time, epoch+1, epochs, total_train_loss, total_train_loss_rec, total_train_loss_gan, total_train_loss_dis)
        print(log)
                                             
        # Model Save
        torch.save(model.state_dict(), os.path.join(save_dir, "ISMAR2023"+MODEL+"("+level+")"+str(epoch)+".pt"))      

        if level=='Hard' and epoch != epochs-1: # resample            
            runner.scheduler_D.step()
            runner.scheduler_G.step()        
            pairer.generate_paired_motion((epoch+1)%210, level)
            train_data_set = Dataset(gpu_id=GPU_ID)
            train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=16, pin_memory=True, drop_last=True)
            NUM_BATCHES = len(train_data_set) // BATCH_SIZE                     
    print('Curriculum Learning ('+level+') Finish!')
    print("Total Time= %.2f" %(time.time()-train_start))
                       
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', help='Model')
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--handedness', default='right', help='Handedness')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size for training')
parser.add_argument('--L', type=int, default=30, help='Clip Length')
parser.add_argument('--W', type=int, default=20, help='Window Size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lambda_gan', type=float, default=0.5, help='lambda gan')
parser.add_argument('--lambda_gan_s_Window', type=int, default=2, help='lambda gan_s window size')
parser.add_argument('--lambda_gan_l_Window', type=int, default=10, help='lambda gan_l window size')
args = parser.parse_args()

if __name__ == '__main__':
    script_start = time.time()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')  
    
    # Training setup.
    MODEL = args.model
    GPU_ID = args.gpu_id
    SHUFFLE = args.shuffle
    HANDEDNESS = args.handedness
    BATCH_SIZE = args.batchSize
    L = args.L
    W = args.W
    T = L-W
    LR = args.lr  # learning rate     
    lambda_gan = args.lambda_gan      
    W_s = args.lambda_gan_s_Window
    W_l = args.lambda_gan_l_Window     

    Easy_epochs=25
    Hard_epochs=210*3
        
    INPUT_SIZE=16
    OUTPUT_SIZE=21
    NUM_EXPERTS=6

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = ISMAR2023(np.random.RandomState(23456), NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE, use_cuda=True, runtime=False)
    discriminator_shorterm = Discriminator(length=W_s, in_dim=36)
    discriminator_longterm = Discriminator(length=W_l, in_dim=36)
    
    runner = Runner(device=device,
                    model=model, 
                    discriminator_shorterm=discriminator_shorterm, 
                    discriminator_longterm=discriminator_longterm, 
                    lr=LR)

    save_dir = "ISMAR2023"+MODEL+","+str(HANDEDNESS)+","+str(BATCH_SIZE)+","+str(LR)+","+str(lambda_gan)+","+nowDatetime
    writer = SummaryWriter('logs/'+save_dir+'/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    REPERTORIES = [{'repertorie': 'Pointing at two targets', 'num': 210},
                   {'repertorie': 'Touching two targets', 'num': 210},
                   {'repertorie': 'Pointing at single target', 'num': 27},
                   {'repertorie': 'Touching single target', 'num': 30}]
    DATA_PAIRER = data_pairer(HANDEDNESS, REPERTORIES, L, W, GPU_ID) 
    DATA_PAIRER.delete()
    
    # Curriculum Learning (Easy)
    Curriculum_Learning(DATA_PAIRER, 'Easy', GPU_ID, Easy_epochs)

    # Curriculum Learning (Hard)
    Curriculum_Learning(DATA_PAIRER, 'Hard', GPU_ID, Hard_epochs)     
    
    writer.flush()
    writer.close()
    
    text_to_save = "Total Time= %.2f" %(time.time()-script_start)
    with open(save_dir+"/training time.txt", "w") as file:
        file.write(text_to_save)