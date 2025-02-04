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

from model import UGNet_MoE, UGNet_MLP, UGNet_Transformer, UGNet_Diffusion
from data_parser import data_parser
from data_loader import Dataset

class Runner(object):                            
    def __init__(self, device, model, lr):               
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)           
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        self.learning_rate = lr
        self.iter = 0       
        
    def run(self, dataloader, epoch, Bernoulli, mode='TRAIN'):
        if mode == 'TRAIN':
            self.model.train()

        epoch_loss = 0

        y_hat = 0        
        Y_hat_list = []
                          
        pbar = tqdm(dataloader)
        for iter, data in enumerate(pbar):
            X = data['data'].to(self.device)                              
            
            for t in range(T):
                x = X[:, t:WINDOW_SIZE+t, :INPUT_SIZE]
                y = X[:, (WINDOW_SIZE-1)+t, -OUTPUT_SIZE:]
                
                if t != 0 and Bernoulli.sample().int() == 1:
                    x[:, -1, :OUTPUT_SIZE] = y_hat
                
                if MODEL != 'Diffusion': # MoE, MLP, Transformer
                    y_hat = self.model(x)                    
                else: # Diffusion                                 
                    y_hat = self.model(y, x)                                                               
                loss = F.l1_loss(y, y_hat, reduction='mean')                    
                
                if mode=='TRAIN':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                y_hat = y_hat.detach()
                Y_hat_list.append(y_hat)
                
                if iter % 50 == 0:
                    if mode == 'TRAIN':
                        writer.add_scalar("Train", loss.item(), self.iter)
                        self.iter += 1
                
                pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(dataloader)))
                pbar.set_postfix({"loss":((loss.item()))})                       
                
                epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        
        self.scheduler.step()
        
        return epoch_loss
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Diffusion', help='Model')
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--directory_id', type=int, default=0, help='Directory ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--handedness', default='right', help='Handedness')
parser.add_argument('--scheduledSampling', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--dataAugmentation', default=False, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--epochs', type=int, default=120, help='Epochs')
parser.add_argument('--c1', type=int, default=30, help='C1')
parser.add_argument('--c2', type=int, default=60, help='C2')
parser.add_argument('--batchSize', type=int, default=32, help='Batch Size')
parser.add_argument('--framerate', type=int, default=30, help='Framerate')
parser.add_argument('--windowLength', type=int, default=30, help='Window Size')
parser.add_argument('--clipLength', type=int, default=45, help='Clip Length')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--expertNumber', type=int, default=9, help='Expert Number')
parser.add_argument('--diffusionStepSize', type=int, default=40, help='Expert Number')
args = parser.parse_args()

# python train_UGNet.py --gpu_id 1 --handedness 'left' --model 'Diffusion'; python train_UGNet.py --gpu_id 1 --handedness 'left' --model 'Transformer'; python train_UGNet.py --gpu_id 1 --handedness 'left' --model 'MLP';

if __name__ == '__main__':
    script_start = time.time()

    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%m%d_%H%M%S')
    
    MODEL = args.model
    GPU_ID = args.gpu_id
    DIRECTORY_ID = args.directory_id
    SHUFFLE = args.shuffle    
    HANDEDNESS = args.handedness
    SCHEDULED_SAMPLING = args.scheduledSampling
    DATA_AUGMENTATION = args.dataAugmentation
    EPOCHS = args.epochs
    C1 = args.c1
    C2 = args.c2
    BATCH_SIZE = args.batchSize
    FRAMERATE = args.framerate
    WINDOW_SIZE = args.windowLength
    CLIP_LENGTH = args.clipLength
    T = CLIP_LENGTH-WINDOW_SIZE+1
    LR = args.lr
    NUM_EXPERTS = args.expertNumber 
    Diffusion_Step_Size = args.diffusionStepSize
            
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  
        
    INPUT_SIZE=24+(5+3)
    OUTPUT_SIZE=24
    
    if MODEL == 'MoE':    
        model = UGNet_MoE(rng=np.random.RandomState(23456), num_experts=NUM_EXPERTS, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, window_size=WINDOW_SIZE, use_cuda=True)
    elif MODEL == 'MLP':    
        model = UGNet_MLP(rng=np.random.RandomState(23456), num_experts=NUM_EXPERTS, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, window_size=WINDOW_SIZE, use_cuda=True)    
    elif MODEL == 'Transformer':    
        model = UGNet_Transformer(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, use_cuda=True)
    elif MODEL == 'Diffusion':
        model = UGNet_Diffusion(T = Diffusion_Step_Size)
        
    runner = Runner(device=device, model=model, lr=LR)

    save_dir = "UGNet("+MODEL+")"+","+str(HANDEDNESS)+","+str(SCHEDULED_SAMPLING)+","+str(DATA_AUGMENTATION)+","+str(EPOCHS)+","+str(C1)+","+ str(C2)+","+\
                                      str(BATCH_SIZE)+","+str(FRAMERATE)+","+str(WINDOW_SIZE)+","+str(CLIP_LENGTH)+","+str(LR)+","+str(NUM_EXPERTS)+","+str(Diffusion_Step_Size)+","+nowDatetime
                                     
    writer = SummaryWriter('logs/'+save_dir+'/')
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
        
    DATA_PARSER = data_parser(HANDEDNESS, REPERTORIES, WINDOW_SIZE, CLIP_LENGTH, "UGNet", DIRECTORY_ID)    
    DATA_PARSER.delete()
    DATA_PARSER.split()
    DATA_PARSER.generate(DATA_AUGMENTATION)
    DATA_PARSER.__len__()
    
    train_data_set = Dataset(directory_id=DIRECTORY_ID, network="UGNet")
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=4, pin_memory=True, drop_last=True)
    NUM_BATCHES = len(train_data_set) // BATCH_SIZE
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), NUM_BATCHES))
    
    P = 1
    train_start = time.time()
    for epoch in range(1, EPOCHS+1):
        if SCHEDULED_SAMPLING:
            if epoch <= C1:
                P = 1
            elif C1 < epoch <= C2:
                P = 1 - (epoch - C1) / float(C2 - C1)
            else:
                P = 0
        Bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor(1-P, dtype=torch.float))  
        
        epoch_start = time.time()
        total_train_loss = runner.run(train_data_loader, epoch, Bernoulli, 'TRAIN')        
        torch.save(model.state_dict(), os.path.join(save_dir, "UGNet("+MODEL+")"+str(epoch)+".pt"))      
        training_time = time.time() - epoch_start
        log = "Train Time = %.1f, [Epoch %d/%d] [Loss: %.7f]" % (training_time, epoch, EPOCHS, total_train_loss)
        print(log)    
    print("Total Time= %.2f" %(time.time()-train_start))                           
    print("\n")    
    writer.flush()
    writer.close()
    
    text_to_save = "Total Time= %.2f" %(time.time()-script_start)
    with open(save_dir+"/training time.txt", "w") as file:
        file.write(text_to_save)    