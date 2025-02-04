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

from model import MPNet_MoE, MPNet_MLP, MPNet_Transformer
from data_parser import data_parser
from data_loader import Dataset

class Runner(object):                                              
    def __init__(self, device, model, lr):               
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.975)
        self.learning_rate = lr
        self.iter = 0
                
    def run(self, dataloader, epoch, mode='TRAIN'):
        if mode == 'TRAIN':
            self.model.train()
                
        epoch_loss = 0        
        
        pbar = tqdm(dataloader)
        for iter, data in enumerate(pbar):            
            X = data['data'].to(self.device)
            
            for t in range(T):
                x = X[:, t:WINDOW_SIZE+t, :INPUT_SIZE]
                y = X[:, (WINDOW_SIZE-1)+t, -OUTPUT_SIZE:]
                                
                y_hat = self.model(x)
                
                loss = F.l1_loss(y_hat, y, reduction='mean')                                
                
                if mode=='TRAIN':
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, iter+1, len(dataloader)))
                pbar.set_postfix({"loss":((loss.item()))})                                 
    
                y_hat = y_hat.detach() 
                
                if iter % 50 == 0:                                                
                    if mode == 'TRAIN':                    
                        writer.add_scalar("Train (Loss)", loss.item(), self.iter)

                        self.iter += 1
                        
                epoch_loss += loss.item()
                                
        epoch_loss /= (len(dataloader))

        runner.scheduler.step()       
        
        return epoch_loss

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MoE', help='Model')
parser.add_argument('--use_cuda', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--directory_id', type=int, default=0, help='Directory ID')
parser.add_argument('--shuffle', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--handedness', default='right', help='Handedness')
parser.add_argument('--dataAugmentation', default=True, type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--epochs', type=int, default=80, help='Epochs')
parser.add_argument('--batchSize', type=int, default=16, help='Batch Size')
parser.add_argument('--framerate', type=int, default=30, help='Framerate')
parser.add_argument('--windowLength', type=int, default=30, help='Window Size')
parser.add_argument('--clipLength', type=int, default=45, help='Clip Length')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--expertNumber', type=int, default=9, help='Expert Number')
parser.add_argument('--hiddenSize', type=int, default=24, help='Hidden Size')
args = parser.parse_args()

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
    DATA_AUGMENTATION = args.dataAugmentation
    EPOCHS = args.epochs
    BATCH_SIZE = args.batchSize
    FRAMERATE = args.framerate
    WINDOW_SIZE = args.windowLength
    CLIP_LENGTH = args.clipLength
    T = CLIP_LENGTH-WINDOW_SIZE+1    
    LR = args.lr
    NUM_EXPERTS = args.expertNumber
    HIDDEN_SIZE = args.hiddenSize
    
    INPUT_SIZE=6+5
    OUTPUT_SIZE=3
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    if MODEL == 'MoE':
        model = MPNet_MoE(rng=np.random.RandomState(23456), num_experts=NUM_EXPERTS, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, use_cuda=True)
    elif MODEL == 'MLP':    
        HIDDEN_SIZE = 24
        model = MPNet_MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, WINDOW_SIZE, use_cuda=True)        
    elif MODEL == 'Transformer':    
        HIDDEN_SIZE = 16
        model = MPNet_Transformer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, use_cuda=True)         
           
    runner = Runner(device=device, model=model, lr=LR)

    save_dir = "MPNet("+MODEL+")"+","+str(HANDEDNESS)+","+str(DATA_AUGMENTATION)+","+str(EPOCHS)+","+str(BATCH_SIZE)+","+str(FRAMERATE)+","+\
                                      str(WINDOW_SIZE)+","+str(CLIP_LENGTH)+","+str(LR)+","+str(NUM_EXPERTS)+","+str(HIDDEN_SIZE)+","+nowDatetime
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
        
    DATA_PARSER = data_parser(HANDEDNESS, REPERTORIES, WINDOW_SIZE, CLIP_LENGTH, "MPNet", DIRECTORY_ID)    
    DATA_PARSER.delete()
    DATA_PARSER.split()
    DATA_PARSER.generate(DATA_AUGMENTATION)
    DATA_PARSER.__len__()
    
    train_data_set = Dataset(directory_id=DIRECTORY_ID, network="MPNet")
    train_data_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=4, pin_memory=True, drop_last=True)
    NUM_BATCHES = len(train_data_set) // BATCH_SIZE
    print('# of train example: {}, # of batches {}'.format(len(train_data_set), NUM_BATCHES))
        
    train_start = time.time()
    for epoch in range(1, EPOCHS+1):
        epoch_start = time.time()
        total_train_loss = runner.run(train_data_loader, epoch, 'TRAIN')        
        torch.save(model.state_dict(), os.path.join(save_dir, "MPNet("+MODEL+")"+str(epoch)+".pt"))      
        training_time = time.time() - epoch_start
        log = "Train Time = %.1f, [Epoch %d/%d] [Loss: %.7f]" % (training_time, epoch, EPOCHS, total_train_loss)
        print(log)
    print("Total Time= %.2f" %(time.time()-train_start))
        
    writer.flush()
    writer.close() 
    
    text_to_save = "Total Time= %.2f" %(time.time()-script_start)
    with open(save_dir+"/training time.txt", "w") as file:
        file.write(text_to_save)    