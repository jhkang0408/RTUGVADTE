import os
import gc
import time
import datetime 
import numpy as np
import os.path as osp

import torch

from model import MPNet_MoE, MPNet_MLP, MPNet_Transformer
from data_parser import data_parser

class Runner(object):                                              
    def __init__(self, device, model):               
        self.device = device
        self.MPNet = model.to(self.device)
        self.iter = 0
        self.Total_Accuracy = 0
        self.Gaze_Accuracy = 0
        self.DeicticGesture_Accuracy = 0
        self.Idle_Accuracy = 0
        
    def run(self, sequence):
        sequence = sequence.to(self.device)        
        sequence = sequence.unsqueeze(0)
        T = sequence.shape[1]-WINDOW_SIZE
                
        for t in range(T):
            Input_MPNet = sequence[:, t:WINDOW_SIZE+t, :11]
            prediction = self.MPNet(Input_MPNet)
            groundtruth = sequence[:, WINDOW_SIZE+t-1, 11:14]
            
            Gaze_prediction = prediction[0,0]
            DeicticGesture_prediction = prediction[0,1]
            Idle_prediction = prediction[0,2]

            Gaze_groundtruth = groundtruth[0,0]
            DeicticGesture_groundtruth = groundtruth[0,1]
            Idle_groundtruth = groundtruth[0,2]                            
                        
            self.Gaze_Accuracy += torch.abs(Gaze_prediction-Gaze_groundtruth)
            self.DeicticGesture_Accuracy += torch.abs(DeicticGesture_prediction-DeicticGesture_groundtruth)
            self.Idle_Accuracy += torch.abs(Idle_prediction-Idle_groundtruth)
            self.Total_Accuracy += (torch.abs(Gaze_prediction-Gaze_groundtruth) + torch.abs(DeicticGesture_prediction-DeicticGesture_groundtruth) + torch.abs(Idle_prediction-Idle_groundtruth))            
            self.iter += 1     
        
    def result(self):
        log = "[Total: %.3f] [Gaze: %.3f] [Deictic Gesture: %.3f] [Idle: %.3f]" % (self.Total_Accuracy/self.iter/3, self.Gaze_Accuracy/self.iter, self.DeicticGesture_Accuracy/self.iter, self.Idle_Accuracy/self.iter)
        with open('Quantitative Evaluation(MPNet Accuracy)'+'(MPNet_'+MPNet_MODEL+')'+"/MPNet Accuracy.txt", "w") as file:            
            file.write(log)
        print(log)

    def run_Stability(self, initial_feature, changed_feature):    
        initial_feature = initial_feature.to(self.device)
        changed_feature = changed_feature.to(self.device) 
        
        T = 30
                                
        feature = initial_feature.repeat(WINDOW_SIZE, 1).unsqueeze(0)
        feature[0,-1,:] = changed_feature 
        
        for t in range(T):
            Input_MPNet = feature
            prediction = self.MPNet(Input_MPNet)            
            
            feature_temp = feature.clone()
            feature[:,:-1,:] = feature_temp[:,1:,:]
            feature[0,-1,:] = changed_feature 
            
            Gaze_prediction = prediction[0,0]
            DeicticGesture_prediction = prediction[0,1]
            Idle_prediction = prediction[0,2]

            Gaze_groundtruth = 0
            DeicticGesture_groundtruth = 0
            Idle_groundtruth = 1
                        
            self.Gaze_Accuracy += torch.abs(Gaze_prediction-Gaze_groundtruth)
            self.DeicticGesture_Accuracy += torch.abs(DeicticGesture_prediction-DeicticGesture_groundtruth)
            self.Idle_Accuracy += torch.abs(Idle_prediction-Idle_groundtruth)
            self.Total_Accuracy += (torch.abs(Gaze_prediction-Gaze_groundtruth) + torch.abs(DeicticGesture_prediction-DeicticGesture_groundtruth) + torch.abs(Idle_prediction-Idle_groundtruth))            
            self.iter += 1    

    def result_Stability(self):
        log = "[Total: %.3f] [Gaze: %.3f] [Deictic Gesture: %.3f] [Idle: %.3f]" % (self.Total_Accuracy/self.iter/3, self.Gaze_Accuracy/self.iter, self.DeicticGesture_Accuracy/self.iter, self.Idle_Accuracy/self.iter)
        with open('Quantitative Evaluation(Stability)'+'(MPNet_'+MPNet_MODEL+')'+"/Stability.txt", "w") as file:            
            file.write(log)
        print(log) 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default='161', help='Directory')
parser.add_argument('--model', default='MoE', help='Model')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--metric', default='MPE', help='Metric')

args = parser.parse_args()

# python test_MPNet.py; python test_MPNet.py --model 'MLP'; python test_MPNet.py --model 'Transformer'
# python test_MPNet.py --subject '172'; python test_MPNet.py --subject '172' --model 'MLP'; python test_MPNet.py --subject '172' --model 'Transformer'
# python test_MPNet.py --subject '173'; python test_MPNet.py --subject '173' --model 'MLP'; python test_MPNet.py --subject '173' --model 'Transformer'
# python test_MPNet.py --subject '174'; python test_MPNet.py --subject '174' --model 'MLP'; python test_MPNet.py --subject '174' --model 'Transformer'
# python test_MPNet.py --subject '180'; python test_MPNet.py --subject '180' --model 'MLP'; python test_MPNet.py --subject '180' --model 'Transformer'

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()
    
    SUBJECT = args.subject
    MPNet_MODEL = args.model
    GPU_ID = args.gpu_id
    METRIC = args.metric
    
    DIR = 'data(raw)/test_MPNet/' + SUBJECT + '/'
                    
    if SUBJECT == '161' or SUBJECT == '172' or SUBJECT == '180':
        if MPNet_MODEL == 'MoE':    
            MPNet_path = "model(trained)/right/MPNet(MoE)(AC)80.pt"
        elif MPNet_MODEL == 'MLP':    
            MPNet_path = "model(trained)/right/MPNet(MLP)(AC)80.pt"
        elif MPNet_MODEL == 'Transformer':    
            MPNet_path = "model(trained)/right/MPNet(Transformer)(AC)80.pt"        
            
    elif SUBJECT == '173' or SUBJECT == '174':
        if MPNet_MODEL == 'MoE':    
            MPNet_path = "model(trained)/left/MPNet(MoE)(AC)80.pt"
        elif MPNet_MODEL == 'MLP':    
            MPNet_path = "model(trained)/left/MPNet(MLP)(AC)80.pt"
        elif MPNet_MODEL == 'Transformer':    
            MPNet_path = "model(trained)/left/MPNet(Transformer)(AC)80.pt" 

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
    
    if METRIC == 'MPE': # Motion Progression Error
        if not os.path.exists('Quantitative Evaluation(MPNet Accuracy)'+'(MPNet_'+MPNet_MODEL+')'):
            os.makedirs('Quantitative Evaluation(MPNet Accuracy)'+'(MPNet_'+MPNet_MODEL+')')  
        
        REPERTORIES = [{'repertorie': 'Pointing at single target', 'num': 27},
                        {'repertorie': 'Pointing at two targets', 'num': 8},
                        {'repertorie': 'Pointing at single target with gaze shift', 'num': 8},
                        {'repertorie': 'Pointing at single target with explanation', 'num': 8},
                        {'repertorie': 'Transition (Pointing)', 'num': 27},
                        {'repertorie': 'Touching single target', 'num': 30},
                        {'repertorie': 'Touching single target with gaze shift', 'num': 8},
                        {'repertorie': 'Touching single target with explanation', 'num': 8},
                        {'repertorie': 'Touching two targets', 'num': 8},
                        {'repertorie': 'Transition (Touching)', 'num': 27}]         
        
        DATA_PARSER = data_parser('', REPERTORIES, WINDOW_SIZE, 45, '', 0)
        DATA_PARSER.raw_data_dir = DIR
        
        runner = Runner(device=device, model=MPNet)    
        with torch.no_grad():
            for repertorie in DATA_PARSER.repertories:
                folder_split = DATA_PARSER.raw_data_dir + repertorie['repertorie'] + '/' + 'split'                  
                for i in range(repertorie['num']):
                    sequence = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                    sequence = torch.tensor(sequence, dtype=torch.float32)
                    first_frame = sequence[0].unsqueeze(0)
                    repeated_frames = first_frame.repeat(29, 1)
                    sequence = torch.cat((repeated_frames, sequence), dim=0)                
                    runner.run(sequence)
            runner.result() 
                
    elif METRIC == 'Stability': # State Stability
        if not os.path.exists('Quantitative Evaluation(Stability)'+'(MPNet_'+MPNet_MODEL+')'):
            os.makedirs('Quantitative Evaluation(Stability)'+'(MPNet_'+MPNet_MODEL+')')                              
            
        REPERTORIES = [{'repertorie': 'Pointing at two targets', 'num': 210},
                       {'repertorie': 'Touching two targets', 'num': 210}]
                               
        runner = Runner(device=device, model=MPNet)    
        with torch.no_grad():
            for repertorie in REPERTORIES:
                folder_split ='data(raw)/train/right/' + repertorie['repertorie'] + '/' + 'split'                  
                for i in range(repertorie['num']):
                    sequence = np.loadtxt(osp.join(folder_split, 'Data'+str(i+1)+'.txt')).astype(np.float32)
                    sequence = torch.tensor(sequence, dtype=torch.float32)
                    initial_feature = sequence[0,:11]      
                    changed_feature = sequence[-1,:11]
                    runner.run_Stability(initial_feature, changed_feature)
            runner.result_Stability() 
            