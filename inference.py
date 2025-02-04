import zmq
import time
import torch
import numpy as np

from model import MPNet_MoE, MPNet_MLP, MPNet_Transformer
from model import UGNet_MoE, UGNet_MLP, UGNet_Transformer, UGNet_Diffusion

WINDOW_SIZE = 30
device = torch.device('cpu')
# device = torch.device('cuda:0')

HANDEDNESS = 'right'

# MPNet
MPNet_Model = 'MoE'
if MPNet_Model =='MoE':
    path = "model(trained)/"+HANDEDNESS+"/MPNet(MoE)(AC)80.pt"
    MPNet = MPNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=6+5, hidden_size=24, output_size=3, use_cuda=True)
elif MPNet_Model =='MLP':
    path = "model(trained)/"+HANDEDNESS+"/MPNet(MLP)(AC)80.pt"
    MPNet = MPNet_MLP(input_size=6+5, hidden_size=24, output_size=3, window_size=WINDOW_SIZE, use_cuda=True)    
elif MPNet_Model =='Transformer':
    path = "model(trained)/"+HANDEDNESS+"/MPNet(Transformer)(AC)80.pt"
    MPNet = MPNet_Transformer(input_size=6+5, hidden_size=16, output_size=3, use_cuda=True)    
MPNet.load_state_dict(torch.load(path, map_location='cpu'))
MPNet = MPNet.to(device)
MPNet.eval()

# UGNet
UGNet_Model = 'MoE'
if UGNet_Model == 'MoE':
    path = "model(trained)/"+HANDEDNESS+"/UGNet(MoE)(AC)120.pt"
    UGNet = UGNet_MoE(rng=np.random.RandomState(23456), num_experts=9, input_size=24+(5+3), output_size=24, window_size=WINDOW_SIZE, use_cuda=True)
elif UGNet_Model == 'MLP':
    path = "model(trained)/"+HANDEDNESS+"/UGNet(MLP)(AC)120.pt"
    UGNet = UGNet_MLP(rng=np.random.RandomState(23456), num_experts=9, input_size=24+(5+3), output_size=24, window_size=WINDOW_SIZE, use_cuda=True)    
elif UGNet_Model == 'Transformer':
    path = "model(trained)/"+HANDEDNESS+"/UGNet(Transformer)(AC)120.pt"
    UGNet = UGNet_Transformer(input_size=24+(5+3), output_size=24, use_cuda=True)
elif UGNet_Model == 'Diffusion':
    path = "model(trained)/"+HANDEDNESS+"/UGNet(Diffusion)(AC)120.pt"
    UGNet = UGNet_Diffusion(T=30)
UGNet.load_state_dict(torch.load(path, map_location='cpu'))
UGNet = UGNet.to(device)
UGNet.eval()

server_address = 'tcp://*:12345'
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(server_address)
print("server initiated")

with torch.no_grad():
    Progression = torch.zeros(1,WINDOW_SIZE,3).to(device)
    
    while True:
        # start_time = time.time()
        msg = socket.recv()
        message = np.fromstring(msg, dtype=np.float32, sep=',')        
                
        input = np.reshape(message, (WINDOW_SIZE, -1))
        input_data = torch.from_numpy(input)
        input_data = input_data.float()
        input_data = torch.unsqueeze(input_data, 0)
        input_data = input_data.to(device)
        
        # start_time = time.time()
        progression = MPNet(input_data[:,:,:11])       
        # print("MPNet: " + str(time.time()-start_time))
        Progression_temp = Progression.clone()
        Progression[:,:-1,:] = Progression_temp[:,1:,:]
        Progression[0,-1,:] = progression.squeeze(1)
                
        input_data = torch.cat((input_data[:,:,11:40], Progression),-1)
        # start_time = time.time()
        if UGNet_Model != 'Diffusion':
            output_data = UGNet(input_data)
        else:
            output_data = UGNet.sample_ddpm(input_data)
        # print("UGNet: " + str(time.time()-start_time))
        output_data = torch.cat((output_data, Progression[:,-1,:]), -1)
        output_data = output_data.detach().cpu().numpy()
        output_data = np.squeeze(output_data)
        
        output = ','.join(str(x) for x in output_data)
        # time.sleep(0.016+0.013) # MLP
        # time.sleep(0.015+0.013) # MoE, Transformer
        time.sleep(0.010) # MoE
        socket.send_string(output)
        # print("Total: " + str(time.time()-start_time))        