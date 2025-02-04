import os
import zmq
import time
import torch
import numpy as np

from model import ISMAR2023

W = 20 # window_size
INPUT_SIZE = 12+4
OUTPUT_SIZE = 21
NUM_EXPERTS = 6
path = "model(trained)/right/ISMAR2023(Hard)629.pt"
# path = "model(trained)/left/ISMAR2023(Hard)629.pt"

device = torch.device('cpu') 
model = ISMAR2023(np.random.RandomState(23456), NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE, use_cuda=True, runtime=True)   
model.load_state_dict(torch.load(path, map_location='cpu'))
model = model.to(device)
model.eval()

server_address = 'tcp://*:12345'
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(server_address)
print("server initiated")

with torch.no_grad():
    
    while True:
        msg = socket.recv()
        message = np.fromstring(msg, dtype=np.float32, sep=',')
        
        input = np.reshape(message, (W, -1))
        input_data = torch.from_numpy(input)
        input_data = input_data.float()
        input_data = torch.unsqueeze(input_data, 0)
        input_data = input_data.to(device)
        
        Local  = input_data[:, :, 0:INPUT_SIZE]
        Remote = input_data[:, :, INPUT_SIZE:INPUT_SIZE*2]
        
        # s = time.time()        
        output_data = model(Local[:, :W, :INPUT_SIZE], Remote[:, :W, :INPUT_SIZE])
        # print(time.time()-s)
        output_data = output_data.detach().cpu().numpy()
        output_data = np.squeeze(output_data)
        output = ','.join(str(x) for x in output_data)
        socket.send_string(output)