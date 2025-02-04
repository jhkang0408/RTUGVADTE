import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import numpy as np
        
class ISMAR2023(nn.Module):
    def __init__(self, rng, num_experts, input_size, output_size, use_cuda=False, runtime=False, **kwargs):
        super(ISMAR2023, self).__init__()
        self.input_size = input_size-4
        self.target_size = 3
        self.output_size = output_size
        self.num_experts = num_experts
        self.runtime = runtime
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        self.gating_network = GatingNetwork(input_size=self.target_size, output_size=self.num_experts)                        
        self.angle_encoder = ExpertMLP(rng=rng, num_experts=self.num_experts, input_size=self.input_size, hidden_size=16, output_size=16, use_cuda=use_cuda)                           
        self.motion_decoder = ExpertMLP(rng=rng, num_experts=self.num_experts, input_size=16+1, hidden_size=16, output_size=self.output_size, use_cuda=use_cuda)                         
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=2, batch_first=True) 
        
    def forward(self, Local, Remote):
        B, W, _ = Local.shape
                        
        Local_Angle = Local[:, :, 0:self.input_size]        
        Remote_Angle = Remote[:, :, 0:self.input_size]
        Local_Target = Local[:, :, self.input_size:self.input_size+self.target_size]    
        Remote_Target = Remote[:, :, self.input_size:self.input_size+self.target_size]
        Scale = Remote[:,:, self.input_size+self.target_size]

        # Gating
        Local_Omega = self.gating_network(Local_Target)
        Remote_Omega = self.gating_network(Remote_Target)
                        
        # Encoding
        Z_Local = self.angle_encoder(Local_Angle.reshape(-1, self.input_size), Local_Omega.reshape(-1, self.num_experts))        
        Z_Remote = self.angle_encoder(Remote_Angle.reshape(-1, self.input_size), Remote_Omega.reshape(-1, self.num_experts))
        Z_Local = Z_Local.view(B, W, -1)
        Z_Remote = Z_Remote.view(B, W, -1)
                        
        # GRU
        Z = self.gru(Z_Local+Z_Remote)
        Z = Z[0][:,-1]
        
        # Decoding
        Y = self.motion_decoder(torch.cat((Z, torch.unsqueeze(Scale[:,-1],-1)), -1), Remote_Omega[:,-1,:])
        return Y

class GatingNetwork(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, **kwargs):
        super(GatingNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size    
        self.gru = nn.GRU(input_size=input_size, hidden_size=output_size, num_layers=2, batch_first=True)
   
    def forward(self, x):
        x = self.gru(x)
        x = x[0]
        x = nn.Softmax(dim=-1)(x)
        return x
        
class ExpertMLP(nn.Module):
    def __init__(self, rng, num_experts, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(ExpertMLP, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_cuda = use_cuda
        self.w_l1, self.b_l1 = self.init_params(self.num_experts, self.input_size, self.hidden_size)
        self.w_l2, self.b_l2 = self.init_params(self.num_experts, self.hidden_size, self.output_size)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        
    def init_params(self, num_experts, input_size, output_size):
        w_bound = np.sqrt(6. / np.prod([input_size, output_size]))
        w = np.asarray(
            self.rng.uniform(low=-w_bound, high=w_bound,
                              size=[num_experts, input_size, output_size]), dtype=np.float32)
        if self.use_cuda:
            w = nn.Parameter(
                torch.cuda.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(
                torch.cuda.FloatTensor(num_experts, output_size).fill_(0),
                requires_grad=True)
        else:
            w = nn.Parameter(
                torch.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(
                torch.FloatTensor(num_experts, output_size).fill_(0),
                requires_grad=True)
        return w, b

    def linearlayer(self, inputs, weights, bias):
        return torch.sum(inputs[..., None] * weights, dim=1) + bias

    def forward(self, x, blending_coef):
        # inputs: B*input_dim
        # Blending_coef : B*experts
        w_l1 = torch.sum(
            blending_coef[..., None, None] * self.w_l1[None], dim=1)
        b_l1 = torch.matmul(blending_coef, self.b_l1)
        w_l2 = torch.sum(
            blending_coef[..., None, None] * self.w_l2[None], dim=1)
        b_l2 = torch.matmul(blending_coef, self.b_l2)
        
        h1 = F.elu(self.bn(self.linearlayer(x, w_l1, b_l1)))
        h2 = self.linearlayer(h1, w_l2, b_l2)
        return h2

class Discriminator(nn.Module):
    def __init__(self, length, in_dim, hidden_dim=1024, out_dim=1):
        super(Discriminator, self).__init__()
        self.fc0 = nn.Conv1d(in_dim, hidden_dim, kernel_size=length, bias=True)        
        self.fc1 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=1, bias=True)
        self.fc2 = nn.Conv1d(hidden_dim//2, out_dim, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.fc0(x)                
        x = self.relu(x)        
        x = self.fc1(x)
        x = self.relu(x)                
        x = self.fc2(x)        
        x = torch.squeeze(x)           
        return x

class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, **kwargs):
        super(ConvolutionalAutoEncoder, self).__init__()        
        # Encoder
        self.conv_encode1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_size, bias=True)         
        self.conv_encode2 = nn.Conv1d(hidden_size, hidden_size//2, kernel_size=16, bias=True)         
        
        # Decoder
        self.conv_decode2 = nn.ConvTranspose1d(hidden_size//2, hidden_size, kernel_size=16, bias=True)
        self.conv_decode1 = nn.ConvTranspose1d(hidden_size, input_size, kernel_size=kernel_size, bias=True) 
        
    def forward(self, x, evaluation):
        # Encoder             
        x = x.permute(0,2,1)                     
        x = self.conv_encode1(x)        
        x = nn.ReLU()(x)
        x = self.conv_encode2(x)          
        if evaluation == False:                                    
            # Decoder                    
            x = self.conv_decode2(x)
            x = nn.ReLU()(x)
            x = self.conv_decode1(x)
            x = x.permute(0,2,1)
        return x 
    
def Get_Model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():        
        buffer_size += buffer.nelement() * buffer.element_size()    
    # print('model size: {:.5f}MB'.format(size_all_mb))    
    print(param_size)

if __name__ == '__main__':

    rng = np.random.RandomState(23456)
    NUM_EXPERTS=6
    INPUT_SIZE=12+4
    OUTPUT_SIZE=21
    B=32
    W=20
    local = torch.ones((B, W, INPUT_SIZE))
    remote = torch.ones((B, W, INPUT_SIZE))
        
    # wo_MoE
    model = ISMAR2023(rng, NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE, False, True).to('cpu')
    output = model(local, remote)
    print("Network_wo_MoE output.shape="+str(output.shape))
    Get_Model_size(model)    
    print("")
    
    # Discriminator        
    model = Discriminator(length=10, in_dim=OUTPUT_SIZE)
    input = torch.rand((B, W, OUTPUT_SIZE))
    output = model(input)
    print("Discriminator output.shape="+str(output.shape))        
    Get_Model_size(model)    
    print("")

    # ConvolutionalAutoEncoder
    INPUT_SIZE=18
    HIDDEN_SIZE=32
    KERNEL_SIZE=15
    W=30
    model = ConvolutionalAutoEncoder(INPUT_SIZE, HIDDEN_SIZE, KERNEL_SIZE)
    input = torch.rand((B, W, INPUT_SIZE))
    output = model(input, False)
    print("ConvolutionalAutoEncoder output.shape="+str(output.shape))
