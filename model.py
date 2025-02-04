import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F
import numpy as np
import math
from functools import partial

class MPNet_MoE(nn.Module):
    def __init__(self, rng, num_experts, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(MPNet_MoE, self).__init__()
        # variable
        self.num_experts = num_experts
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        self.input_size_gating = 5
        self.input_size_moe = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size  
        # model        
        self.gating_network = GatingNetwork_MPNet(input_size=self.input_size_gating, output_size=self.num_experts)
        self.gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)        
        self.moe_network = ExpertMLP_MPNet(rng=rng, num_experts=self.num_experts, input_size=self.hidden_size, hidden_size=self.hidden_size//2, output_size=self.output_size, use_cuda=use_cuda)                                                            

    def forward(self, X):
        # Input
        X_gating_network = X[:,:,-self.input_size_gating:]
        X_moe_network = X[:,:,:]

        # Gating Network
        Omega = self.gating_network(X_gating_network)
        
        # Recurrent Module
        Z = self.gru(X_moe_network)
        Z = Z[0][:,-1,:]
        
        # Prediction Network
        Y = self.moe_network(Z, Omega)
        Y = torch.clamp(Y, min=0, max=1)
        
        return Y

class GatingNetwork_MPNet(nn.Module):
    def __init__(self, input_size=None, output_size=None, **kwargs):
        super(GatingNetwork_MPNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=output_size, num_layers=2, batch_first=True)
   
    def forward(self, X):
        X = self.gru(X)
        X = X[0][:,-1,:]
        Y = nn.Softmax(dim=-1)(X)
        
        return Y

class ExpertMLP_MPNet(nn.Module):
    def __init__(self, rng, num_experts, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(ExpertMLP_MPNet, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_cuda = use_cuda
        self.w_l1, self.b_l1 = self.init_params(self.num_experts, self.input_size, self.hidden_size)
        self.w_l2, self.b_l2 = self.init_params(self.num_experts, self.hidden_size, self.hidden_size)
        self.w_l3, self.b_l3 = self.init_params(self.num_experts, self.hidden_size, self.output_size)
        
    def init_params(self, num_experts, input_size, output_size):
        w_bound = np.sqrt(6. / np.prod([input_size, output_size]))
        w = np.asarray(self.rng.uniform(low=-w_bound, high=w_bound, size=[num_experts, input_size, output_size]), dtype=np.float32)
        if self.use_cuda:
            w = nn.Parameter(torch.cuda.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.cuda.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        else:
            w = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        return w, b

    def linearlayer(self, inputs, weights, bias):
        return torch.sum(inputs[..., None] * weights, dim=1) + bias

    def forward(self, X, blending_coef):
        # inputs: B*input_dim
        # Blending_coef : B*experts
        w_l1 = torch.sum(blending_coef[..., None, None] * self.w_l1[None], dim=1)
        b_l1 = torch.matmul(blending_coef, self.b_l1)
        w_l2 = torch.sum(blending_coef[..., None, None] * self.w_l2[None], dim=1)
        b_l2 = torch.matmul(blending_coef, self.b_l2)
        w_l3 = torch.sum(blending_coef[..., None, None] * self.w_l3[None], dim=1)
        b_l3 = torch.matmul(blending_coef, self.b_l3) 
        
        X = F.elu(self.linearlayer(X, w_l1, b_l1))
        X = F.elu(self.linearlayer(X, w_l2, b_l2))
        Y = self.linearlayer(X, w_l3, b_l3)
        
        return Y

class MPNet_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, window_size, use_cuda=False, **kwargs):
        super(MPNet_MLP, self).__init__()
        # variable
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        # model        
        self.linear1 = nn.Linear(input_size*window_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, X):        
        # Recurrent Module
        X = X.reshape(X.shape[0], -1)        
        # Prediction Network        
        Z = F.elu(self.linear1(X))
        Z = F.elu(self.linear2(Z))
        Y = self.linear3(Z)
        Y = torch.clamp(Y, min=0, max=1)
        
        return Y

class MPNet_Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(MPNet_Transformer, self).__init__()
        # variable
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        # model        
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=hidden_size, batch_first=True), num_layers=4)                
        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        
    def forward(self, X):        
        Z = self.linear_input(X)
        Z = self.transformer(Z)[:,-1,:]         
        Y = self.linear_output(Z)
        Y = torch.clamp(Y, min=0, max=1)
        
        return Y
            
class UGNet_MoE(nn.Module):
    def __init__(self, rng, num_experts, input_size, output_size, window_size, use_cuda=False, **kwargs):
        super(UGNet_MoE, self).__init__()
        # variable
        self.num_experts = num_experts
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        self.input_size_gating = 5+3
        self.input_size_moe = input_size - self.input_size_gating
        self.output_size = output_size
        # model
        self.gating_network = GatingNetwork_UGNet(input_size=self.input_size_gating, output_size=self.num_experts)        
        self.moe_network = ExpertMLP_UGNet(rng=rng, 
                                           num_experts=self.num_experts, 
                                           input_size=self.input_size_moe+(self.input_size_gating*window_size), 
                                           hidden_size=self.input_size_moe+(self.input_size_gating*window_size)//2, 
                                           output_size=self.output_size, 
                                           use_cuda=use_cuda)
        
    def forward(self, X):
        # Input
        X_gating_network = X[:, :, self.input_size_moe:]
        X_moe_network = torch.cat((X[:, -1, :self.input_size_moe], X[:, :, -self.input_size_gating:].reshape(X.shape[0], -1)), -1)

        # Gating Network
        Omega = self.gating_network(X_gating_network)

        # Motion Network
        Y = self.moe_network(X_moe_network, Omega)
        
        return Y

class GatingNetwork_UGNet(nn.Module):
    def __init__(self, input_size=None, output_size=None, **kwargs):
        super(GatingNetwork_UGNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=output_size, num_layers=2, batch_first=True)
        
    def forward(self, X):
        X = self.gru(X)
        X = X[0][:,-1,:]
        Y = nn.Softmax(dim=-1)(X)
        
        return Y

class ExpertMLP_UGNet(nn.Module):
    def __init__(self, rng, num_experts, input_size, hidden_size, output_size, use_cuda=False, **kwargs):
        super(ExpertMLP_UGNet, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.use_cuda = use_cuda
        self.w_l1, self.b_l1 = self.init_params(self.num_experts, self.input_size, self.hidden_size)
        self.w_l2, self.b_l2 = self.init_params(self.num_experts, self.hidden_size, self.hidden_size)
        self.w_l3, self.b_l3 = self.init_params(self.num_experts, self.hidden_size, self.output_size)
        
    def init_params(self, num_experts, input_size, output_size):
        w_bound = np.sqrt(6. / np.prod([input_size, output_size]))
        w = np.asarray(self.rng.uniform(low=-w_bound, high=w_bound, size=[num_experts, input_size, output_size]), dtype=np.float32)
        if self.use_cuda:
            w = nn.Parameter(torch.cuda.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.cuda.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        else:
            w = nn.Parameter(torch.FloatTensor(w), requires_grad=True)
            b = nn.Parameter(torch.FloatTensor(num_experts, output_size).fill_(0), requires_grad=True)
        return w, b

    def linearlayer(self, inputs, weights, bias):
        return torch.sum(inputs[..., None] * weights, dim=1) + bias

    def forward(self, X, blending_coef):
        # inputs: B*input_dim
        # Blending_coef : B*experts
        w_l1 = torch.sum(blending_coef[..., None, None] * self.w_l1[None], dim=1)
        b_l1 = torch.matmul(blending_coef, self.b_l1)
        w_l2 = torch.sum(blending_coef[..., None, None] * self.w_l2[None], dim=1)
        b_l2 = torch.matmul(blending_coef, self.b_l2)
        w_l3 = torch.sum(blending_coef[..., None, None] * self.w_l3[None], dim=1)
        b_l3 = torch.matmul(blending_coef, self.b_l3) 
        
        X = F.elu(self.linearlayer(X, w_l1, b_l1))
        X = F.elu(self.linearlayer(X, w_l2, b_l2))
        Y = self.linearlayer(X, w_l3, b_l3)
        
        return Y

class UGNet_MLP(nn.Module):
    def __init__(self, rng, num_experts, input_size, output_size, window_size, use_cuda=False, **kwargs):
        super(UGNet_MLP, self).__init__()
        # variable
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')                              
        self.input_size_feature = 5+3
        self.input_size_pose = input_size - self.input_size_feature
        self.input_size = self.input_size_pose+(self.input_size_feature*window_size)
        self.output_size = output_size
        # model
        self.linear1 = nn.Linear(self.input_size, self.input_size*4)
        self.linear2 = nn.Linear(self.input_size*4, self.input_size)
        self.linear3 = nn.Linear(self.input_size, self.output_size)        
        
    def forward(self, X):
        # Input
        X = torch.cat((X[:, -1, :self.input_size_pose], X[:, :, -self.input_size_feature:].reshape(X.shape[0], -1)), -1)
        Z = F.elu(self.linear1(X))
        Z = F.elu(self.linear2(Z))
        Y = self.linear3(Z)
        
        return Y

class UGNet_Transformer(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=False, **kwargs):
        super(UGNet_Transformer, self).__init__()
        # variable
        self.input_size_feature = 5+3 # target + progression                
        self.input_size_pose = input_size - self.input_size_feature                 
        # model
        self.linear_input_pose = nn.Linear(self.input_size_pose, 160)
        self.linear_input_feature = nn.Linear(self.input_size_feature, 160) 
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=160, nhead=4, dim_feedforward=160, batch_first=True), num_layers=4)        
        self.linear_output = nn.Linear(160, output_size)
        
    def forward(self, X):        
        Z_feature = self.linear_input_feature(X[:, :, self.input_size_pose:])
        Z_pose = self.linear_input_pose(X[:, -1, :self.input_size_pose])        
        Z = self.transformer(torch.cat((Z_feature, Z_pose.unsqueeze(1)), 1))[:,-1,:] 
        Y = self.linear_output(Z)        
        
        return Y

class UGNet_Diffusion(nn.Module):
    def __init__(self, T, **kwargs):
        super(UGNet_Diffusion, self).__init__()         
        self.T = T
        self.schedule_mode = "cosine"
        self.norm_type = "group_norm"
        self.act_type = "SiLU"
        self.frame_dim = 24+24+(5+3)*30
        # self.time_emb_dim = 64
        # self.hidden_dim = 256
        # self.layer_num = 4
        self.time_emb_dim = 64
        self.hidden_dim = 128
        self.layer_num = 10 
        
        self.model = NoiseDecoder(self.frame_dim, 
                                  self.hidden_dim, 
                                  self.time_emb_dim, 
                                  self.layer_num, 
                                  self.norm_type, 
                                  self.act_type)
        self.time_mlp = torch.nn.Sequential(PositionalEmbedding(self.time_emb_dim, 1.0),
                                            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim),
                                            nn.SiLU(),
                                            torch.nn.Linear(self.time_emb_dim, self.time_emb_dim))
        
        betas = self._generate_diffusion_schedule()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1. / alphas)))
        self.register_buffer("reciprocal_sqrt_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas_cumprod_m1", to_torch(np.sqrt(1. / alphas_cumprod -1)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    def _generate_diffusion_schedule(self, s=0.008):
        def f(t, T):
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
        
        if self.schedule_mode == 'cosine':  
            # from https://arxiv.org/abs/2102.09672  
            alphas = []
            f0 = f(0, self.T)

            for t in range(self.T + 1):
                alphas.append(f(t, self.T) / f0)
            
            betas = []

            for t in range(1, self.T + 1):
                betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
            return np.array(betas)
        
        elif self.schedule_mode == 'uniform':
            # from original ddpm paper
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start, beta_end, self.T)
        
        elif self.schedule_mode == 'quadratic':
            beta_start = 0.0001
            beta_end = 0.02
            return np.linspace(beta_start**0.5, beta_end**0.5, self.T) ** 2
        
        elif self.schedule_mode == 'sigmoid':
            beta_start = 0.0001
            beta_end = 0.02
            betas = np.linspace(-6, 6, self.T)
            return 1/(1+np.exp(-betas)) * (beta_end - beta_start) + beta_start
        
        else:
            assert(False), "Unsupported diffusion schedule: {}".format(self.schedule_mode)
    
    @torch.no_grad()
    def extract(self, a, ts, x_shape):
        b, *_ = ts.shape
        out = a.gather(-1, ts)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def add_noise(self, x, ts):
        return x + self.extract(self.sigma, ts, x.shape) * torch.randn_like(x)

    def perturb_x(self, x, ts, noise):
        return (
            self.extract(self.sqrt_alphas_cumprod, ts, x.shape) * x +
            self.extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape) * noise
        )   

    @torch.no_grad()
    def sample_ddpm(self, X):                
        last_x = torch.cat((X[:, -1, :24], X[:, :, -8:].reshape(X.shape[0], -1)), -1)
        x = torch.randn(X.shape[0], 24, device=last_x.device) 
        for t in range(self.T - 1, -1, -1):
            ts = torch.tensor([t], device = last_x.device).repeat(last_x.shape[0])
            te = self.time_mlp(ts)
            x = self.model(last_x, x, te).detach()                                        
            if t > 0:
                x = self.add_noise(x, ts)        
        return x   

    def forward(self, Y, X):
        next_x = Y
        cur_x = torch.cat((X[:, -1, :24], X[:, :, -8:].reshape(X.shape[0], -1)), -1)
        bs = cur_x.shape[0]
        device = cur_x.device
        
        ts = torch.randint(0, self.T, (bs,), device=device)
        
        time_emb = self.time_mlp(ts) 

        noise = torch.randn_like(next_x)
        perturbed_x = self.perturb_x(next_x, ts.clone(), noise)
        
        estimated = self.model(cur_x, perturbed_x, time_emb)
        return estimated

class NoiseDecoder(nn.Module):
    def __init__(self, frame_size, hidden_size, time_emb_size, layer_num, norm_type, act_type):
        super().__init__()

        self.input_size = frame_size
        layers = []
        for _ in range(layer_num): 
            if act_type == 'ReLU':
                non_linear = nn.ReLU()
            elif act_type == 'SiLU':
                non_linear = nn.SiLU() 
            linear = nn.Linear(hidden_size + 24+24+(5+3)*30 + time_emb_size, hidden_size)
            if norm_type == 'layer_norm':
                norm_layer = nn.LayerNorm(hidden_size)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(16, hidden_size)

            layers.append(norm_layer)
            layers.extend([non_linear, linear])
            
        self.net = nn.ModuleList(layers)
        self.fin = nn.Linear(24+24+(5+3)*30 + time_emb_size, hidden_size)
        self.fco = nn.Linear(hidden_size + 24+24+(5+3)*30  + time_emb_size, 24)
        self.act = nn.SiLU()

    def forward(self, cur_x, next_x, time_emb):        
        x0 = next_x
        y0 = cur_x
        x = torch.cat([cur_x, next_x, time_emb], dim=-1)
        x = self.fin(x)
        for i, layer in enumerate(self.net):
            if i % 3 == 2:
                x = torch.cat([x, x0, y0, time_emb], dim=-1)
                x = layer(x)
            else:
                x = layer(x)
        x = torch.cat([x, x0, y0, time_emb],dim=-1)         
        x = self.fco(x)
        return x

class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

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
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.5f}MB'.format(size_all_mb))    
    print(param_size)

if __name__ == '__main__':

    rng = np.random.RandomState(23456)
    WINDOW_SIZE=30
    BATCH_SIZE=32
    
    # MPNet_MoE
    NUM_EXPERTS=9
    INPUT_SIZE=6+5 # Target:2+3
    HIDDEN_SIZE=24
    OUTPUT_SIZE=3
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = MPNet_MoE(rng, NUM_EXPERTS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, False).to('cpu')
    output = model(input)
    print("MPNet_MoE output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")
    
    # MPNet_MLP
    NUM_EXPERTS=9
    INPUT_SIZE=6+5 # Target:2+3
    HIDDEN_SIZE=24
    OUTPUT_SIZE=3
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = MPNet_MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, WINDOW_SIZE, False).to('cpu')
    output = model(input)
    print("MPNet_MLP output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")
    
    # MPNet_Transformer
    NUM_EXPERTS=9
    INPUT_SIZE=6+5 # Target:2+3
    HIDDEN_SIZE=16
    OUTPUT_SIZE=3
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = MPNet_Transformer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, False).to('cpu')
    output = model(input)
    print("MPNet_Transformer output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")    
    
    # UGNet_MoE
    NUM_EXPERTS=9
    INPUT_SIZE=24+(5+3) # Target: 2+3, Phase: 3
    OUTPUT_SIZE=24
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = UGNet_MoE(rng, NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE, WINDOW_SIZE, False).to('cpu')
    output = model(input)
    print("UGNet_MoE output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")
        
    # UGNet_MLP
    NUM_EXPERTS=9
    INPUT_SIZE=24+(5+3) # Target: 2+3, Phase: 3
    OUTPUT_SIZE=24
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = UGNet_MLP(rng, NUM_EXPERTS, INPUT_SIZE, OUTPUT_SIZE, WINDOW_SIZE, False).to('cpu')
    output = model(input)
    print("UGNet_MLP output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")
    
    # UGNet_Transformer
    NUM_EXPERTS=9
    INPUT_SIZE=24+(5+3) # Target: 2+3, Phase: 3
    OUTPUT_SIZE=24
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = UGNet_Transformer(INPUT_SIZE, OUTPUT_SIZE, False).to('cpu')
    output = model(input)
    print("UGNet_Transformer output.shape="+str(output.shape))
    Get_Model_size(model)
    print("")      
    
    # UGNet_Diffusion
    INPUT_SIZE=24+(5+3) # Target: 2+3, Phase: 3
    OUTPUT_SIZE=24    
    HIDDEN_SIZE=192
    MODEL_STEP_SIZE=20
    STEP_SIZE=10  
    NUM_LAYERS=4    
    DEVICE = torch.device('cpu')   
    input = torch.ones((BATCH_SIZE, WINDOW_SIZE, INPUT_SIZE))        
    model = UGNet_Diffusion(T = MODEL_STEP_SIZE)
    estimated = model(torch.rand(BATCH_SIZE, OUTPUT_SIZE), input)
    print("UGNet_Diffusion output.shape="+str(estimated.shape))
    # print("UGNet_Diffusion alphas="+str(model.alphas))
    # print("UGNet_Diffusion betas="+str(model.betas))
    # print("UGNet_Diffusion betas.shape="+str(model.betas.shape))
    Get_Model_size(model)
    
    output = model.sample_ddpm(input)
    print("UGNet_Diffusion DDPM output="+str(output.shape))
    print("")    
           
    # ConvolutionalAutoEncoder
    model = ConvolutionalAutoEncoder(input_size=18, hidden_size=256, output_size=18, kernel_size=15)
    output = model(torch.rand(30,30,18), False)
    print("ConvolutionalAutoEncoder output.shape="+str(output.shape))    