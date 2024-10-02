from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse
import math

class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """
    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input

def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size)) #64
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size)) #64
        self.variance_epsilon = eps

    def forward(self, x): #[20, 431, 64]
        u = x.mean(-1, keepdim=True) #[20, 431, 1]
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels #64
        self.out_channels = out_channels #64
        self.lin1 = GraphLinear(in_channels, out_channels // 2) #[64, 32]
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2)#[32, 32]
        self.lin2 = GraphLinear(out_channels // 2, out_channels) #[32, 64]
        self.skip_conv = GraphLinear(in_channels, out_channels) #[64, 64]
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = BertLayerNorm(in_channels) #[64]
        self.norm1 = BertLayerNorm(out_channels // 2) #[32]
        self.norm2 = BertLayerNorm(out_channels // 2) #[32]

    def forward(self, x, adj):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2) #[20, 64, 431]
        y = self.lin1(trans_y).transpose(1,2) #[20, 494, 32]

        y = F.relu(self.norm1(y)) #[20, 431, 32]
        y = self.conv(y, adj)
        # trans_y = F.relu(self.norm2(y)).transpose(2, 3)
        trans_y = F.relu(self.norm2(y)).transpose(1, 2)
        # y = self.lin2(trans_y).transpose(2, 3)
        y = self.lin2(trans_y).transpose(1, 2)
        z = x+y

        return z

# class GraphResBlock(torch.nn.Module):
#     """
#     Graph Residual Block similar to the Bottleneck Residual Block in ResNet
#     """
#     def __init__(self, in_channels, out_channels, mesh_type='body'):
#         super(GraphResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.conv = GraphConvolution(self.in_channels, self.out_channels, mesh_type)
#         print('Use BertLayerNorm and GeLU in GraphResBlock')
#         self.norm = BertLayerNorm(self.out_channels)
#     def forward(self, x):
#         y = self.conv(x)
#         y = self.norm(y)
#         y = gelu(y)
#         z = x+y
#         return z

class GraphLinear(torch.nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """
    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels #64
        self.out_channels = out_channels #32
        self.W = torch.nn.Parameter(torch.FloatTensor(out_channels, in_channels)) #[64, 32]
        self.b = torch.nn.Parameter(torch.FloatTensor(out_channels)) #[32]
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]

class GraphConvolution(torch.nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""
    def __init__(self, in_features, out_features, bias=True): #[32, 32]
        super(GraphConvolution, self).__init__()
        device=torch.device('cuda')
        self.in_features = in_features
        self.out_features = out_features

        #[431, 431]
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))#[32, 32]
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6. / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj): #[20, 431, 32]
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight) #[431, 32]*[32, 32]
            output = torch.matmul(adj, support) #[431, 431]*[431, 32]
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight) #[431, 32]*[32, 32]
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(adj[i], support))

            output = torch.stack(output, dim=0) #[20, 431, 32]
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'