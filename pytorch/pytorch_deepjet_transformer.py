import torch
import torch.nn as nn
import numpy as np
import copy

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class InputConv(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)
        
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = self.bn2(sc + x2)
        else:
            x = self.bn2(x2)
        return x
    
class LinLayer(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)
        
        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = self.bn2(sc + x2)
        else:
            x = self.bn2(x2)
        return x

class InputProcess(nn.Module):

    def __init__(self, **kwargs):
        super(InputProcess, self).__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(16, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(16,64)
        self.cpf_conv2 = InputConv(64,128)
        self.cpf_conv3 = InputConv(128,128)
        
        self.npf_bn0 = torch.nn.BatchNorm1d(8, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(8,64)
        self.npf_conv2 = InputConv(64,128)
        self.npf_conv3 = InputConv(128,128)
        
        self.vtx_bn0 = torch.nn.BatchNorm1d(14, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(14,64)
        self.vtx_conv2 = InputConv(64,128)
        self.vtx_conv3 = InputConv(128,128)

#        self.pxl_bn0 = torch.nn.BatchNorm1d(7, eps = 0.001, momentum = 0.1)
 #       self.pxl_conv1 = InputConv(7,64)
  #      self.pxl_conv2 = InputConv(64,128)
   #     self.pxl_conv3 = InputConv(128,128)

        self.cpf_dropout = nn.Dropout(0.1)
        self.cpf_lin = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.cpf_bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
        self.cpf_act = nn.ReLU()
        
        self.npf_dropout = nn.Dropout(0.1)
        self.npf_lin = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.npf_bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
        self.npf_act = nn.ReLU()
        
        self.vtx_dropout = nn.Dropout(0.1)
        self.vtx_lin = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.vtx_bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
        self.vtx_act = nn.ReLU()

#        self.pxl_dropout = nn.Dropout(0.1)
 #       self.pxl_lin = torch.nn.Conv1d(64, 128, kernel_size=1)
  #      self.pxl_bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
   #     self.pxl_act = nn.ReLU()

    def forward(self, cpf, npf, vtx, pxl):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf_sc = self.cpf_dropout(self.cpf_bn(self.cpf_act(self.cpf_lin(cpf))))
        cpf = self.cpf_conv2(cpf, cpf_sc, skip = True)
        cpf = self.cpf_conv3(cpf, cpf, skip = True)
        cpf = torch.transpose(cpf, 1, 2)
        
        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
        npf_sc = self.npf_dropout(self.npf_bn(self.npf_act(self.npf_lin(npf))))
        npf = self.npf_conv2(npf, npf_sc, skip = True)
        npf = self.npf_conv3(npf, npf, skip = True)
        npf = torch.transpose(npf, 1, 2)
        
        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx_sc = self.vtx_dropout(self.vtx_bn(self.vtx_act(self.vtx_lin(vtx))))
        vtx = self.vtx_conv2(vtx, vtx_sc, skip = True)
        vtx = self.vtx_conv3(vtx, vtx, skip = True)
        vtx = torch.transpose(vtx, 1, 2)

#        pxl = self.pxl_bn0(torch.transpose(pxl, 1, 2))
 #       pxl = self.pxl_conv1(pxl, pxl, skip = False)
  #      pxl_sc = self.pxl_dropout(self.pxl_bn(self.pxl_act(self.pxl_lin(pxl))))
   #     pxl = self.pxl_conv2(pxl, pxl_sc, skip = True)
    #    pxl = self.pxl_conv3(pxl, pxl, skip = True)
     #   pxl = torch.transpose(pxl, 1, 2)

        return cpf, npf, vtx#, pxl
    
class DenseClassifier(nn.Module):

    def __init__(self, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)
             
        self.LinLayer1 = LinLayer(143,143)
        self.LinLayer2 = LinLayer(143,143)
        self.LinLayer3 = LinLayer(143,143)

    def forward(self, x):
        
        x = self.LinLayer1(x, x, skip = True)
        x = self.LinLayer2(x, x, skip = True)
        x = self.LinLayer3(x, x, skip = True)
        
        return x
    
class AttentionPooling(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

        self.ConvLayer = torch.nn.Conv1d(128, 1, kernel_size=1)
        self.Softmax = nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        a = self.ConvLayer(torch.transpose(x, 1, 2))
        a = self.Softmax(a)
        
        y = torch.matmul(a,x)
        y = torch.squeeze(y, dim = 1)
        y = self.dropout(self.bn(self.act(y)))
        
        return y
    
class HF_TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super(HF_TransformerEncoderLayer, self).__init__()
        #Initial Conv Layer
        self.InputConv = InputConv(d_model,d_model)
        #MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout0 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.ReLU()
        super(HF_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self,src,mask):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src = torch.transpose(src, 1, 2)
        src = torch.transpose(self.InputConv(src, src, skip = True), 1, 2)

        src2 = torch.transpose(src, 0, 1)
        src2 = self.self_attn(src2,src2,src2)[0]
        src2 = torch.transpose(src2, 0, 1)
        src = src + src2
        src = self.norm0(src)
        
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm1(src)
        return src
    
class HF_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(HF_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        #self.norm = norm

    def forward(self,src, mask):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        mask = mask

        for mod in self.layers:
            output = mod(output, mask)

        #if self.norm is not None:
        #    output = self.norm(output)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.ReLU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
    
class DeepJetTransformer(nn.Module):

    def __init__(self,
                 num_classes = 6,
                 **kwargs):
        super(DeepJetTransformer, self).__init__(**kwargs)
        
        self.InputProcess = InputProcess()
        self.DenseClassifier = DenseClassifier()
        self.Linear = nn.Linear(143, num_classes)
        self.pooling = AttentionPooling()

        self.global_bn = torch.nn.BatchNorm1d(15, eps = 0.001, momentum = 0.1)

        self.EncoderLayer = HF_TransformerEncoderLayer(d_model=128, nhead=8, dropout = 0.1)
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=3)

    def forward(self, global_vars, cpf, npf, vtx):

#        cpf[:,:,2] = torch.abs(cpf[:,:,2])
 #       npf[:,:,2] = torch.abs(npf[:,:,2])
        cpf = cpf[:,:,:16]
        pxl = 1
        
        mask = torch.cat((cpf[:,:,0] == 0.0, npf[:,:,0] == 0.0, vtx[:,:,0] == 0.0),dim = 1)
#        mask = torch.unsqueeze(mask, 2)
 #       mask = tile(mask, 0, 8)
  #      mask = torch.unsqueeze(mask[:,:,0] != 0, 2).type(torch.cuda.FloatTensor)
   #     mask = (torch.matmul(mask, torch.transpose(mask, 2, 1)) != 1).type(torch.cuda.FloatTensor)
    #    mask = mask - mask*1e30

        global_vars = self.global_bn(global_vars)
        cpf, npf, vtx = self.InputProcess(cpf[:,:,:], npf, vtx, pxl)
        
        enc = torch.cat((cpf,npf,vtx), dim = 1)
        enc = self.Encoder(enc, mask)
        enc = self.pooling(enc)
        
        x = torch.cat((global_vars, enc), dim = 1)
        x = self.DenseClassifier(x)
        
        output = self.Linear(x)
        
        return output
