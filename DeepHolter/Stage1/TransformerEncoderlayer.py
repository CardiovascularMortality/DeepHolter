import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import torch.nn.functional as F
'''
Implemented by https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
        d_model: The dimensionality of the input.
        num_heads: The number of attention heads in the multi-head attention.
        d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
        dropout: The dropout rate used for regularization.
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout1d(dropout)
        self.dropout2 = nn.Dropout1d(dropout)
    def forward(self, q,k,v,mask):
        '''
        x: The input tensor of shape (batch_size, seq_len, d_model).
        '''
        x = self.norm1(q + self.dropout1(self.self_attn(q, k, v, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x
    
class PositionalEncoding(nn.Module):
    '''
    isn't appropriate for signal embedding and feature embedding
    '''
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x, mask):

        false_count = (~mask).sum(dim=1)

        batch_size, seq_length, d_model = x.shape

        pe_expanded = self.pe.repeat(batch_size, 1, 1)  # (batch_size, max_seq_length, d_model)

        mask_extended = torch.arange(seq_length,device=x.device).unsqueeze(0).expand(batch_size, seq_length) < false_count.unsqueeze(1)

        pe_out = x + (pe_expanded * mask_extended.unsqueeze(-1))

        return pe_out

class BiphasePositionalEncoding(nn.Module):
    def __init__(self, d_model:int=512, max_hour_length:int=73):
        super(BiphasePositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_hour_length, d_model)
        hour_idx = torch.arange(0, max_hour_length,dtype=torch.float).unsqueeze(1)
        hours = hour_idx % 24
        # days = hour_idx // 24
        freq_levels = torch.tensor([2**(i//(d_model//4)) for i in range(d_model//2)]) #(d_model//2,)
        mod_factors = torch.tensor([1+(i//(d_model//4)) for i in range(d_model//2)]) #(d_model//2,)
        angle = (2*math.pi*hours/24)*freq_levels+(math.pi*hours/2)*mod_factors 
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor, hour_onehot:torch.Tensor)->torch.Tensor:
        hour = torch.argmax(hour_onehot,dim=-1)
        return (x + self.pe[:,hour])[0]
    
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, max_seq_length):
        '''
        d_model: The dimensionality of the input.
        num_heads: The number of attention heads in the multi-head attention.
        num_layers: The number of encoder layers.
        d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
        max_seq_length: The maximum sequence length.
        dropout: The dropout rate used for regularization.
        '''
        super(Transformer, self).__init__()
        # self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.positional_encoding = BiphasePositionalEncoding(d_model, max_seq_length)
        self.num_heads = num_heads
        self.signal_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.feature_cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # self.cls_token = nn.init.normal_(self.cls_token, mean=0.0, std=1)  
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout1d(dropout)
    def forward(self, x, mask, hour_onehot, feature_embed):
        '''
        x: The input tensor of shape (batch_size, seq_len, d_model). x is the embedding trained from Resnet34(signal) or Linear(feature).
        '''
        # print(x.shape)

        expand_cls_token = self.signal_cls_token.expand(x.size(0), -1,-1)  
        x = torch.cat((expand_cls_token, x), dim=1)  

        x = self.dropout(self.positional_encoding(x,hour_onehot))

        mask = (mask==0)
        mask = mask.view(mask.size(0), 1, 1, mask.size(1)).expand(-1, self.num_heads, -1, -1)
        for encoder_layer in self.encoder_layers:
            feature_embed = encoder_layer(feature_embed,x,x,mask)#Signal+feature
            # x = encoder_layer(x,x,x,mask)#signal only
            # feature_embed = encoder_layer(feature_embed,feature_embed,feature_embed,mask) #feature only
        return feature_embed