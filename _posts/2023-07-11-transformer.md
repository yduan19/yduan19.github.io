---
layout: post
title: Transformer from scratch using PyTorch
date: 2023-07-11 08:57:00-0400
description: based on the paper "Attention is all you need"
tags: jupyter-notebook
categories: machine_learning
related_posts: true
---


<h1 align="left" style="color:purple;font-size: 2em;" >Overview</h1>




* [1. Introduction](#section1)
* [2. Import libraries](#section2)
* [3. Basic components](#section3)
  - [Create Word Embeddings](#section4)
  - [Positional Encoding](#section5)
  - [Self Attention](#section6)
* [4. Encoder](#section7)
* [5. Decoder](#section8)
* [6. Testing our code](#section9)
* [7. Some useful resources](#section10)

<a class="anchor" id="section1"></a>
<h2 style="color:purple;font-size: 2em;">1. Introduction</h2>
<img src = "https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" width=330 height=470>

<a class="anchor" id="section2"></a>
<h2 style="color:purple;font-size: 2em;">2. Import Libraries</h2>

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
```



<a class="anchor" id="section3"></a>
<h2 style="color:purple;font-size: 2em;">3. Basic components</h2>

<a class="anchor" id="section4"></a>
<h2 style="color:purple;font-size: 1.5em;">Word Embeddings</h2>

Each word will be mapped to corresponding $$d_{model}=512$$ embedding vector. Suppose we have batch size of N=32 and sequence length of T=10 (10 words). The the output will be NxTxC (32X10X512).

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings, i.e. d_model
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector, i.e. (batch, seq_len, vocab_size)
        Returns:
            out: embedding vector (batch, seq_len, embed_dim)
        """
        out = self.embed(x)
        return out
```

<a class="anchor" id="section5"></a>
<h2 style="color:purple;font-size: 1.5em;"> Positional Encoding</h2>

$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$$ 

$$PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})$$

Here $$pos$$ is the position of the word in the sentence, and $$i$$ refers to position along embedding vector dimension.

```python
class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding, d_model
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))

        # adding batch dimension for broadcasting
        pe = pe.unsqueeze(0) 

        # register buffer in Pytorch ->
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        # x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + self.pe[:, :seq_len].detach()
        return x

# src = torch.zeros(2,12)
# model = PositionalEmbedding(2,12)
# print(model(src).shape)
```

<a class="anchor" id="section6"></a>
<h2 style="color:purple; font-size: 1.5em;"> Self Attention</h2>

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output

```