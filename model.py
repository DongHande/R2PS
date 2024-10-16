# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
import torch.nn.functional as F

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, input_dim, output_dim = 1):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.decoder = nn.Linear(input_dim, output_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        return x

class Unixcoder(nn.Module):   
    def __init__(self, encoder):
        super(Unixcoder, self).__init__()
        self.encoder = encoder
    
    def forward(self, code_inputs=None, nl_inputs=None): 
        if code_inputs is not None:
            outputs = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

class Unixcoder_RR(nn.Module): 
    def __init__(self, encoder): # , cosqa_flag = 0, query_len = 128
        super(Unixcoder_RR, self).__init__()
        self.encoder = encoder
        hidden_size = self.encoder.pooler.dense.out_features
        self.decoder = nn.Linear(hidden_size, 1)
        self.head = RobertaLMHead(hidden_size, 1)
    
    def forward(self, inputs): 
        outputs = self.encoder(inputs,attention_mask=inputs.ne(1))[0]
        outputs = (outputs*inputs.ne(1)[:,:,None]).sum(1)/inputs.ne(1).sum(-1)[:,None]
        
        return torch.tanh(self.head(outputs))
