import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import activation_layer, DNN

class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function name used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024,
                 device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        if split_half:
            linear_size = sum(
                    layer_size[:-1]) // 2 + layer_size[-1]
        else:
            linear_size = sum(layer_size)
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        #         for tensor in self.conv1ds:
        #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        # self.to(device)
        self.cin_linear = nn.Linear(linear_size, 1)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        result = self.cin_linear(result)
        return result
    
    
class Xdeepfm(nn.Module):
    def __init__(self, num_con, num_cat, categorical_cardinality=[], embed_dim=4, dropout=0.0, use_fm=True, use_dnn=True, cin_layer_size=(256, 128,), cin_split_half=True, cin_activation='relu'):
        super(Xdeepfm, self).__init__()

        self.categorical_embedding = nn.ModuleList([nn.Embedding(categorical_cardinality[i], embed_dim) for i in range(num_cat)])
        self.continuous_embedding = nn.Linear(num_con, 64)
        
        self.cin_layer_size = cin_layer_size
        
        self.use_fm = len(self.cin_layer_size) > 0 and num_con > 0
        self.use_dnn = use_dnn
        if self.use_fm:
            field_num = num_cat
            self.fm = CIN(field_num, self.cin_layer_size, cin_activation, cin_split_half)
        self.dnn_hidden_unit = [64, 64]
        self.dnn = DNN(embed_dim * num_cat + 64, self.dnn_hidden_unit, activation='relu', dropout_rate=dropout)
        
        
        if use_fm:
            self.w_fm = nn.Parameter(torch.randn((1,)) * 0.1)
        if use_dnn:
            self.w_dnn = nn.Parameter(torch.randn((1,)))
        # self.bias = nn.Parameter(torch.zeros((1,)))
        
        
    def forward(self, x_cat=None, x_con=None):
        assert x_con is not None or x_cat is not None, "x_con and x_cat cannot be both None"
        logit = torch.zeros([x_con.shape[0], 1]).to(x_con.device)
        cat_embedding = None 
        if self.use_fm and len(self.categorical_embedding) > 0:
            x_cat = x_cat.long()
            cat_embedding = [self.categorical_embedding[i](x_cat[:, i]).unsqueeze(1) for i in range(len(self.categorical_embedding))]
            logit += self.fm(torch.cat(cat_embedding, dim=1)) * self.w_fm
            
        if self.use_dnn:
            x_con = self.continuous_embedding(x_con)
            if cat_embedding is not None:
                x = torch.cat([x_con, torch.cat(cat_embedding, dim=2).squeeze()], dim=1)
            logit += self.dnn(x) * self.w_dnn        
            
        return logit 