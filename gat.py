import torch
import torch.nn as nn


# class Gat(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, heads=1):
#         super(Gat, self).__init__()
#         self.hidden_size = hidden_size
#         self.embed_layer = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.linear = nn.Linear(hidden_size, 1)
#         self.dropout = nn.Dropout(dropout)
        
#         self.gat_model = GAT(num_of_layers=num_layers, num_heads_per_layer=[heads]*num_layers, num_features_per_layer=[hidden_size]*num_layers+[1], add_skip_connection=True, bias=True, dropout=dropout, log_attention_weights=False)
        
#     def forward(self, x, adj):
#         x = self.embed_layer(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         bs = x.size(0)
#         out = []
#         for i in range(bs):
#             mask = adj[i] 
#             mask = mask.masked_fill(mask == 0, -1e9)
#             mask[mask==1] = 0
#             out.append(self.gat_model(x[i], mask)[0])
#         out = torch.stack(out)
#         return out
            
class GAT(torch.nn.Module):

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, add_skip_connection=True, bias=True,
                 dropout=0.6, log_attention_weights=False, v2=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'
        if v2:
            GATLayer = GATV2Layer
        else:
            GATLayer = GATLayerImp2  # pick the implementation you want! Imp1, Imp2 or Imp3!
        num_heads_per_layer = [1] + num_heads_per_layer  # trick - so that I can nicely create GAT layers below

        gat_layers = []  # collect GAT layers
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],  # consequence of concatenation
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                concat=True if i < num_of_layers - 1 else False,  # last GAT layer does mean avg, the others do concat
                activation=nn.ELU() if i < num_of_layers - 1 else None,  # last layer just outputs raw scores
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights
            )
            gat_layers.append(layer)

        # self.gat_net = nn.Sequential(
        #     *gat_layers,
        # )
        self.gat_net = nn.ModuleList(gat_layers)

    # data is just a (in_nodes_features, topology) tuple, I had to do it like this because of the nn.Sequential:
    # https://discuss.pytorch.org/t/forward-takes-2-positional-arguments-but-3-were-given-for-nn-sqeuential-with-linear-layers/65698
    def forward(self, x, adj):
        for net in self.gat_net:
            x, adj = net(x, adj)
        return x
        
    
    
    
class GATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.

    """

    head_dim = 2

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #
        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, 1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        bs, n, _, _ = out_nodes_features.size()
        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (bs, N, FIN) -> (bs, N, 1, FIN), out features are (bs, N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(2)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(bs, n, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (bs, N, NH, FOUT) -> (bs, N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(bs, n, self.num_of_heads * self.num_out_features)
        else:
            # shape = (bs, N, NH, FOUT) -> (bs, N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)
    
    
class GATLayerImp2(GATLayer):

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)

    def forward(self, in_nodes_features, adj):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #
        bs, num_of_nodes, _ = in_nodes_features.size()
        
        assert adj.shape == (bs, num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({bs},{num_of_nodes},{num_of_nodes}), got shape={adj.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(bs, num_of_nodes, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well

        scores_source = torch.sum((nodes_features_proj * self.scoring_fn_source), dim=-1, keepdim=True)
        scores_target = torch.sum((nodes_features_proj * self.scoring_fn_target), dim=-1, keepdim=True)
        # src shape = (bs, NH, N, 1) and trg shape = (bs, NH, 1, N)
        scores_source = scores_source.transpose(1, 2)
        scores_target = scores_target.permute(0, 2, 3, 1)

        # shape [bs, NH, N, N] where N - number of nodes in the graph
        all_scores = self.leakyReLU(scores_source + scores_target)
        
        connectivity_mask = adj.unsqueeze(1).repeat(1, self.num_of_heads, 1, 1)
        connectivity_mask = connectivity_mask.masked_fill(connectivity_mask == 0, -1e9)
        connectivity_mask[connectivity_mask==1] = 0
        # shape [bs, NH, N, N]
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)
        # batch matrix multiply, shape = (bs, NH, N, N) * (bs, NH, N, FOUT) -> (bs, NH, N, FOUT)
        out_nodes_features = torch.matmul(all_attention_coefficients, nodes_features_proj.transpose(1, 2))
        # shape = (bs, N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(0, 2, 1, 3)
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, adj)
    
    
class GATV2Layer(GATLayer):

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__(num_in_features, num_out_features, num_of_heads, concat, activation, dropout_prob,
                         add_skip_connection, bias, log_attention_weights)
        
        self.scoring_fn = nn.Parameter(torch.Tensor(num_of_heads, num_out_features))
        nn.init.xavier_uniform_(self.scoring_fn)
        
        self.linear_proj_source = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.linear_proj_target = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        nn.init.xavier_uniform_(self.linear_proj_source.weight)
        nn.init.xavier_uniform_(self.linear_proj_target.weight)

    def forward(self, in_nodes_features, adj):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #
        bs, num_of_nodes, _ = in_nodes_features.size()
        
        assert adj.shape == (bs, num_of_nodes, num_of_nodes), \
            f'Expected connectivity matrix with shape=({bs},{num_of_nodes},{num_of_nodes}), got shape={adj.shape}.'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (bs, N, FIN) * (bs, FIN, NH*FOUT) -> (bs, N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        source_nodes_features_proj = self.linear_proj_source(in_nodes_features).view(bs, num_of_nodes, self.num_of_heads, self.num_out_features)
        target_nodes_features_proj = self.linear_proj_target(in_nodes_features).view(bs, num_of_nodes, self.num_of_heads, self.num_out_features)
        
        source_nodes_features_proj = self.dropout(source_nodes_features_proj)  # in the official GAT imp they did dropout here as well
        target_nodes_features_proj = self.dropout(target_nodes_features_proj)  # in the official GAT imp they did dropout here as well
        
        scores_source = source_nodes_features_proj.unsqueeze(3)
        scores_target = target_nodes_features_proj.unsqueeze(3)
        # src shape = (bs, NH, N, 1, FOUT) and trg shape = (bs, NH, 1, N, FOUT)
        scores_source = scores_source.transpose(1, 2)
        scores_target = scores_target.permute(0, 2, 3, 1, 4)

        # shape [bs, NH, N, N, FOUT] -> [bs, NH, N, N] where N - number of nodes in the graph
        all_scores = self.leakyReLU(scores_source + scores_target)
        all_scores = torch.einsum('hd,bhnmd->bhnm', self.scoring_fn, all_scores)
        
        connectivity_mask = adj.unsqueeze(1).repeat(1, self.num_of_heads, 1, 1)
        connectivity_mask = connectivity_mask.masked_fill(connectivity_mask == 0, -1e9)
        connectivity_mask[connectivity_mask==1] = 0
        # shape [bs, NH, N, N]
        all_attention_coefficients = self.softmax(all_scores + connectivity_mask)
        # batch matrix multiply, shape = (bs, NH, N, N) * (bs, NH, N, FOUT) -> (bs, NH, N, FOUT)
        out_nodes_features = torch.matmul(all_attention_coefficients, nodes_features_proj.transpose(1, 2))
        # shape = (bs, N, NH, FOUT)
        out_nodes_features = out_nodes_features.permute(0, 2, 1, 3)
        out_nodes_features = self.skip_concat_bias(all_attention_coefficients, in_nodes_features, out_nodes_features)
        return (out_nodes_features, adj)
    
    
class Gat(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, heads=1, dropout=0):
        super(Gat, self).__init__()
        self.hidden_size = hidden_size
        self.proj_layer = nn.Linear(input_size, hidden_size)
        # self.gnn_model = GAT(hidden_size, hidden_size*2, num_layers, hidden_size, v2=False, dropout=dropout)
        self.gnn_model = GAT(num_of_layers=num_layers, num_heads_per_layer=[heads]*num_layers, num_features_per_layer=[hidden_size]*(num_layers + 1), add_skip_connection=True, bias=True, dropout=dropout, log_attention_weights=False)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, adj):
        x = self.proj_layer(x)
        out = self.gnn_model(x, adj)
        out = self.linear(out)
        return out