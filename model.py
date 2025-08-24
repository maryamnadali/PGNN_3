import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb

####################### Basic Ops #############################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True, aggregation='mean', comb_mode='concat', num_heads=4):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable
        self.aggregation = aggregation
        self.comb_mode = comb_mode

        # Nonlinear Class is to compute s(u,v) but through neural network (in paper its not leranable)
        # Nonlinear class is used to compute s(v, u) as a learnable function
        # whereas in the original PGNN paper, s(v, u) = 1 / (d_sp(v, u) + 1) is fixed and non-learnable.
        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        #from 2d to d (used in forward function)
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        if self.aggregation == 'mlp':
          self.mlp_after_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
          )

        self.num_heads = num_heads
        if self.comb_mode == 'xattn':
            assert output_dim % self.num_heads == 0, "output_dim must be divisible by num_heads"
            self.head_dim = output_dim // self.num_heads
            self.Wq = nn.Linear(input_dim, output_dim, bias=False)
            self.Wk = nn.Linear(input_dim, output_dim, bias=False)
            self.Wv = nn.Linear(input_dim, output_dim, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1]))
        
        if self.comb_mode == 'concat':
            # ----- مسیر فعلی (بدون تغییر) -----
            messages = subset_features * dists_max.unsqueeze(-1)          # [n, m, input_dim]
            self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)  # [n, m, input_dim]
            messages = torch.cat((messages, self_feature), dim=-1)        # **concat** → [n, m, 2*input_dim]
            messages = self.linear_hidden(messages).squeeze()             # 2d → d → [n, m, d]
            messages = self.act(messages)                                 # [n, m, d]

        elif self.comb_mode == 'concat':  # comb_mode == 'xattn'
            # 1) گیت فاصله (مثل concat)
            gated = subset_features * dists_max.unsqueeze(-1)             # [n, m, input_dim]

            # 2) Q/K/V مشترک
            Q = self.Wq(feature)                # [n, d]
            K = self.Wk(gated)                  # [n, m, d]
            V = self.Wv(gated)                  # [n, m, d]

            # 3) چندسَری (فعلاً 4؛ آماده برای 1 یا 8)
            n, m, d = K.shape
            h = self.num_heads
            dh = d // h

            Qh = Q.view(n, h, dh)                             # [n, h, dh]
            Kh = K.view(n, m, h, dh)                          # [n, m, h, dh]
            Vh = V.view(n, m, h, dh)                          # [n, m, h, dh]

            # 4) امتیاز per‑anchor و گیت سیگموید (حفظ شکل n×m×d)
            score = (Kh * Qh.unsqueeze(1)).sum(-1) / (dh ** 0.5)   # [n, m, h]
            attn = torch.sigmoid(score).unsqueeze(-1)              # [n, m, h, 1]

            Mh = Vh * attn                                         # [n, m, h, dh]
            messages = Mh.reshape(n, m, d)                         # [n, m, d]
            messages = self.act(messages)

        # print("subset_features=",len(subset_features))
        # print("dists_max=",dists_max.size())
        # print("messages=",messages.size())
        # print("messages=",messages.size())

        # out_position=messages.mean(dim=1)# do this only for ppi,cora,email
        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out


        #out_structure = torch.mean(messages, dim=1)  # n*d
        if self.aggregation == 'mean':
            out_structure = torch.mean(messages, dim=1)
        elif self.aggregation == 'sum':
            out_structure = torch.sum(messages, dim=1)
        elif self.aggregation == 'max':
            out_structure = torch.max(messages, dim=1)[0]
        elif self.aggregation == 'min':
            out_structure = torch.min(messages, dim=1)[0]    
        elif self.aggregation == 'mlp':
          pooled = torch.mean(messages, dim=1)      # می‌تونی sum رو هم تست کنی
          out_structure = self.mlp_after_pool(pooled) 
        else:
            raise NotImplementedError(f"Unknown aggregation: {self.aggregation}")

        return out_position, out_structure


### Non linearity
### Nonlinear Class is to compute s(u,v) but through nn (in paper its not leranable)
### Nonlinear class is used to compute s(v, u) as a learnable function,
### whereas in the original PGNN paper, s(v, u) = 1 / (d_sp(v, u) + 1) is fixed and non-learnable.
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x




####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GIN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x



class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, aggregation='mean', comb_mode='concat', **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.aggregation = aggregation
        self.comb_mode = comb_mode
                     
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim, aggregation=self.aggregation, comb_mode=self.comb_mode)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim, aggregation=self.aggregation, comb_mode=self.comb_mode)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim, aggregation=self.aggregation, comb_mode=self.comb_mode) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim, aggregation=self.aggregation, comb_mode=self.comb_mode)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position

class MyAttention(nn.Module):
    def __init__(self, feature_dim):  # feature_dim = Embedding dimension
        super(MyAttention, self).__init__()
        # self.W_Q = nn.Linear(feature_dim, feature_dim)
        # self.W_K = nn.Linear(feature_dim, feature_dim)
        # self.W_V = nn.Linear(feature_dim, feature_dim)
        # self.d_model = feature_dim

        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)

    def forward(self, P, S):
        # print("p=",P.size())
        # print("s=",S.size())
        # Initialize attention outputs
        P_Attention = torch.zeros_like(P)
        S_Attention = torch.zeros_like(S)

        for i in range(P.size(0)):
            T = torch.stack((P[i,:], S[i,:]), dim=0)
            attn_output, _ = self.attention(T, T, T)

            # Compute Q, K, V
            # Q = self.W_Q(T)  # Queries
            # K = self.W_K(T)  # Keys
            # V = self.W_V(T)  # Values

            # Compute attention scores
            # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

            # Apply softmax to get attention weights
            # A = F.softmax(scores, dim=-1)

            # Compute the final embeddings
            # H = torch.matmul(A, V)

            P_Attention[i,:] = attn_output[0]
            S_Attention[i,:] = attn_output[1]

        return P_Attention, S_Attention


class ATTSP(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
    # def __init__(self, input_dim, number_of_anchors, feature_dim=32, dropout=0.5):
        super(ATTSP, self).__init__()
        #------------------------------------GCN-----------
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)
        #--------------------------------------------------
        #--------------------PGNN-------------------------
        self.pgnn=PGNN(input_dim=input_dim, feature_dim=feature_dim,
                            hidden_dim=hidden_dim, output_dim=output_dim, comb_mode=kwargs.get('comb_mode', 'concat'))
        # if layer_num == 1:
        #     hidden_dim = output_dim
        # if feature_pre:
        #     self.linear_pre = nn.Linear(input_dim, feature_dim)
        #     self.pgnn_first = PGNN_layer(feature_dim, hidden_dim)
        # else:
        #     self.pgnn_first = PGNN_layer(input_dim, hidden_dim)
        # if layer_num>1:
        #     self.pgnn_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        #     self.pgnn_out = PGNN_layer(hidden_dim, output_dim)
        #--------------------------------------------------
        self.att = MyAttention(output_dim)

    def forward(self, data):
        #---------------------------------GCN
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)   # output of GCN
        #---------------------------------PGNN
        # x2 = data.x
        x_position=self.pgnn(data)
        # if self.feature_pre:
        #     x2 = self.linear_pre(x2)
        # x_position, x2 = self.pgnn_first(x2, data.dists_max, data.dists_argmax)
        # if self.layer_num == 1:
        #     return x_position
        # # x = F.relu(x) # Note: optional!
        # if self.dropout:
        #     x2 = F.dropout(x2, training=self.training)
        # for i in range(self.layer_num-2):
        #     _, x2 = self.pgnn_hidden[i](x2, data.dists_max, data.dists_argmax)
        #     # x = F.relu(x) # Note: optional!
        #     if self.dropout:
        #         x2 = F.dropout(x2, training=self.training)
        # x_position, x2 = self.pgnn_out(x2, data.dists_max, data.dists_argmax)
        # x_position = F.normalize(x_position, p=2, dim=-1) #--- output of PGNN
        #------------------------------------------------
        # print("position=",x_position.size())
        # print("structure",x.size())

        p, s = self.att(x_position, x)

        return  s + p
