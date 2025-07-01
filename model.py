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
    def __init__(self, input_dim, output_dim, dist_trainable=True, anchor_use_mode='concat'):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable
        self.anchor_use_mode = anchor_use_mode

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        # لایه های مناسب برای هر حالت
        if self.anchor_use_mode == 'attention':
            self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4)
            self.linear_hidden = nn.Linear(input_dim, output_dim)
        elif self.anchor_use_mode == 'concat':
            self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        else:  # sum, mean
            self.linear_hidden = nn.Linear(input_dim, output_dim)

        self.linear_out_position = nn.Linear(output_dim, 1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape(
            (dists_argmax.shape[0], dists_argmax.shape[1], feature.shape[1])
        )
        messages = subset_features * dists_max.unsqueeze(-1)  # [N, M, D]

        # ترکیب بر اساس حالت انتخابی
        if self.anchor_use_mode == 'attention':
            # Cross-Attention: Query=feature [N, D], Key/Value=messages [N, M, D]
            Q = feature.unsqueeze(1).permute(1, 0, 2)   # [1, N, D]
            K = messages.permute(1, 0, 2)               # [M, N, D]
            V = messages.permute(1, 0, 2)               # [M, N, D]
            attn_out, _ = self.attention(Q, K, V)
            attn_out = attn_out.permute(1, 0, 2).squeeze(1)  # [N, D]
            out_structure = self.act(self.linear_hidden(attn_out))
            # برای out_position: میشه از همون خروجی attention یا aggregate خاص استفاده کرد
            out_position = self.linear_out_position(out_structure).squeeze(-1)
        elif self.anchor_use_mode == 'concat':
            self_feature = feature.unsqueeze(1).repeat(1, messages.shape[1], 1)  # [N, M, D]
            messages_cat = torch.cat((messages, self_feature), dim=-1)            # [N, M, 2D]
            messages_cat = self.linear_hidden(messages_cat)
            messages_cat = self.act(messages_cat)
            out_structure = messages_cat.mean(dim=1)  # [N, D]
            out_position = self.linear_out_position(messages_cat).mean(dim=1).squeeze(-1)
        elif self.anchor_use_mode == 'sum':
            summed = messages + feature.unsqueeze(1)   # [N, M, D]
            summed = self.linear_hidden(summed)
            summed = self.act(summed)
            out_structure = summed.mean(dim=1)         # [N, D]
            out_position = self.linear_out_position(summed).mean(dim=1).squeeze(-1)
        elif self.anchor_use_mode == 'mean':
            meaned = ((messages + feature.unsqueeze(1)) / 2)  # [N, M, D]
            meaned = self.linear_hidden(meaned)
            meaned = self.act(meaned)
            out_structure = meaned.mean(dim=1)                # [N, D]
            out_position = self.linear_out_position(meaned).mean(dim=1).squeeze(-1)
        else:
            # حالت پیش فرض (concat)
            self_feature = feature.unsqueeze(1).repeat(1, messages.shape[1], 1)
            messages_cat = torch.cat((messages, self_feature), dim=-1)
            messages_cat = self.linear_hidden(messages_cat)
            messages_cat = self.act(messages_cat)
            out_structure = messages_cat.mean(dim=1)
            out_position = self.linear_out_position(messages_cat).mean(dim=1).squeeze(-1)

        return out_position, out_structure



### Non linearity
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
                 feature_pre=True, layer_num=2, dropout=True, anchor_use_mode='concat', **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim, anchor_use_mode=anchor_use_mode)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim, anchor_use_mode=anchor_use_mode)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([
                PGNN_layer(hidden_dim, hidden_dim, anchor_use_mode=anchor_use_mode)
                for _ in range(layer_num - 2)
            ])
            self.conv_out = PGNN_layer(hidden_dim, output_dim, anchor_use_mode=anchor_use_mode)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            x = F.normalize(x, p=2, dim=-1)   # نرمال‌سازی اختیاری
            return x                          # ← حالا [N, output_dim]
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x = F.normalize(x, p=2, dim=-1)   # نرمال‌سازی embedding نهایی
        return x  



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
                            hidden_dim=hidden_dim, output_dim=output_dim,)
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
