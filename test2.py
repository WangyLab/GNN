import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch_geometric
import torch_geometric.nn as gnn
import torch_geometric.data as gdata
from torch_geometric.data import Data
from torch.nn import Sequential as Seq, Linear as Lin, ReLU as ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import reset
from math import sqrt, isnan
from tqdm import tqdm
import sys
import os.path
# from ax.service.managed_loop import optimize as ax_opt
from torch.utils.tensorboard import SummaryWriter
import random
from torch_scatter import scatter_mean, scatter_sum

# 储存数据集
class BondEnergyDataset(gdata.InMemoryDataset):
    def __init__(self, file, root, transform=None, pre_transform=None):
        self.df = pd.read_csv(file)
        super(BondEnergyDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # root: Root directory where the dataset should be saved

    @property
    def raw_file_names(self):
        # 存储在raw_dir目录下
        return []

    @property
    def processed_file_names(self):
        # 存储在processed_dir目录下
        return ['corrected_th10.0.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        for _, row in self.df.iterrows():
            # x:用于储存每个节点的特征，shape:[num_nodes, num_node_features]
            # edge_index:用于储存节点之间的边，shape:[2, num_edges]
            # edge_attr:储存边的特征，shape:[num_edges, num_edge_features]
            x = torch.tensor(eval(row['Atoms']), dtype=torch.float)
            edge_index = torch.tensor(eval(row['Connections']), dtype=torch.long)

            e = torch.tensor([[row['Energy']]], dtype=torch.float)
            # y = torch.tensor([[row['EnergyDiff']]], dtype=torch.float)
            # avg = torch.tensor([[row['EnergyAverage']]], dtype=torch.float)

            edge_attr = torch.tensor(eval(row['EdgeFeatures']), dtype=torch.float)
            edge_type = torch.tensor(eval(row['EdgeTypes']), dtype=torch.long)

            # bond_level = torch.tensor([[row['BondLevel']]], dtype=torch.float)
            g = torch.tensor([[1]], dtype=torch.float)

            key = {'Mol': row['Molecule'], 'Large': row['Large']}

            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=None, edge_attr=edge_attr, edge_type=edge_type)
            data.g = g
            data.e = e
            data.key = key
            data.avg = None
            # data.bond_level = bond_level
            data_list.append(data)

        data, slices = self.collate(data_list)
        #将数据进行整合，list中的全部data全部融合到一个data里边，slices好像是切片了，存的都是片段性的idx
        torch.save((data, slices), self.processed_paths[0])


train_set = BondEnergyDataset(file="./train3_th2.5.csv", root='./train')
test_set = BondEnergyDataset(file="./test3_th2.5.csv", root='./test')

batch_size = 64
train_set_loader = gdata.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)


# class ICLin(nn.Sequential):
#     def __init__(self, dim0, dim1, p=0.1):
#         super().__init__(nn.Dropout(p), nn.BatchNorm1d(dim0), nn.Linear(dim0, dim1))


class EdgeModel(nn.Module):
    def __init__(self, num_node_features, num_edge_features, out_features):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Seq(Lin(num_node_features + num_node_features + num_edge_features, 128), ReLU(), Lin(128, 128),
                            ReLU(), Lin(128, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features_out, out_features):
        # num_edge_features_out为EdgeModel输出
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(num_node_features + num_edge_features_out, 256), ReLU(), Lin(256, 256), ReLU(),
                              Lin(256, 256))
        self.node_mlp_2 = Seq(Lin(num_node_features + 256, 256), ReLU(), Lin(256, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)  #将原子属性和边的属性融合一起
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        #col里边存放的都是idx，直接指明srctensor的选定维度的idx数据在outtensor中的idx位置，如果有多个数据在一个idx，就按照减少方法来减少数据量为一个
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, num_node_features, num_global_features, out_channels):
        super(GlobalModel, self).__init__()
        self.global_mlp = Seq(Lin(num_global_features + num_node_features, 256), ReLU(), Lin(256, 256), ReLU(),
                              Lin(256, out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)  #按照batch来平均
        return self.global_mlp(out)


device = torch.device('cpu')
def get_edge_batch_tensor(batch_size):
    arr = []
    for i in range(batch_size):
        arr.append(i)
        arr.append(i)
    return torch.tensor(arr, dtype=torch.long, device=device)
# 由于原子之间建连接为无向图，所以要append两次
edge_batch = get_edge_batch_tensor(batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn_node = gnn.BatchNorm(train_set.num_node_features)
        self.bn_edge = gnn.BatchNorm(train_set.num_edge_features)
        self.meta1 = gnn.MetaLayer(EdgeModel(train_set.num_node_features, train_set.num_edge_features, 512),
                                   NodeModel(train_set.num_node_features, 512, 128),
                                   GlobalModel(128, 1, 128))
        self.meta2 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta3 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta4 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta5 = gnn.MetaLayer(EdgeModel(128, 512, 512),
                                   NodeModel(128, 512, 128),
                                   GlobalModel(128, 128, 128))
        self.meta6 = gnn.MetaLayer(EdgeModel(128, 512, 128),
                                   None,
                                   None)
        self.lin1 = nn.Linear(128, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, e, g, edge_type = data.x, data.edge_index, data.edge_attr, data.g, data.edge_type
        x = self.bn_node(x)   # 对原子特征进行批归一化
        e = self.bn_edge(e)   # 对边特征进行批归一化
        x, e, g = self.meta1(x, edge_index, e, g, data.batch)
        x, e, g = self.meta2(x, edge_index, e, g, data.batch)
        x, e, g = self.meta3(x, edge_index, e, g, data.batch)
        x, e, g = self.meta4(x, edge_index, e, g, data.batch)
        x, e, g = self.meta5(x, edge_index, e, g, data.batch)
        _, e, _ = self.meta6(x, edge_index, e, g, data.batch)

        count = data.e.size(0)
        eb = edge_batch
        if count != batch_size:
            eb = get_edge_batch_tensor(count)

        selected = e.t()[:, edge_type == 0].t()   #根据edge_type来选择想要的数据
        selected = scatter_sum(selected, eb, dim=0)  #该网络只使用了0来选择键连
        # y = torch.cat([selected, avg], dim=1)
        y = selected
        y = self.lin1(y)
        y = self.lin2(F.relu(y))
        return y


def train_evaluate(name, params, train_epochs=100, continue_epoch=-1):
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_func = nn.MSELoss()

    writer = SummaryWriter("log/")
    # if not os.path.isdir("result/" + name):
    #     os.mkdir("result/" + name)

    if continue_epoch >= 0:
        loaded = torch.load('epoch{}.pt'.format(continue_epoch))
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['opt'])
        for k, v in params.items():
            optimizer.param_groups[0][k] = v
        print('actual lr: {}'.format(optimizer.param_groups[0]['lr']))

    for k, v in params.items():
        optimizer.param_groups[0][k] = v
        # optimizer.param_groups： 是长度为2的list，其中的元素是2个字典
        # optimizer.param_groups[0]： 长度为6的字典，包括['amsgrad', 'params', 'lr', 'betas', 'weight_decay', 'eps']
        # 这里k为'lr', v为1e-4
    print('actual lr: {}'.format(optimizer.param_groups[0]['lr']))

    print('===== Training with {}, {} epochs planned ====='.format(params, train_epochs))

    last_loss_eval = 0.0
    for epoch in range(continue_epoch + 1, continue_epoch + 1 + train_epochs):
        model.train()
        # model.train()的作用是启用Batch Normalization和Dropout
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()
        # model.train()是保证BN层能够用到每一批数据的均值和方差
        # 对于Dropout，model.train()是随机取一部分网络连接来训练更新参数
        for idx, data in enumerate(tqdm(train_set_loader, file=sys.stdout), start=epoch * len(train_set_loader)):
            # for data in train_set_loader:
            data = data.to(device)  # data为batch
            out = model(data)
            loss = loss_func(out, data.e)   #左边是pred y， 右边是真实y，也就是target
            writer.add_scalar('loss/train_batch', loss, idx)
            # 写到了tensorboard里
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
            if isnan(loss):
                print('')
                print('nan!')
                exit()

        model.eval()
        # model.eval()的作用是不启用Batch Normalization和Dropout
        # 如果模型中有BN层(Batch Normalization)和Dropout，在测试时添加model.eval()
        # model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变
        # 对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元
        # 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值
        # 这是model中含有BN层和Dropout所带来的的性质
        loss_func_eval = nn.MSELoss(reduction='sum')
        loss_eval = 0.0
        out_dataframe = pd.DataFrame(columns=['Mol', 'Large', 'Source', 'Predicted'])
        for edata in tqdm(test_set_loader, file=sys.stdout):
            edata = edata.to(device)  # edata为batch，最后的batch为54
            out = model(edata)
            loss_eval = loss_eval + loss_func_eval(out, edata.e).detach().cpu().numpy()
            # out为预测的，edata.e为测试集的键能，每一轮的损失都加到loss_eval
            src_eval = edata.e.cpu().numpy()
            #原本可能为gputensor，需要先转换为cpu tensor再转换为numpy，但是如果tensor原来是标量的话，可以直接.item()取出来
            predicted_eval = out.detach().cpu().numpy()
            for i in range(edata.num_graphs):
                out_dataframe = out_dataframe.append(pd.DataFrame({
                    'Mol': edata.key['Mol'][i],
                    'Large': edata.key['Large'][i],
                    'Source': src_eval[i, 0],
                    'Predicted': predicted_eval[i, 0]
                }, index=[0]), ignore_index=True)
        out_dataframe.to_csv('epoch{:d}.csv'.format(epoch), index=False)
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict()}, 'epoch{:d}.pt'.format(epoch))
        last_loss_eval = loss_eval / len(test_set)
        print('epoch{:d} completed with loss {}'.format(epoch, last_loss_eval))
        writer.add_scalar('loss/test_epoch', last_loss_eval, epoch)

        if isnan(last_loss_eval) or last_loss_eval > 1e7:
            return 1e7
    return last_loss_eval


# best_parameters, values, experiment, model = ax_opt(
#     parameters=[
#         {"name": "lr", "type": "range", "bounds": [1e-7, 1e-3], "log_scale": True},
#         {"name": "momentum", "type": "range", "bounds": [0.5, 1.0]},
#     ],
#     evaluation_function=lambda params: train_evaluate(params, 20),
#     minimize=True
# )

# print(">>>> Best parameters: {}".format(best_parameters))

# train_evaluate("metaconv", {'lr': 1e-5}, 100)
# train_evaluate("metaconv", {'lr': 2e-5}, 100, continue_epoch=99)
# train_evaluate("metaconv", {'lr': 5e-5}, 100, continue_epoch=199)
# train_evaluate("metaconv", {'lr': 1e-4}, 100, continue_epoch=299)
train_evaluate("metaconv", {'lr': 1e-4}, 1)