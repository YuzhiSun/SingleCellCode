# 这是一个示例 Python 脚本。
from collections import Counter

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

# 生成DEC可以使用的数据，gcn嵌入和表达谱加权, 该函数目前弃用因为无法将标签一起写入矩阵
def make_data_for_DEC():
    gene_express = pd.read_csv('../data/SingleCellData/human_top_geneMatrix.csv', index_col=0)
    gene_express.reset_index(inplace=True)
    gene_express['index'] = gene_express['index'].apply(lambda x: str(x).lower())

    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())
    gene_express_gcn = pd.merge(gene_express,gene_gcn,left_on='index',right_on='Symbl',how='inner')
    gene_express_gcn.drop(columns=['Symbl','index'], inplace=True)
    gene_express_matrix = gene_express_gcn.iloc[:,:-20]
    gcn_matrix = gene_express_gcn.iloc[:,-20:]
    gene_express_tensor = torch.tensor(gene_express_matrix.values)
    gcn_tensor = torch.tensor(gcn_matrix.values)
    gene_nums, sample_nums,gcn_dims = gene_express_tensor.size()[0], gene_express_tensor.size()[1],gcn_tensor.size()[1]
    gene_express_tensor = torch.expand_copy(gene_express_tensor,[gcn_dims,gene_nums,sample_nums])
    gene_express_tensor = torch.permute(gene_express_tensor,[0,2,1])
    gcn_tensor = torch.expand_copy(gcn_tensor,[sample_nums,gene_nums,gcn_dims])
    gcn_tensor = torch.permute(gcn_tensor,[2,0,1])
    express_gcn_tensor = gene_express_tensor + gcn_tensor

    return express_gcn_tensor


def concat_express_and_label_source_data():
    gene_express = pd.read_csv('../data/SingleCellData/human_top_geneMatrix.csv', index_col=0)
    gene_express = (gene_express - gene_express.min()) / (gene_express.max() - gene_express.min())
    gene_express.reset_index(inplace=True)
    gene_express['index'] = gene_express['index'].apply(lambda x: str(x).lower())

    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn_scala = gene_gcn.set_index('Symbl')
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala = (gene_gcn_scala - gene_gcn_scala.min()) / (gene_gcn_scala.max() - gene_gcn_scala.min())
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala.reset_index(inplace=True)
    gene_gcn_scala.rename(columns={'index':'Symbl'}, inplace=True)
    gene_gcn_scala.to_csv(r'../data/train/gcn_scala_feas.csv', index=False)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())
    target_gene = gene_gcn['Symbl'].tolist()

    target_gene_express = gene_express[gene_express['index'].isin(target_gene)]
    pd.DataFrame(target_gene_express['index'].tolist(),columns=['gene']).to_csv(r'../data/train/gene_for_train.csv', index=False)
    target_gene_express.set_index('index', inplace=True)
    target_gene_express = target_gene_express.T
    target_gene_express.reset_index(inplace=True)
    label = pd.read_csv('..\data\SingleCellData\human_cell.csv')


    target_gene_express_label = pd.merge(target_gene_express,label,left_on='index',right_on='barcode',how='inner')
    target_gene_express_label.drop(columns=['barcode'], inplace=True)
    # 剔除数量过少的
    label_count = target_gene_express_label[['assigned_cluster','index']].groupby(by=['assigned_cluster']).count()
    label_count = label_count[label_count['index'] > 260]
    label = label_count.reset_index()
    label.rename(columns={'assigned_cluster':'CellType','index':'count'}, inplace=True)
    label.reset_index(inplace=True)
    label.rename(columns={'index':'label'}, inplace=True)
    label.to_csv(r'../data/train/label_mapping_filtered.csv',index=False)
    target_gene_express_label = pd.merge(target_gene_express_label,label,left_on='assigned_cluster',right_on='CellType')
    target_gene_express_label.drop(columns=['assigned_cluster','CellType','index','count'], inplace=True)
    target_gene_express_label.to_csv(r'../data/train/gene_express_label_filtered.csv', index=False)
    print()

# 预处理成套数据
def concat_express_and_label_and_gcn_data(file_name = 'ssr1'):
    source_gene_express = pd.read_csv('../data/new2/Fibroid/new2_F_merge.csv', index_col=0)
    target_gene_express = pd.read_csv('../data/new2/Fibroid/Fib_merge.csv', index_col=0)
    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    src_label = pd.read_csv('../data/new2/Fibroid/new2_meta.csv')
    tar_label = pd.read_csv('../data/new2/Fibroid/Fib_meta.csv')
    source_gene_express.reset_index(inplace=True)
    source_gene_express['index'] = source_gene_express['index'].apply(lambda x: str(x).lower())
    target_gene_express.reset_index(inplace=True)
    target_gene_express['index'] = target_gene_express['index'].apply(lambda x: str(x).lower())
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())

    source_gene_set = set(source_gene_express['index'].tolist())
    target_gene_set = set(target_gene_express['index'].tolist())
    gcn_gene_set = set(gene_gcn['Symbl'].tolist())
    common_gene_set = list(source_gene_set & target_gene_set & gcn_gene_set)

    source_gene_express = source_gene_express[source_gene_express['index'].isin(common_gene_set)]
    source_gene_express = source_gene_express.sort_values(by=['index'])
    target_gene_express = target_gene_express[target_gene_express['index'].isin(common_gene_set)]
    target_gene_express = target_gene_express.sort_values(by=['index'])
    gene_gcn = gene_gcn[gene_gcn['Symbl'].isin(common_gene_set)]

    gene_gcn_scala = gene_gcn.set_index('Symbl')
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala = (gene_gcn_scala - gene_gcn_scala.min()) / (gene_gcn_scala.max() - gene_gcn_scala.min())
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala.reset_index(inplace=True)
    gene_gcn_scala.rename(columns={'index':'Symbl'}, inplace=True)
    gene_gcn_scala.to_csv(fr'../data/train/{file_name}/gcn_scala_feas.csv', index=False)

    source_gene_express.set_index('index', inplace=True)
    target_gene_express.set_index('index', inplace=True)
    source_gene_express = (source_gene_express - source_gene_express.min()) / (
            source_gene_express.max() - source_gene_express.min())
    target_gene_express = (target_gene_express - target_gene_express.min()) / (
                target_gene_express.max() - target_gene_express.min())

    # pd.DataFrame(source_gene_express['index'].tolist(),columns=['gene']).to_csv(fr'../data/train/{file_name}/gene_for_train.csv', index=False)


    source_gene_express = source_gene_express.T
    source_gene_express.reset_index(inplace=True)

    target_gene_express = target_gene_express.T
    target_gene_express.reset_index(inplace=True)

    source_gene_express_label = pd.merge(source_gene_express,src_label,left_on='index',right_on='Cell',how='inner')
    source_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)
    target_gene_express_label = pd.merge(target_gene_express, tar_label, left_on='index', right_on='Cell', how='inner')
    target_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)

    # 剔除数量过少的
    label_count = source_gene_express_label[['Cell_type','index']].groupby(by=['Cell_type']).count()
    target_label_count = target_gene_express_label[['Cell_type', 'index']].groupby(by=['Cell_type']).count()
    label_count = label_count[label_count['index'] > 0]
    label_count = label_count.reset_index()
    label_count.rename(columns={'Cell_type':'CellType','index':'count'}, inplace=True)
    label_count.reset_index(inplace=True)
    label_count.rename(columns={'index':'label'}, inplace=True)
    label_count.to_csv(fr'../data/train/{file_name}/label_mapping_filtered.csv',index=False)

    source_gene_express_label = pd.merge(source_gene_express_label,label_count,left_on='Cell_type',right_on='CellType')
    source_gene_express_label.drop(columns=['Cell_type','CellType','index','count'], inplace=True)
    source_gene_express_label.to_csv(fr'../data/train/{file_name}/source_gene_express_label.csv', index=False)

    valid_gene_express_label = pd.merge(target_gene_express_label, label_count, left_on='Cell_type',
                                         right_on='CellType')
    valid_gene_express_label.drop(columns=['Cell_type', 'CellType', 'index', 'count'], inplace=True)
    valid_gene_express_label.to_csv(fr'../data/train/{file_name}/valid_gene_express_label.csv', index=False)

    target_gene_express_label.drop(columns=['index'], inplace=True)
    target_gene_express_label.rename(columns={'Cell_type':'label'}, inplace=True)
    target_gene_express_label.to_csv(fr'../data/train/{file_name}/test_gene_express_label.csv', index=False)
#保留全部原始基因
def left_concat_express_and_label_and_gcn_data(file_name = 'ssr1_left'):
    filename_ = '../data/SRR3_result/SRR3_Myometrium'
    source_gene_express = pd.read_csv(f'{filename_}/SRR3_M_merge.csv', index_col=0)
    target_gene_express = pd.read_csv(f'{filename_}/My_merge.csv', index_col=0)
    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    src_label = pd.read_csv(f'{filename_}/SRR3_meta.csv')
    tar_label = pd.read_csv(f'{filename_}/My_meta.csv')
    source_gene_express.reset_index(inplace=True)
    source_gene_express['index'] = source_gene_express['index'].apply(lambda x: str(x).lower())
    target_gene_express.reset_index(inplace=True)
    target_gene_express['index'] = target_gene_express['index'].apply(lambda x: str(x).lower())
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())

    source_gene_set = set(source_gene_express['index'].tolist())
    target_gene_set = set(target_gene_express['index'].tolist())
    gcn_gene_set = set(gene_gcn['Symbl'].tolist())
    common_gene_set = list(source_gene_set & target_gene_set & gcn_gene_set)

    # source_gene_express = source_gene_express[source_gene_express['index'].isin(common_gene_set)]
    source_gene_express = source_gene_express.sort_values(by=['index'])
    # target_gene_express = target_gene_express[target_gene_express['index'].isin(common_gene_set)]
    target_gene_express = target_gene_express.sort_values(by=['index'])
    # gene_gcn = gene_gcn[gene_gcn['Symbl'].isin(common_gene_set)]

    gene_gcn_scala = gene_gcn.set_index('Symbl')
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala = (gene_gcn_scala - gene_gcn_scala.min()) / (gene_gcn_scala.max() - gene_gcn_scala.min())
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala.reset_index(inplace=True)
    gene_gcn_scala.rename(columns={'index':'Symbl'}, inplace=True)
    gene_gcn_scala.to_csv(fr'../data/train/{file_name}/gcn_scala_feas.csv', index=False)

    source_gene_express.set_index('index', inplace=True)
    target_gene_express.set_index('index', inplace=True)
    source_gene_express = (source_gene_express - source_gene_express.min()) / (
            source_gene_express.max() - source_gene_express.min())
    target_gene_express = (target_gene_express - target_gene_express.min()) / (
                target_gene_express.max() - target_gene_express.min())

    # pd.DataFrame(source_gene_express['index'].tolist(),columns=['gene']).to_csv(fr'../data/train/{file_name}/gene_for_train.csv', index=False)


    source_gene_express = source_gene_express.T
    source_gene_express.reset_index(inplace=True)

    target_gene_express = target_gene_express.T
    target_gene_express.reset_index(inplace=True)

    source_gene_express_label = pd.merge(source_gene_express,src_label,left_on='index',right_on='Cell',how='inner')
    source_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)
    target_gene_express_label = pd.merge(target_gene_express, tar_label, left_on='index', right_on='Cell', how='inner')
    target_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)

    # 剔除数量过少的
    label_count = source_gene_express_label[['Cell_type','index']].groupby(by=['Cell_type']).count()
    target_label_count = target_gene_express_label[['Cell_type', 'index']].groupby(by=['Cell_type']).count()
    label_count = label_count[label_count['index'] > 0]
    label_count = label_count.reset_index()
    label_count.rename(columns={'Cell_type':'CellType','index':'count'}, inplace=True)
    label_count.reset_index(inplace=True)
    label_count.rename(columns={'index':'label'}, inplace=True)
    label_count.to_csv(fr'../data/train/{file_name}/label_mapping_filtered.csv',index=False)

    source_gene_express_label = pd.merge(source_gene_express_label,label_count,left_on='Cell_type',right_on='CellType')
    source_gene_express_label.drop(columns=['Cell_type','CellType','index','count'], inplace=True)
    source_gene_express_label.to_csv(fr'../data/train/{file_name}/source_gene_express_label.csv', index=False)

    valid_gene_express_label = pd.merge(target_gene_express_label, label_count, left_on='Cell_type',
                                         right_on='CellType')
    valid_gene_express_label.drop(columns=['Cell_type', 'CellType', 'index', 'count'], inplace=True)
    valid_gene_express_label.to_csv(fr'../data/train/{file_name}/valid_gene_express_label.csv', index=False)

    target_gene_express_label.drop(columns=['index'], inplace=True)
    target_gene_express_label.rename(columns={'Cell_type':'label'}, inplace=True)
    target_gene_express_label.to_csv(fr'../data/train/{file_name}/test_gene_express_label.csv', index=False)


# 生成保留全部原始基因的数据集
def make_dot_features(gene_express_path = '../data/train/dataset_human_mus/source_gene_express_label.csv',
                      gcn_path = '../data/train/dataset_human_mus/gcn_scala_feas.csv',
                      target_path = '../data/train/dataset_human_mus/source_for_model.csv',
                      zipgcn=True):
    gene_express = pd.read_csv(gene_express_path)
    label_df = gene_express[['label','sample']]
    gene_express.drop(columns=['label','sample'], inplace=True)

    gene_express = gene_express.T
    gene_express.reset_index(inplace=True)
    gcn = pd.read_csv(gcn_path)
    gcn['Symbl'] = gcn['Symbl'].apply(lambda x: str(x).lower())
    gene_exp_gcn = pd.merge(gene_express, gcn, left_on='index', right_on='Symbl', how='left')
    gene_exp_gcn.fillna(0,inplace=True)
    gene_exp_gcn_tmp = gene_exp_gcn.drop(columns=['Symbl','index'])
    exp_tensor = torch.tensor(np.asarray(gene_exp_gcn_tmp.iloc[:,:-20]))
    gcn_tensor = torch.tensor(np.asarray(gene_exp_gcn_tmp.iloc[:,-20:]))
    dot_tensor = torch.mm(exp_tensor.T,gcn_tensor)
    gene_express.set_index('index',inplace=True)
    gene_express = gene_express.T
    dot_df = pd.DataFrame(np.asarray(dot_tensor))
    if zipgcn:
        dot_df = dot_df.sum(axis=1)
    dot_df = (dot_df - dot_df.min()) / (dot_df.max()  - dot_df.min())
    gene_express_dot = pd.concat([gene_express,dot_df,label_df],axis=1)
    gene_express_dot.to_csv(target_path, index=False)
# 废弃该函数
def concat_express_and_label_0711():
    version = '0711'
    gene_express = pd.read_csv('../data/SingleCellData/human_top_geneMatrix0711.csv', index_col=0)
    gene_express_mus = pd.read_csv('../data/SingleCellData/mouse_human_geneMatrix0711.csv', index_col=0)
    gene_express_mus.reset_index(inplace=True)
    mus_gene = list(set(gene_express_mus['index'].tolist()))
    gene_express.reset_index(inplace=True)
    # 取交集
    gene_express = gene_express[gene_express['index'].isin(mus_gene)]
    gene_express.set_index('index', inplace=True)

    gene_express = (gene_express - gene_express.min()) / (gene_express.max() - gene_express.min())
    gene_express.reset_index(inplace=True)
    gene_express['index'] = gene_express['index'].apply(lambda x: str(x).lower())

    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn_scala = gene_gcn.set_index('Symbl')
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala = (gene_gcn_scala - gene_gcn_scala.min()) / (gene_gcn_scala.max() - gene_gcn_scala.min())
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala.reset_index(inplace=True)
    gene_gcn_scala.rename(columns={'index':'Symbl'}, inplace=True)
    gene_gcn_scala.to_csv(r'../data/train/gcn_scala_feas.csv', index=False)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())
    target_gene = gene_gcn['Symbl'].tolist()

    target_gene_express = gene_express[gene_express['index'].isin(target_gene)]
    pd.DataFrame(target_gene_express['index'].tolist(),columns=['gene']).to_csv(fr'../data/train/gene_for_train_{version}.csv', index=False)
    target_gene_express.set_index('index', inplace=True)
    target_gene_express = target_gene_express.T
    target_gene_express.reset_index(inplace=True)
    label = pd.read_csv('..\data\SingleCellData\human_cell.csv')
    label_mapping = sorted(list(set(label['assigned_cluster'].tolist())))
    label_mapping = pd.DataFrame(label_mapping,columns=['CellType']).reset_index().rename(columns={'index':'label'})
    label_mapping.to_csv(fr'../data/train/label_mapping_{version}.csv', index=False)
    target_gene_express_label = pd.merge(target_gene_express,label,left_on='index',right_on='barcode',how='inner')
    target_gene_express_label.drop(columns=['barcode'], inplace=True)
    target_gene_express_label = pd.merge(target_gene_express_label,label_mapping,left_on='assigned_cluster',right_on='CellType')
    target_gene_express_label.drop(columns=['assigned_cluster','CellType','index'], inplace=True)
    target_gene_express_label.to_csv(fr'../data/train/gene_express_label_{version}.csv', index=False)
    print()
def sample_func(path ='../data/train/ssr1_zip_gcn/source_for_model.csv', ratio=0.1):
    data = pd.read_csv(path)
    y_data = data['label']
    train_x, test_x, train_y, test_y = train_test_split(data, y_data, test_size=ratio, random_state=32, stratify=y_data)
    a = Counter(train_y)  # 训练集里面各个标签的出现次数
    b = Counter(test_y)  # 测试集里面各个标签的出现次数
    c = Counter(y_data)

    test_x.to_csv(path[:-4]+'_sample.csv',index=False)
    print()
def make_train_valid_test():
    data = pd.read_csv(fr'../data/train/mouse10_100/source_for_model.csv')
    x_data = data.iloc[:,:-1]
    y_data = data.iloc[:,-1]
    train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.5, random_state=32, stratify=y_data)
    train_data = pd.concat([train_x,train_y],axis=1)
    valid_data = pd.concat([test_x,test_y],axis=1)
    a = Counter(train_y)  # 训练集里面各个标签的出现次数
    b = Counter(test_y)  # 测试集里面各个标签的出现次数
    c = Counter(y_data)
    valid_data.to_csv(fr'../data/train/test_mouse/source_for_model.csv',index=False)

    print()

class GeneDataSet(data.Dataset):

    def __init__(self, gene_express_path, gcn_path,target_gene_path):
        gene_express = pd.read_csv(gene_express_path)
        self.gcn_feas = pd.read_csv(gcn_path)
        self.genes = pd.read_csv(target_gene_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]

    def make_matrix(self,gene_express,gcn,index):

        gcn['Symbl'] = gcn['Symbl'].apply(lambda x:str(x).lower())
        gene_exp = pd.DataFrame(gene_express.iloc[index,:])
        gene_exp.rename(columns={index:'express'}, inplace=True)
        gene_exp.reset_index(inplace=True)
        gene_exp_gcn = pd.merge(gene_exp,gcn,left_on='index',right_on='Symbl',how='left')
        gene_exp_gcn.drop(columns=['Symbl'],inplace=True)
        gene_exp_tensor = torch.tensor(gene_exp_gcn['express'])
        gene_gcn_feas = np.asarray(gene_exp_gcn.iloc[:,2:])
        gene_gcn_tensor = torch.tensor(gene_gcn_feas)
        gene_exp_tensor = torch.unsqueeze(gene_exp_tensor,dim=1)
        gene_exp_gcn_tensor = gene_gcn_tensor + gene_exp_tensor
        if not ((gene_exp_gcn_tensor.shape[0] == 1109) & (gene_exp_gcn_tensor.shape[1] == 20)):
            print('error!!!!!!!!')
        return gene_exp_gcn_tensor

    def __getitem__(self, index):
        feature = self.make_matrix(self.feas, self.gcn_feas,index)
        feature = torch.reshape(feature,[-1])
        label = np.asarray(self.label)
        label = torch.tensor(label[index])
        feature = feature.to(torch.float32)
        label = label.to(torch.float32)
        return feature, label


    def __len__(self):
        return len(self.label)




class GeneNoGCNDataSet(data.Dataset):

    def __init__(self, gene_express_path, gcn_path,target_gene_path):
        gene_express = pd.read_csv(gene_express_path)
        self.gcn_feas = pd.read_csv(gcn_path)
        self.genes = pd.read_csv(target_gene_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]

    def __getitem__(self, index):
        label = np.asarray(self.label)
        label = torch.tensor(label[index])
        feature = np.asarray(self.feas)
        feature = torch.tensor(feature[index,:])
        feature = feature.to(torch.float32)
        label = label.to(torch.float32)
        return feature, label


    def __len__(self):
        return len(self.label)

class GeneDotGCNDataSet(data.Dataset):

    def __init__(self, gene_express_path, testmode=False):
        gene_express = pd.read_csv(gene_express_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]
        self.test = testmode
    def __getitem__(self, index):
        feature = np.asarray(self.feas.iloc[index,:])
        feature = torch.tensor(np.asarray(feature))
        feature = feature.to(torch.float32)

        if self.test:

            return feature
        else:
            label = np.asarray(self.label)
            label = torch.tensor(label[index])
            label = label.to(torch.float32)
            return feature, label


    def __len__(self):
        return len(self.label)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    pass

    # filename = 'new2_F'
    # concat_express_and_label_and_gcn_data(filename)
    # # left_concat_express_and_label_and_gcn_data(filename)
    #
    # plus = '_gcn'
    # zipgcn = True
    # gene_express_path = f'../data/train/{filename}/source_gene_express_label.csv'
    # gcn_path = f'../data/train/{filename}/gcn_scala_feas.csv'
    # target_path = f'../data/train/{filename}{plus}/source_for_model.csv'
    # make_dot_features(gene_express_path,gcn_path,target_path,zipgcn)
    # gene_express_path = f'../data/train/{filename}/valid_gene_express_label.csv'
    # target_path = f'../data/train/{filename}{plus}/valid_for_model.csv'
    # make_dot_features(gene_express_path, gcn_path, target_path, zipgcn)
    # gene_express_path = f'../data/train/{filename}/test_gene_express_label.csv'
    # target_path = f'../data/train/{filename}{plus}/test_for_model.csv'
    # make_dot_features(gene_express_path, gcn_path, target_path, zipgcn)


    # gene_express_path = '../data/train/gene_express_label_gcn.csv'
    # dataset1 = GeneDotGCNDataSet(gene_express_path)
    # #检验生成的数据是否形状一致
    # for i in range(0,dataset1.__len__()):
    #     if i % 500 == 0:print(f'{i} is ok')
    #     dataset1.__getitem__(i)
    # dataset1.__len__()
    path = '../data/train/new2_F_gcn/valid_for_model.csv'
    ratio = 0.1
    sample_func(path, ratio)
    # make_train_valid_test()
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
