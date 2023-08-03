import click
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import uuid
import matplotlib.pyplot as plt
from util import GeneDataSet, GeneNoGCNDataSet, GeneDotGCNDataSet
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy
from sklearn.model_selection import train_test_split
@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=200,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=300,
)
@click.option(
    "--trans-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=100,
)
@click.option(
    "--dec-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=50,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, trans_epochs, dec_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope
    filename='ssr3_M_gcn'
    root_path = f'../data/train/{filename}/'
    label_path = f'../data/train/{filename[:-4]}/'
    dataset_path = root_path + 'test_for_model.csv'
    origin_data = pd.read_csv(dataset_path)
    sure_label = pd.read_csv(label_path+'label_mapping_filtered.csv')
    sure_label = sure_label['CellType'].tolist()
    ds_predict = GeneDotGCNDataSet(dataset_path,testmode=True)
    true = pd.read_csv(dataset_path)[['label']]
    true['label'] = true['label'].apply(lambda x: x if x in sure_label else 'unknown')
    true.rename(columns={'label':'true_type'}, inplace=True)
    label_mapping = true.drop_duplicates(ignore_index=True)
    label_mapping.reset_index(inplace=True)
    label_mapping.rename(columns={'index':'label'}, inplace=True)
    true = pd.merge(true,label_mapping,on='true_type',how='left')
    true = np.asarray(true['label'])
    transfer_model = torch.load(f'../data/train/model/{filename}.pth')
    transfer_model.eval()
    predicted = predict(
        ds_predict, transfer_model, 1024, silent=True, return_actual=False, cuda=cuda
    )

    predicted = predicted.cpu().numpy()
    origin_data['clustering_res'] = predicted
    reassignment, accuracy = cluster_accuracy(true, predicted)
    print("Final DEC accuracy: %s" % accuracy)

    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]  # TODO numpify
    confusion = confusion_matrix(true, predicted_reassigned)
    normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    )

    sns.heatmap(normalised_confusion)
    print()
    plt.show()
    origin_data[['sample','label','clustering_res']].to_csv(f'../data/train/{filename}/clustering_res_new.csv',index=False)
    print()
if __name__ == "__main__":
    main()
