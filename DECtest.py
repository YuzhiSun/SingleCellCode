import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import uuid

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
    default=100,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=300,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    train_path = f'../data/train/dataset_human_mus/source_for_model.csv'
    valid_path = f'../data/train/dataset_human_mus/source_for_model.csv'

    ds_train, ds_val = GeneDotGCNDataSet(train_path), GeneDotGCNDataSet(valid_path)
    input_size = ds_val.feas.shape[1]
    label_kind = len(set(list(ds_val.label.values)))
    autoencoder = StackedDenoisingAutoEncoder(
        [input_size, 500, 500, 2000, 20], final_activation=None
    )
    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_train,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 50, gamma=0.1),
        corruption=0.2,
    )
    print("Training stage.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_train,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer,  50, gamma=0.1),
        corruption=0.2,
        update_callback=training_callback,
    )
    print("DEC stage.")
    model = DEC(cluster_number=label_kind, hidden_dimension=20, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.00005, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=150,
        batch_size=256,
        optimizer=dec_optimizer,
        # stopping_delta=0.000000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)
    if not testing_mode:
        predicted_reassigned = [
            reassignment[item] for item in predicted
        ]  # TODO numpify
        confusion = confusion_matrix(actual, predicted_reassigned)
        normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
        )
        confusion_id = uuid.uuid4().hex
        sns.heatmap(normalised_confusion).get_figure().savefig(
            "confusion_%s.png" % confusion_id
        )
        print("Writing out confusion diagram with UUID: %s" % confusion_id)
        writer.close()


if __name__ == "__main__":
    main()
