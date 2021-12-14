from helpers import *
from Projet_KTAS_009 import data_tr_PCA, data_vl_PCA, data_ts_PCA,  data_tr_MDS, data_vl_MDS, data_ts_MDS, data_tr_TSNE, data_vl_TSNE, data_ts_TSNE

if __name__ == "__main__":
    train, valid, test = get_train_valid_reduced(32, data_tr_TSNE, data_vl_TSNE, data_ts_TSNE)
    lr = 0.02
    epochs = 10
    net = Net(num_feature=16)
    momentum = 0.9
    optim = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model = Model(net, optim, criterion, batch_metrics=["accuracy"])
    model.fit_generator(train, valid, epochs=epochs)
    model.evaluate_generator(test)