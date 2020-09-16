import numpy as np
import pandas as pd
from lstm_classification import LstmLayer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(cm, labels_name, title):
    plt.imshow(cm, cmap=plt.cm.Blues)
    # label 坐标轴标签说明
    indices = range(len(cm))
    plt.xticks(indices, labels_name)
    plt.yticks(indices, labels_name)
    plt.colorbar()
    plt.xlabel('predict')
    plt.ylabel('true')
    plt.title(title)
    # 显示数据
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            plt.text(first_index, second_index, cm[first_index][second_index])

def data_set():
    fault_id = {'non':0, 'f1':1}
    id_to_onehot = {0:[1, 0], 1:[0, 1]}
    train_label = np.zeros((6000,2))
    train_label[:3000, :] = id_to_onehot[fault_id['non']]
    train_label[3000:, :] = id_to_onehot[fault_id['f1']]

    features = pd.read_csv('icing_features.csv')
    features = np.array(features.iloc[:, 1:21])

    x_train = np.mat(features[:6000, :])
    x_test = np.mat(features[6000:, :])
    y_train = np.mat(train_label)
    return x_train, x_test, y_train

if __name__ == '__main__':
    x_train, x_test, y_train = data_set()
    seq = 6000
    loss_min = 1
    l = LstmLayer(20, 100, 2, 0.1, 1e-3, 0.5)
    epoch = 200
    loss = np.zeros((epoch,1))
    for j in range(epoch):
        l.forward(x_train)
        los = np.sum(np.multiply(-np.log(l.y_list), y_train))/seq
        print(j, los)
        loss[j] = los
        if (los < loss_min):
            loss_min = los
            l_min = l
        e = l.y_list - y_train
        l.backward(e)
    l_min.forward(x_train)
    re_train = np.argmax(l_min.y_list, 1)
    l_min.forward(x_test)
    re_test = np.argmax(l_min.y_list, 1)
    ac = np.argmax(y_train, 1)

    labels = ['0', '1']
    cm1 = confusion_matrix(ac, re_train)
    cm2 = confusion_matrix(ac, re_test)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cm1, labels, "train Confusion Matrix")
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cm2, labels, "test Confusion Matrix")
    plt.show()
