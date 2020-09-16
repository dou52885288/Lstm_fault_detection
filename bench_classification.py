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
            plt.text(second_index, first_index, cm[first_index][second_index])
def data_set():
    fault_id = {'f1':1, 'f2':2, 'f3':3, 'f4':4, 'non':5, 'f5':6, 'f6':7, 'f7':8}
    id_to_onehot = {1:[1, 0, 0, 0, 0, 0, 0, 0],
                    2:[0, 1, 0, 0, 0, 0, 0, 0],
                    3:[0, 0, 1, 0, 0, 0, 0, 0],
                    4:[0, 0, 0, 1, 0, 0, 0, 0],
                    5:[0, 0, 0, 0, 1, 0, 0, 0],
                    6:[0, 0, 0, 0, 0, 1, 0, 0],
                    7:[0, 0, 0, 0, 0, 0, 1, 0],
                    8:[0, 0, 0, 0, 0, 0, 0, 1]}
    label = np.zeros((8000,8))
    label[0:1000, :] = id_to_onehot[fault_id['f1']]
    label[1000:2000, :] = id_to_onehot[fault_id['f2']]
    label[2000:3000, :] = id_to_onehot[fault_id['f3']]
    label[3000:4000, :] = id_to_onehot[fault_id['f4']]
    label[4000:5000, :] = id_to_onehot[fault_id['non']]
    label[5000:6000, :] = id_to_onehot[fault_id['f5']]
    label[6000:7000, :] = id_to_onehot[fault_id['f6']]
    label[7000:8000, :] = id_to_onehot[fault_id['f7']]

    features = pd.read_csv('features1.csv')
    features = np.array(features.iloc[:,1:16])

    x_train = np.mat(features[:8000,:])
    x_test = np.mat(features[8000:,:])
    y = np.mat(label)
    return x_train, x_test, y

if __name__ == '__main__':
    x_train, x_test, y_train = data_set()
    seq = 8000
    loss_min = 3
    l = LstmLayer(15, 100, 8, 0.1, 1e-3, 0.5)
    epoch = 400
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
    C1 = confusion_matrix(ac, re_train)
    C2 = confusion_matrix(ac, re_test)

    labels = ['0','1','2','3','4','5','6','7']
    cm1 = confusion_matrix(ac, re_train)
    print(cm1)
    cm2 = confusion_matrix(ac, re_test)
    print(cm2)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cm1, labels, "train Confusion Matrix")
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(cm2, labels, "test Confusion Matrix")
    plt.show()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
