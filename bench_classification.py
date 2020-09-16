import numpy as np
import pandas as pd
from lstm_classification import LstmLayer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        print(los)
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
    fig = plt.figure(figsize=(20, 8))
    C1 = confusion_matrix(ac, re_train, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    C2 = confusion_matrix(ac, re_test, labels=[0, 1, 2, 3, 4, 5, 6, 7])

    ax1 = fig.add_subplot(241)
    sns.heatmap(C1, annot=True, ax=ax1)
    ax2 = fig.add_subplot(242)
    sns.heatmap(C2, annot=True, ax=ax2)

    ax1.set_title('confusion matrix')  #标题
    ax1.set_xlabel('predict')  # x轴
    ax1.set_ylabel('true')
    ax2.set_title('confusion matrix')  #标题
    ax2.set_xlabel('predict')  # x轴
    ax2.set_ylabel('true')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
