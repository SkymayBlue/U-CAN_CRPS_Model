import os
import pandas as pd
import numpy as np
import random
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib not found!")
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils as np_utils
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, \
    precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
try:
    from skrebate import ReliefF, SURF
except ModuleNotFoundError:
    print("skrebate not found!")
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.utils import Sequence
SEEDS = 46946
random.seed(SEEDS)
np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)
tf.compat.v1.set_random_seed(SEEDS)

TOTAL_CLUSTER = ["CRPS1", "CRPS2", "CRPS3", "CRPS4", "CRPS5", "CRPSn"]
CLUSTER_NAME = {0: TOTAL_CLUSTER[0], 1: TOTAL_CLUSTER[1], 2: TOTAL_CLUSTER[2], 3: TOTAL_CLUSTER[3],
                4: TOTAL_CLUSTER[5], 5: TOTAL_CLUSTER[4]}
REVER_CLUSTER_NAME = {0: TOTAL_CLUSTER[0], 1: TOTAL_CLUSTER[1], 2: TOTAL_CLUSTER[2], 3: TOTAL_CLUSTER[3],
                      4: TOTAL_CLUSTER[4], 5: TOTAL_CLUSTER[5]}


# check result dir exists!
def check_dir(workdir):
    if not os.path.exists(workdir):
        print("the workdir not exist!")
        exit(0)
    else:
        return True


def makedir(workdir):
    if not os.path.exists(workdir):
        print("create dir %s" % workdir)
        os.makedirs(workdir)
    return workdir


# if feature not found in data, we add 0
def uniform_features(dataframe, features):
    for i in features:
        if i not in dataframe.columns.to_list():
            dataframe.loc[:, i] = 0
    sort_dataframe = dataframe.loc[:, features]
    return sort_dataframe


def minmax_data(df):
    scaler = MinMaxScaler(feature_range=(-10, 10))
    print("do MinMaxScaler!")
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df


def roubst_data(df):
    scaler = RobustScaler(quantile_range=(25, 75))
    print("do RobustScaler!")
    scaled = scaler.fit_transform(df)
    df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df


# for feature selections!
def load_dataset(ssgseaf, norm):
    ssgsea_df = pd.read_csv(ssgseaf, index_col=0).T
    # try to normlizations
    # print("read raw data:\n")
    # print(ssgsea_df)
    if norm == 1:
        ssgsea_df = minmax_data(ssgsea_df)
    elif norm == 2:
        ssgsea_df = roubst_data(ssgsea_df)
    elif norm == 3:
        ssgsea_df = ssgsea_df*10
    elif norm == 0:
        ssgsea_df = ssgsea_df
    else:
        print("not select norm!")
    # print("out normlized data:\n")
    # print(ssgsea_df)
    return ssgsea_df


# for load raw data
def load_dataset_selected(ssgseaf, newf, norm):
    ssgsea_df = pd.read_csv(ssgseaf, index_col=0).T
    top2000_features = []
    if os.path.exists(newf):
        top2000_features = open(newf, "r").read().strip().split("\n")
    else:
        print("Please check the features !")
        exit(0)
    selected_ssgsea_df = uniform_features(ssgsea_df, top2000_features)
    if norm == 1:
        normed_ssgsea_df = minmax_data(selected_ssgsea_df)
    elif norm == 2:
        normed_ssgsea_df = roubst_data(selected_ssgsea_df)
    elif norm == 3:
        normed_ssgsea_df = selected_ssgsea_df*10
    else:
        print("not select norm!")
        normed_ssgsea_df = selected_ssgsea_df
    # print("out normlized data:\n")
    # print(normed_ssgsea_df)
    return normed_ssgsea_df


def df2np(X, y):
    x_train = X.values.astype(float)
    y_train = np_utils.to_categorical(y)
    return x_train, y_train


def split_dataset(dataset, label, train_data_path="", split=2):
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.1, stratify=label,
                                                        random_state=SEEDS, shuffle=True)
    if split == 3:
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=len(y_test),
                                                                    stratify=y_train, random_state=SEEDS, shuffle=True)
        samplename = x_train.index.to_list()
        samplename.extend(x_validate.index.to_list())
        samplename.extend(x_test.index.to_list())
        crps = list(y_train)
        crps.extend(list(y_validate))
        crps.extend(list(y_test))
        label = ["train"]*len(y_train)+['validate']*len(y_validate)+['test']*len(y_test)
        samples_splited = pd.DataFrame.from_dict({"samplename": samplename,
                                                  "CRPS": crps,
                                                  "label": label})
        samples_splitedf = os.path.join(train_data_path, "samples_splited_train.csv")
        samples_splited.to_csv(samples_splitedf)
        return x_train, x_validate, x_test, y_train, y_validate, y_test
    return x_train, x_test, y_train, y_test


# https://newbedev.com/how-to-perform-feature-selection-with-gridsearchcv-in-sklearn-in-python
def select_feature(X, label, newf, num_f=2000):
    print("RFECV with: \n")
    print(X)
    print(label)
    rf = RandomForestClassifier(random_state=SEEDS)
    rfecv = RFECV(estimator=rf, cv=StratifiedKFold(5, shuffle=True, random_state=SEEDS), scoring="accuracy",
                  step=200, min_features_to_select=num_f, n_jobs=6)
    # param_grid = [{"estimator__n_estimators": [10, 50, 80],
    #                "step": [200, 500, 1000]}]
    # gridsearch = GridSearchCV(rfecv, param_grid, cv=StratifiedKFold(2, shuffle=True, random_state=SEEDS))
    # print("GridSearch find the beat estimator")
    # gridsearch.fit(X_train, y_train)
    # print('best score:%f' % gridsearch.best_score_)
    # best_rfecv = gridsearch.best_estimator_
    rfecv.fit(X, label)
    print(rfecv.grid_scores_)
    feature_df = pd.DataFrame(columns=["features", "support", "ranking"])
    for f, s, r in zip(X.columns.to_list(), rfecv.support_, rfecv.ranking_):
        row = pd.Series({"features": f, "support": s, "ranking": r})
        feature_df = feature_df.append(row, ignore_index=True)
    feature_df = feature_df.sort_values(by='ranking')
    feature_df.to_csv(newf)


def select_feature_(X, label, newf, num_f=2000):
    print("ReliefF with: \n")
    print(X)
    X1, label1 = df2np(X, label)
    print(X1)
    print(label)
    fs = ReliefF(n_features_to_select=num_f, n_neighbors=20, n_jobs=6, verbose=True)
    fs.fit(X1, label)
    feature_df = pd.DataFrame(columns=["features", "score"])
    for feature_name, feature_score in zip(X.columns.to_list(), fs.feature_importances_):
        row = pd.Series({"features": feature_name, "score": feature_score})
        feature_df = feature_df.append(row, ignore_index=True)
    feature_df = feature_df.sort_values(by='score', ascending=False)
    feature_df.to_csv(newf)


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# 读取数据并扰动，然后取sampledata/batch的样本喂入模型
class DataGenerator(Sequence):

    def __init__(self, sampledata, labels, samples_w, dims, batch_size, n_class=6, n_channels=1, shuffle=False):
        self.sampledata = sampledata
        self.list_IDs = list(range(len(sampledata)))   # sampleIDs
        self.n_class = n_class
        self.labels = labels         # samples class
        self.samples_w = samples_w    # samples weight
        self.dim = dims               # features
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    # every epoch with such length samples
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y, sample_w = self.__data_generation(list_IDs_temp)
        return X, np_utils.to_categorical(y, num_classes=self.n_class), sample_w

        # Generates data containing batch_size samples
        # list_IDs_temp: batch samples select after shuffle
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)
        sample_w = np.empty(self.batch_size, dtype=float)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i, :, :] = self.sampledata[ID, :, :]
            y[i] = self.labels[ID]
            sample_w[i] = self.samples_w[ID]
        return X, y, sample_w

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class DataGenerator1(Sequence):

    def __init__(self, sampledata, labels, dims, batch_size, n_class=6, n_channels=1, shuffle=False):
        self.sampledata = sampledata
        self.list_IDs = list(range(len(sampledata)))   # sampleIDs
        self.n_class = n_class
        self.labels = labels         # samples class
        self.dim = dims               # features
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    # every epoch with such length samples
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, np_utils.to_categorical(y, num_classes=self.n_class)

        # Generates data containing batch_size samples
        # list_IDs_temp: batch samples select after shuffle
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, self.dim, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i, :, :] = self.sampledata[ID, :, :]
            y[i] = self.labels[ID]
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# 读取数据并扰动，然后取sampledata/batch的样本喂入模型
class DataGeneratorSub(Sequence):

    def __init__(self, sampledata, labels, dims, batch_size, n_class=6, n_channels=1, shuffle=False):
        self.sampledata = sampledata
        self.list_IDs = list(range(len(sampledata)))      # sampleIDs
        self.labels = labels          # samples class
        self.dim = dims               # features
        self.batch_size = batch_size
        self.n_class = n_class
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    # every epoch with such length samples
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

        # Generates data containing batch_size samples
        # list_IDs_temp: batch samples select after shuffle
    def __data_generation(self, list_IDs_temp):
        # X = np.empty((self.batch_size, self.dim, self.n_channels))
        # y = np.empty((self.batch_size, self.n_class), dtype=int)
        # Generate data
        X = self.sampledata[list_IDs_temp]
        y = self.labels[list_IDs_temp]
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def plot_his(history, folder, best_epoch=100):
    fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(15, 7))
    ax[0, 0].plot(history['acc'])
    ax[0, 0].plot(history['val_acc'])
    ax[0, 0].vlines(x=best_epoch, ymin=0, ymax=1, colors="grey", linestyles='--')
    ax[0, 0].set_title('model accuracy')
    ax[0, 0].set_ylabel('accuracy')
    ax[0, 0].set_ylim(0.0, 1.05)
    ax[0, 0].set_xlabel('epoch')
    ax[0, 0].legend(['train', 'validate'], loc='upper left')
    ax[0, 1].plot(history['loss'])
    ax[0, 1].plot(history['val_loss'])
    ax[0, 1].vlines(x=best_epoch, ymin=0, ymax=1, colors="grey", linestyles='--')
    ax[0, 1].set_title('model loss')
    ax[0, 1].set_ylabel('loss')
    ax[0, 1].set_ylim(0.0, 15.05)
    ax[0, 1].set_xlabel('epoch')
    ax[0, 1].legend(labels=['train', 'validate'], loc='upper left')
    ax[0, 2].plot(history.loc[35:, 'loss'])
    ax[0, 2].plot(history.loc[35:, 'val_loss'])
    ax[0, 2].vlines(x=best_epoch, ymin=0, ymax=1, colors="grey", linestyles='--')
    ax[0, 2].set_title('model loss')
    ax[0, 2].set_ylabel('loss')
    ax[0, 2].set_ylim(0.0, 2.05)
    ax[0, 2].set_xlabel('epoch from 35')
    ax[0, 2].legend(labels=['train', 'validate'], loc='upper left')
    ax[1, 0].plot(history['recall_m'])
    ax[1, 0].plot(history['val_recall_m'])
    ax[1, 0].vlines(x=best_epoch, ymin=0, ymax=1, colors="grey", linestyles='--')
    ax[1, 0].set_title('model recall')
    ax[1, 0].set_ylabel('recall')
    ax[1, 0].set_ylim(0.0, 1.05)
    ax[1, 0].set_xlabel('epoch')
    ax[1, 0].legend(['train', 'validate'], loc='upper left')
    ax[1, 1].plot(history['precision_m'])
    ax[1, 1].plot(history['val_precision_m'])
    ax[1, 1].vlines(x=best_epoch, ymin=0, ymax=1, colors="grey", linestyles='--')
    ax[1, 1].set_title('model precision')
    ax[1, 1].set_ylabel('precision')
    ax[1, 1].set_ylim(0.0, 1.05)
    ax[1, 1].set_xlabel('epoch')
    ax[1, 1].legend(['train', 'validate'], loc='upper left')
    plt.savefig(os.path.join(folder, "metric.png"))
    plt.close()


class ClassReport:
    def __init__(self, label, pred_prob, pred_label):
        self.pred_prob = pred_prob    # 预测概率np.array
        self.pred_label = pred_label   # 预测label
        self.label = label        # 实际label

    def get_confusion(self, file_name):
        import seaborn as sns
        label_list = [REVER_CLUSTER_NAME[i] for i in np.argmax(self.label, axis=1)]
        pred_label_list = [REVER_CLUSTER_NAME[i] for i in np.argmax(self.pred_label, axis=1)]
        conf = pd.DataFrame(confusion_matrix(label_list, pred_label_list), columns=TOTAL_CLUSTER, index=TOTAL_CLUSTER)
        conf_ratio = conf / conf.sum(axis=1)
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))
        p1 = sns.heatmap(conf, annot=True, cmap="PiYG", fmt="d", ax=ax1)
        p2 = sns.heatmap(conf_ratio, annot=True, cmap="PiYG", fmt=".3g", ax=ax2)
        ax1.set_title('Confusion Matrix')
        ax1.set_ylabel('True label')
        ax1.set_xlabel('Predicted label')
        p1.set_yticklabels(p1.get_yticklabels(), rotation=0)
        ax2.set_title('Confusion Matrix')
        ax2.set_ylabel('True label')
        ax2.set_xlabel('Predicted label')
        p2.set_yticklabels(p2.get_yticklabels(), rotation=0)
        plt.savefig(file_name.replace(".xlsx", ".png"))
        plt.close()
        conf.to_excel(file_name)

    def roc_cruve(self, file_name):
        from itertools import cycle
        from sklearn.metrics import auc
        lw = 2
        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        roc_auc = dict()
        average_precision = dict()
        for i in range(len(self.label[0])):
            fpr[i], tpr[i], _ = roc_curve(self.label[:, i], self.pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(self.label[:, i], self.pred_prob[:, i])
            average_precision[i] = average_precision_score(self.label[:, i], self.pred_prob[:, i])
        fpr["micro"], tpr["micro"], _ = roc_curve(self.label.ravel(), self.pred_prob.ravel())
        precision["micro"], recall["micro"], _ = precision_recall_curve(self.label.ravel(), self.pred_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        average_precision["micro"] = average_precision_score(self.label.ravel(), self.pred_prob.ravel())
        # plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))
        colors = cycle(['darkblue', 'red', 'lime', 'deepskyblue', 'darkorchid', "gray"])
        for i, color in zip(range(len(self.label[0])), colors):
            ax1.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC of {0} (area = {1:0.2f})'.format(TOTAL_CLUSTER[i], roc_auc[i]))
            ax2.plot(precision[i], recall[i], color=color, lw=lw,
                     label='PRC of {0} (area = {1:0.2f})'.format(TOTAL_CLUSTER[i], average_precision[i]))
        ax1.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC (area = {0:0.2f})'.format(roc_auc["micro"]))
        ax2.plot(precision["micro"], recall["micro"],
                 label='micro-average PRC (area = {0:0.2f})'.format(average_precision["micro"]))
        ax1.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('AUROC curve for CRPS')
        ax1.legend(loc="lower right")
        ax2.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('AUPRC curve for CRPS')
        ax2.legend(loc="lower right")
        plt.savefig(file_name)
        plt.close()

    def report(self, file_name):
        import seaborn as sns
        res = pd.DataFrame(
            classification_report(self.label, self.pred_label, target_names=TOTAL_CLUSTER, output_dict=True))
        plt.figure(figsize=(8.5, 4))
        sns.heatmap(res.iloc[:-1, :].T, annot=True, cmap="PiYG")
        plt.yticks(rotation=0)
        plt.title('Classification Report')
        plt.xlabel('Metric Value')
        plt.savefig(file_name.replace(".xlsx", ".png"))
        plt.close()
        res.T.to_excel(file_name)


# 对于模型训练过程，产生多个类别的综合评判值
def recall_m(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_pos = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_pos / (all_pos + K.epsilon())
    return recall


# 预测中大于0.5认为是positive 预测
def precision_m(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_pos / (pred_pos + K.epsilon())
    return precision


def f1score_m(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f2score_m(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 5 * ((precision * recall) / (4*precision + recall + K.epsilon()))


# https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomCallback, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_loss = np.Inf
        self.best_loss = np.Inf
        self.best_val_acc = 0.0
        self.best_acc = 0.0
        self.best_f2score_m = 0.0
        self.best_val_f2score_m = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        acc = logs.get('acc')
        val_loss = logs.get('val_loss')
        loss = logs.get('loss')
        val_f2score_m = logs.get('val_f2score_m')
        f2score_m = logs.get('f2score_m')
        # If BOTH the validation loss AND acc does not improve for 'patience' epochs, stop training early.
        # acc up val_loss down or acc up val_acc up
        if ((np.less(val_loss, self.best_val_loss) and np.less(loss, self.best_loss)) and
           (np.greater(acc, self.best_acc) and np.greater(val_acc, self.best_val_acc))) or \
           (np.greater(f2score_m, self.best_f2score_m) and (np.greater(val_f2score_m, self.best_val_f2score_m))):
            self.best_val_loss = val_loss
            self.best_loss = loss
            self.best_val_acc = val_acc
            self.best_acc = acc
            self.best_f2score_m = f2score_m
            self.best_val_f2score_m = val_f2score_m
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
