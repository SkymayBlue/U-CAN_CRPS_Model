SEEDS = 46946
import argparse
import os
import random as rn
rn.seed(SEEDS)
import numpy as np
np.random.seed(SEEDS)
import pandas as pd
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(SEEDS)
tf.compat.v1.set_random_seed(SEEDS)

from src.crc_classiser_basic_imblance_nadam import resnet50
from src.crc_classiser_utils import load_dataset_selected, makedir, REVER_CLUSTER_NAME


# load weights from trained model
def load_model_weights(model, modelf):
    model = model.load_weights(modelf)
    return model


class Predictions():

    def __init__(self, predictf):
        self.predictf = predictf
        self.summary = np.array([], dtype=np.float64)
        self.resf = ""
        # run the predictions
        self.predictions()

    def predictions(self, workdir="./", modeld="./model/model_res50_nadam_f1score_imblearn",
                    features="./features/ReliefF_select_feature_selected.csv", norm=1):
        # read new data file for predictions
        predictf = self.predictf
        dataframe = load_dataset_selected(predictf, features, norm)
        sampleanmes = dataframe.index.to_list()
        nfeatures = dataframe.shape[1]
        n_class = 6
        params = {'channel': 2000, 'avg_pool': 1, 'avg_pool_s': 1, 'method': "resnet50", 'lr': 0.001,
                  'optimozer': "nadam"}
        input_channel = params['channel']
        model = resnet50(n_class, nfeatures, params)
        model_dir = os.path.join(workdir, modeld)
        print("model loaded from %s" % os.path.join(model_dir, 'model_weights_resnet_best.h5'))
        model.load_weights(os.path.join(model_dir, 'model_weights_resnet_best.h5'))
        # reshape the features
        first_feature = dataframe.values.astype(float).reshape(-1, int(nfeatures / input_channel), input_channel)
        predictions = model.predict(first_feature)  # prediction probality
        i = 0
        res_prob = pd.DataFrame(predictions, index=sampleanmes, columns=REVER_CLUSTER_NAME.values())  # np.array
        samples_label = {}
        for sample in predictions:
            sample = list(sample)
            if max(sample) >= 0.5:  # 最终的结果超过0.5，认为可以判定类型
                sample_label = REVER_CLUSTER_NAME[sample.index(max(sample))]
                samples_label[sampleanmes[i]] = sample_label
            else:
                samples_label[sampleanmes[i]] = np.nan
            i += 1
        res = pd.DataFrame.from_dict(samples_label, orient="index", columns=["prediction"])
        count_res = res['prediction'].value_counts()
        count_res.sort_index(inplace=True)
        resdf = pd.concat([res_prob, res], axis=1)
        predict_dir = makedir(os.path.join(workdir, "predict"))
        database_name = os.path.basename(predictf).split("_")[0]
        predict_resf = os.path.join(predict_dir, "predictions_only.xlsx")
        # if os.path.exists(predict_resf):
        #     writer = pd.ExcelWriter(predict_resf, engine="openpyxl", mode='a')
        # else:
        writer = pd.ExcelWriter(predict_resf, mode='w')
        resdf.to_excel(writer, sheet_name=database_name + "_predict")
        count_res.to_excel(writer, sheet_name=database_name + "_Counter")
        writer.close()
        self.summary = count_res
        self.resf = predict_resf

    def get_predictions_summary(self):
        return self.summary

    def predicition_results_file(self):
        return self.resf


def main():
    p = argparse.ArgumentParser(description="About model train and explain! the result")
    p.add_argument("-w", dest="workdir", default=".", help="workdir for save all result and log file")
    p.add_argument("-m", dest="model", default="./model/model_res50_nadam_f1score_imblearn", help="short model name!")
    p.add_argument("-p", dest="predict", default="./predict/GSE35896_logtpm_mxT.csv", type=str,
                   help="the new samples for predict!")
    p.add_argument("-n", dest="norm", type=int, default=1,
                   help="do normlization or not，1:MinMaxScaler the data, 2:RobustScaler the data,"
                        "3:all data multiply 10, 0: don't scale!")
    p.add_argument("-f", dest="features", type=str, default="./features/ReliefF_select_feature_selected.csv",
                   help="prepare data should with new features or not. default is the select 2000 features")
    args = p.parse_args()
    # workdir = os.path.join(args.workdir, "norm")
    res = Predictions(args.predict)
    print(res.get_predictions_summary())


# if __name__ == '__main__':
#     main()