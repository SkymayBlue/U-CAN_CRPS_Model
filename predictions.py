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
from src.crc_classiser_utils import load_dataset_selected, makedir, REVER_CLUSTER_NAME, exp2pathByRpy2


# load weights from trained model
def load_model_weights(model, modelf):
    model = model.load_weights(modelf)
    return model


class Predictions():

    def __init__(self, predictf, dsource):
        self.predictf = predictf
        self.dsource = dsource
        if dsource == "es":
            self.to_predict_df = self._get_inputdf()
            self.data_check_log = "input the enrichscore matrix, will check the features"
        elif dsource == "expression":
            self.to_predict_df = exp2pathByRpy2(self._get_inputdf())
            self.data_check_log = self._check_log()
        else:
            self.data_check_log = self._return()
            # self.summary = np.array([], dtype=np.float64)
        self.summary = ""
        self.resf = ""
        self.resdf = pd.DataFrame()
        # run the predictions
        self.predictions()

    def _return(self):
        return "please input espression matrix or enrich score matrix!"

    def _get_inputdf(self):
        input_df = pd.read_csv(self.predictf, index_col=0)
        return input_df

    def _check_log(self):
        max_value = self.to_predict_df.select_dtypes(include=[np.number]).max().max()
        if max_value > 50:
            self.to_predict_df = self.to_predict_df.map(lambda x: np.log2(x))
            return "input expression will be log2!"
        elif max_value < 0:
            return "please check your input matrix with right value!"
        else:
            return "value check is OK!"

    def predictions(self, workdir="./", modeld="./model/model_res50_nadam_f1score_imblearn",
                    features="./features/ReliefF_select_feature_selected.csv", norm=1):
        # read new data file for predictions
        predict_df = self.to_predict_df
        dataframe = load_dataset_selected(predict_df, features, norm)
        if type(dataframe) == str:
            self.resf = None
            self.data_check_log = dataframe
        elif isinstance(dataframe,pd.DataFrame):
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
                if max(sample) >= 0.5:  # proba >0.5
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
            database_name = os.path.basename(self.predictf).split("_")[0]
            predict_resf = os.path.join(predict_dir, "predictions_only.xlsx")
            # if os.path.exists(predict_resf):
            #     writer = pd.ExcelWriter(predict_resf, engine="openpyxl", mode='a')
            # else:
            writer = pd.ExcelWriter(predict_resf, mode='w')
            resdf.to_excel(writer, sheet_name=database_name + "_predict")
            # count_res.to_excel(writer, sheet_name=database_name + "_Counter")
            writer.close()
            self.resdf = resdf
            self.summary = count_res
            self.resf = predict_resf
            self.data_check_log = self.data_check_log + "; Prediction is Done"
        else:
            self.resf = ""
            self.data_check_log = 'some error happend, please concant to the author with an example of your input!'
    
    def get_predict(self):
        return self.resdf
    
    def get_predictions_summary(self):
        return self.summary

    def predicition_results_file(self):
        return self.resf, self.data_check_log


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