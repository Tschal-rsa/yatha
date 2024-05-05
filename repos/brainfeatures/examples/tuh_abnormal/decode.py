import logging
import pickle
import time
import os

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import pandas as pd
import numpy as np

from repos.brainfeatures.utils.sun_grid_engine_util import parse_run_args
from repos.brainfeatures.data_set.tuh_abnormal import TuhAbnormal, add_meta_feature
from repos.brainfeatures.data_set.edf_abnormal import EDFAbnormal, add_meta_feature

from repos.brainfeatures.experiment.experiment import Experiment
from repos.brainfeatures.utils.file_util import json_store


def root_mean_squared_error(y_true, y_pred, sample_weight=None,
                            multioutput='uniform_average'):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight,
                                      multioutput))


def run_exp(train_dir, eval_dir, model, n_folds_or_repetitions,
            n_jobs, n_recordings, task, C, gamma, bootstrap,
            min_samples_split, min_samples_leaf, criterion, max_depth,
            max_features, n_estimators, result_dir,
            feature_vector_modifier=None):

    train_set_feats = EDFAbnormal(train_dir, target="pathological", extension=".h5",
                                  n_recordings=n_recordings, subset="train")
    train_set_feats.load()

    eval_set_feats = None
    if eval_dir is not None:
        eval_set_feats = EDFAbnormal(eval_dir, target="pathological", subset="eval",
                                     extension=".h5")
        eval_set_feats.load()

    if model == "rf":
        if task == "age":
            clf = RandomForestRegressor(
                bootstrap=bootstrap,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                n_estimators=n_estimators,
                criterion=criterion,
                n_jobs=n_jobs
            )
        else:
            clf = RandomForestClassifier(
                bootstrap=bootstrap,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                n_estimators=n_estimators,
                criterion=criterion,
                n_jobs=n_jobs
            )
    else:
        assert model == "svm", "unknown model"
        if task == "age":
            clf = SVR(
                C=C,
                gamma=gamma
            )
        else:
            clf = SVC(
                C=C,
                gamma=gamma
            )

    if task == "age":
        metrics = [root_mean_squared_error]
    else:
        metrics = [accuracy_score, roc_auc_score]

    exp = Experiment(
        devel_set=train_set_feats,
        estimator=clf,
        preproc_function=None,
        n_splits_or_repetitions=n_folds_or_repetitions,
        feature_generation_params=None,
        feature_generation_function=None,
        n_jobs=n_jobs,
        metrics=metrics,
        eval_set=eval_set_feats,
        feature_vector_modifier=feature_vector_modifier,
    )
    return exp


def write_kwargs(kwargs):
    result_dir = kwargs["result_dir"]
    # save predictions and rf feature importances for further analysis
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    json_store(kwargs, result_dir + "config.json")


def save_predictions_and_info():
    result_dir = kwargs["result_dir"]
    if kwargs["eval_dir"] is None:
        set_name = "devel"
    else:
        set_name = "eval"

    exp.predictions["train"].to_csv(result_dir + "predictions_train.csv")
    exp.predictions[set_name].to_csv(
        result_dir + "predictions_" + set_name + ".csv")
    if kwargs["model"] == "rf":
        for key, value in enumerate(exp.info[set_name]):
            value.to_csv(key + '_' + set_name + '.csv')


def create_df_from_feature_importances(id_, importances):
    """ create a pandas data frame from predictions and labels to an id"""
    df = pd.DataFrame()
    repeated_id = np.repeat(id_, len(importances))
    for i in range(len(repeated_id)):
        row = {"id": repeated_id[i],
               "importances": importances[i]}
        df = df.append(row, ignore_index=True)
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    kwargs = parse_run_args()
    start_time = time.time()
    exp = run_exp(**kwargs)
    exp.run()
    end_time = time.time()
    run_time = end_time - start_time
    logging.info("Experiment runtime: {:.2f} sec".format(run_time))

    pickle.dump(exp, open(kwargs["result_dir"] + "exp.pkl", "wb"))

    write_kwargs(kwargs)
    save_predictions_and_info()
