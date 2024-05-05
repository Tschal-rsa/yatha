import pickle, os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix

# /home/zhangrui/Desktop/EEGProject


def load_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def _create_folder_if_not_exist(filename):
    """ Makes a folder if the path does not already exist """
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def save_pickle(obj, filename, protocol=4):
    """ Basic pickle/dill dumping """
    _create_folder_if_not_exist(filename)
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=protocol)

def load_ds(dataset):
    X = []
    for root, ds, fs in os.walk(dataset):
        for f in fs:
            fullname = os.path.join(root, f)
            raw_data = pd.read_hdf(fullname, key='data')
            for x in raw_data.values:
                X.append(x)
    return X


def load_dataset(filename,num_threshold=None):
    # X_no = load_ds(filename + "normal_green")
    # X_mi = load_ds(filename + "mild_blue")
    X_mo = load_ds(filename + "moderate_orange")
    X_se = load_ds(filename + "severe_red")
    if num_threshold:
        # X_no = X_no[:num_threshold]
        # X_mi = X_mi[:num_threshold//1]
        X_mo = X_mo[:num_threshold]
        X_se = X_se[:num_threshold]
    # X = X_no+X_mi+X_mo+X_se
    X = X_mo+X_se
    # X = X_no+X_mi
    # print("len(X_no)",len(X_no))
    # print("len(X_mi)",len(X_mi))
    print("len(X_mo)",len(X_mo))
    print("len(X_se)",len(X_se))
    # Y_no = [0] * len(X_no)
    # Y_mi = [1] * len(X_mi)
    # Y_mi = [0] * len(X_mi)
    Y_mo = [0] * len(X_mo)
    Y_se = [1] * len(X_se)

    # Y = Y_no+Y_mi+Y_mo+Y_se
    Y = Y_mo+Y_se
    # Y = Y_no+Y_mi

    return X, Y


if __name__ == '__main__':
    metric = "cwt"
    num_threshold = 2527
    # num_threshold = None
    X_train, Y_train = load_dataset("/data/lzy/脑电/Processed/feature/"+metric+"/train/",num_threshold)
    X_val, Y_val = load_dataset("/data/lzy/脑电/Processed/feature/"+metric+"/val/",num_threshold=300)

    clf = RandomForestClassifier(
    # clf = AdaBoostClassifier(
    # clf = XGBClassifier(
        bootstrap=True,
        # max_depth=100,
        max_features="auto",
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=2000,
        criterion="entropy",
        # criterion="gini",
        n_jobs=4
    )
    print(np.array(X_train).shape)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_val)
    Y_probn = clf.predict_proba(X_val)

    print(np.array(Y_pred).shape,np.array(Y_probn).shape)
    # save_pickle(fpr, './Result/fpr_ad_all_foo.pkl')
    # save_pickle(tpr, './Result/tpr_ad_all_foo.pkl')
    predictions_validation = Y_probn[:, 1]
    fpr, tpr, _ = roc_curve(Y_val, predictions_validation)
    roc_auc = auc(fpr, tpr)
    print('Acc: ', accuracy_score(Y_val, Y_pred))
    print('Precision:', precision_score(Y_val, Y_pred,average="macro"))
    print('F1:', f1_score(Y_val, Y_pred, average='weighted'))
    print('Auc：', roc_auc)

    # res = roc_auc_score(Y_val,Y_probn,multi_class = "ovr")
    # print("AUC:",res)
    # print('Acc: ', accuracy_score(Y_val, Y_pred))
    # print('Precision:', precision_score(Y_val, Y_pred,average="macro"))
    # print('F1:', f1_score(Y_val, Y_pred, average='weighted'))

    cm = confusion_matrix(Y_val, Y_pred)
    print(cm)

    fname = "/data/lzy/脑电/Processed/feature/"+metric+"/train/mild_blue/LT-0363_0.h5"
    raw = pd.read_hdf(fname, key='data')
    feature_label = list(raw.columns.values)

    feature_scores = pd.Series(clf.feature_importances_, index=feature_label).sort_values(ascending=False)
    print(feature_scores[0:10])
    print("sum of scores:",feature_scores.values.sum())

    feature_importances = clf.feature_importances_
    df = pd.DataFrame(feature_importances).T
    df.columns = feature_label # type: ignore
    df.to_csv("ada_feature_importances_AD_Normal.csv")

    # 决策树可视化
    print("tree nums: ", len(clf.estimators_))
    # estimator = clf.estimators_[0]
    # feature_names = feature_label
    # y_train_str = np.array(Y_train).astype("str")
    # y_train_str[y_train_str == '0'] = 'Normal'
    # y_train_str[y_train_str == '1'] = 'AD'

    # import os
    # from sklearn.tree import export_graphviz

    # export_graphviz(estimator, out_file='tree.dot', feature_names=feature_names,
    #                 class_names=y_train_str,
    #                 rounded=True, proportion=True,
    #                 label='root', precision=2, filled=True
    #                 )
    # from subprocess import call
    # (graph,) = pydot.graph_from_dot_file('tree.dot')
    # graph.write_png('somefile.png')
    # # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    # # from IPython.display import Image
    # # Image(filename='tree.png')

    # feature_importances_ = clf.feature_importances_
    # fname = "/data/lzy/脑电/Processed/feature/cwt/train/mild_blue/LT-0361_0.h5"
    # raw = pd.read_hdf(fname, key='data')
    # feature_label = list(raw.columns.values)
    # df = pd.DataFrame(feature_importances).T
    # df.columns = feature_label
    # df.to_csv("ada_feature_importances_case_AD_Normal.csv")
