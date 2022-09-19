import numpy as np 
import pandas as pd
import itertools

useful = ['v22', 'v31', 'v50', 'v66', 'v110', 'v56', 'v113', 'v79', 'v129', 'v47', 'v3', 'v38', 'v74', 'v24', 'v75', 'v26', 'v46', 'v51', 'v72', 'v64', 'v32', 'v71', 'v100', 'v109', 'v89']+['v22', 'v31', 'v50', 'v66', 'v113', 'v56', 'v79', 'v47', 'v110', 'v74', 'v3', 'v38', 'v24', 'v129', 'v30', 'v40', 'v118', 'v49', 'v100', 'v35', 'v92', 'v10', 'v84', 'v116', 'v117']+['v22', 'v31', 'v50', 'v66', 'v56', 'v129', 'v110', 'v113', 'v79', 'v47', 'v74', 'v3', 'v38', 'v24', 'v40', 'v72', 'v10', 'v100', 'v124', 'v73', 'v67', 'v36', 'v49', 'v35', 'v44']+['v22', 'v31', 'v50', 'v66', 'v47', 'v56', 'v110', 'v113', 'v129', 'v79', 'v3', 'v38', 'v24', 'v74', 'v75', 'v40', 'v72', 'v87', 'v48', 'v105', 'v111', 'v10', 'v118', 'v32', 'v20']+['v22', 'v31', 'v50', 'v66', 'v110', 'v79', 'v56', 'v113', 'v47', 'v75', 'v74', 'v129', 'v24', 'v8', 'v3', 'v30', 'v72', 'v40', 'v83', 'v94', 'v18', 'v92', 'v73', 'v76', 'v58']
useful1 = ['v22', 'v31', 'v50', 'v66', 'v110', 'v56', 'v113', 'v79', 'v129', 'v47', 'v3', 'v38', 'v74', 'v24', 'v75', 'v26', 'v46', 'v51', 'v72', 'v64', 'v32', 'v71', 'v100', 'v109', 'v89']
useful2 = ['v22', 'v31', 'v50', 'v66', 'v113', 'v56', 'v79', 'v47', 'v110', 'v74', 'v3', 'v38', 'v24', 'v129', 'v30', 'v40', 'v118', 'v49', 'v100', 'v35', 'v92', 'v10', 'v84', 'v116', 'v117']
useful3 = ['v22', 'v31', 'v50', 'v66', 'v56', 'v129', 'v110', 'v113', 'v79', 'v47', 'v74', 'v3', 'v38', 'v24', 'v40', 'v72', 'v10', 'v100', 'v124', 'v73', 'v67', 'v36', 'v49', 'v35', 'v44']
useful4 = ['v22', 'v31', 'v50', 'v66', 'v47', 'v56', 'v110', 'v113', 'v129', 'v79', 'v3', 'v38', 'v24', 'v74', 'v75', 'v40', 'v72', 'v87', 'v48', 'v105', 'v111', 'v10', 'v118', 'v32', 'v20']
useful5 = ['v22', 'v31', 'v50', 'v66', 'v110', 'v79', 'v56', 'v113', 'v47', 'v75', 'v74', 'v129', 'v24', 'v8', 'v3', 'v30', 'v72', 'v40', 'v83', 'v94', 'v18', 'v92', 'v73', 'v76', 'v58']


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    percent = 100 * (start_mem - end_mem) / start_mem
    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))
    return df


def feature_engineering_1way(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 2-combinations of values
    # in this list
    # for example:
    # list(itertools.combinations([1,2,3], 2)) will return
    # [(1, 2), (1, 3), (2, 3)]
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:,c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df 


def feature_engineering_2way(df, cat_cols):
    """
    This function is used for feature engineering
    :param df: the pandas dataframe with train/test data
    :param cat_cols: list of categorical columns
    :return: dataframe with new features
    """
    # this will create all 3-combinations of values
    # in this list
    # for example:
    # list(itertools.combinations([1,2,3], 3))
    combi = list(itertools.combinations(cat_cols, 3))
    for c1, c2, c3 in combi:
        df.loc[:,c1 + "_" + c2 + "_" + c3] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
    return df


def prepare_data(data, usefuls):

    features = [col for col in data.columns if col not in ("ID", "target", "kfold")]

    todrop = list(set(features).difference(usefuls))
    data.drop(columns=todrop, inplace=True)

    features = [col for col in data.columns if col not in ("ID", "target", "kfold")]

    dis_col = [col for col in features if data[col].dtype in ["int32", "int8"]]
    cat_col = [col for col in features if data[col].dtype == "O"]
    cont_col = [col for col in features if data[col].dtype not in dis_col + cat_col]

    # Imputing categorical columns with NONE
    for col in cat_col :
        data['NA_' + col] = data[col].isna().astype(np.int8)
        data[col].fillna('NONE', inplace=True)

    # Imputing numerical columns with 999
    for col in cont_col :
        data['NA_' + col] = data[col].isna().astype(np.int8)
        data[col].fillna(999, inplace=True)
    
    return data
