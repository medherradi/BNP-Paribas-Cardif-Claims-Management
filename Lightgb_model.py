import pandas as pd 
import numpy as np 
from utils import useful4, useful
from utils import reduce_mem_usage
from utils import feature_engineering_1way, feature_engineering_2way, prepare_data
from category_encoders.target_encoder import TargetEncoder
from category_encoders.count import CountEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss



def lightgbmclassifier(fold, **kwargs):

    # all columns are features except id, target and kfold columns
    features = [f for f in kwargs['data'].columns if f not in ("ID", "target", "kfold")]

    # now it’s time to count encode the features
    
    # initialize CountEncoder for each feature column
    countencoder = CountEncoder(cols=cat_col)
   
    # fit count encoder on all kwargs['data'] and transform it
    kwargs['data'][cat_col] = countencoder.fit_transform(kwargs['data'][cat_col]) 
    kwargs['test'][cat_col] = countencoder.transform(kwargs['test'][cat_col])  
    
    # get training kwargs['data'] using folds
    df_train = kwargs['data'][kwargs['data'].kfold != fold].reset_index(drop=True)

    # get validation kwargs['data'] using folds
    df_valid = kwargs['data'][kwargs['data'].kfold == fold].reset_index(drop=True)

    # get training kwargs['data']
    x_train = df_train[features]
    y_train = df_train["target"]

    # get validation kwargs['data']
    x_valid = df_valid[features]
    y_valid = df_valid["target"]

    # initialize TargetEncoder for each feature column
    targetencoder = TargetEncoder(cols=dis_col, min_samples_leaf=1, smoothing=1)

    # fit target encoder on train kwargs['data'] and transform it
    x_train.loc[:,dis_col] = targetencoder.fit_transform(x_train[dis_col], y_train)

    # fit target encoder on validation kwargs['data'] and transform it
    x_valid.loc[:,dis_col] = targetencoder.transform(x_valid[dis_col])
    kwargs['test'].loc[:,dis_col] = targetencoder.transform(kwargs['test'][dis_col])

    # initialize xgboost model
    model = LGBMClassifier(boosting_type="gbdt", 
                           num_leaves=31, 
                           max_depth=10, 
                           learning_rate=0.04, 
                           n_estimators=1000, 
                           objective="binary",
                           min_split_gain=0.0, 
                           min_child_weight=0.001, 
                           min_child_samples=20, 
                           subsample=0.8, 
                           subsample_freq=0, 
                           colsample_bytree=1.0)

    # fit model on training kwargs['data'] (ohe)
    model.fit(x_train, y_train.values)

    print("================================================================")
    print("================================================================")

    print(len(x_valid.columns), x_valid.columns)
    print(len(kwargs['test'].columns), kwargs['test'].columns)

    print("================================================================")
    print("================================================================")

    if "ID" in kwargs['test'].columns:
        kwargs['test'].drop(columns=['ID'], inplace=True)

    # predict on validation kwargs['data']
    # we need the probability values 
    # we will use the probability of 1s
    preds = model.predict_proba(x_valid)[:,1]
    test_preds = model.predict_proba(kwargs['test'])[:,1]
    loss = log_loss(y_valid, preds)

    # print loss
    print(f"Fold = {fold}, LOSS = {loss}")
    
    df_valid.loc[:,"lgbmpred"] = preds

    return (test_preds, df_valid[["ID", "kfold", "lgbmpred", "target"]])


if __name__ == "__main__":
    
    data_dict = {"lightgbm_useful4": useful4, "lighgbm_useful": useful}

    for k, v in data_dict.items():

        # read the train csv file
        train = pd.read_csv("train_folds.csv")
        test = pd.read_csv("test.csv")

        # reduce memory
        train = reduce_mem_usage(df=train)
        test = reduce_mem_usage(df=test)

        # prepare kwargs['data']
        train = prepare_data(data=train, usefuls=v)
        test = prepare_data(data=test, usefuls=v)
        

        # prepare categorical columns for feature engineering
        # I used just 1 way feature engineering because of computational power (interaction between categorical columns)
        cat_col = [col for col in train.columns if train[col].dtype == "O"]
        train = feature_engineering_1way(train, cat_cols=cat_col)
        test = feature_engineering_1way(test, cat_cols=cat_col)
        #train = feature_engineering_2way(train, cat_cols=cat_col)
        #test = feature_engineering_2way(test, cat_cols=cat_col)

        # prepare all features for training process
        dis_col = [col for col in train.columns if train[col].dtype in ["int32", "int8"] and col not in ["ID", "target", "kfold"] and train[col].nunique()>2]
        cat_col = [col for col in train.columns if train[col].dtype == "O"]
        cont_col = [col for col in train.columns if col not in dis_col + cat_col + ["ID", "target", "kfold"] and train[col].nunique()>2]

        #print(train["kfold"].value_counts())
        # training lightgbmclassifier model for 10 fold cross validation 
        lgbmpred_list = []
        lgbmpred_test = []
        for fold in range(10):
            temp_test_df, temp_df = lightgbmclassifier(fold=fold, data=train, test=test)
            lgbmpred_list.append(temp_df)
            lgbmpred_test.append(temp_test_df)
        
        df_valid = pd.concat(lgbmpred_list)
        pred_test = np.mean(np.column_stack(lgbmpred_test), axis=1)
        pred_test = pd.DataFrame(pred_test, columns=[f'lgbmpred_test_{k}'])
        print(df_valid.shape)
        df_valid.to_csv(f"predictions/{k}_pred.csv", index=False)
        pred_test.to_csv(f"predictions/{k}_test_pred.csv", index=True)
