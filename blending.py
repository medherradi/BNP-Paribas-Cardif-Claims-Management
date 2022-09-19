from pathlib import Path
import pandas as pd 
import numpy as np


if __name__ ==  "__main__":
    
    print("=================== preparing data ==========================")
    #df1 = pd.read_csv("predictions/catboost_useful5_pred.csv")
    #df2 = pd.read_csv("predictions/lightgbm_useful4_pred.csv")
    #df3 = pd.read_csv("predictions/xgboost_useful5_pred.csv")

    # Catboost predictions for (useful and useful5)
    catboost_test = pd.read_csv('predictions/catboost_useful_test_pred.csv')
    catboost_test5 = pd.read_csv('predictions/catboost_useful5_test_pred.csv')  

    # Lightgbm predictions for (useful and useful4)
    lightbgm_test = pd.read_csv('predictions/lighgbm_useful_test_pred.csv')
    lightbgm_test4 = pd.read_csv('predictions/lightgbm_useful4_test_pred.csv')

    # XGboost predictions for (useful and useful4)
    xgboost_test = pd.read_csv('predictions/xgboost_useful_test_pred.csv')
    xgboost_test4 = pd.read_csv('predictions/xgboost_useful4_test_pred.csv')

    #print('=================== removing columns ==========================')
    #lightbgm_test.drop(columns='Unnamed: 0',inplace=True)
    #lightbgm_test4.drop(columns='Unnamed: 0',inplace=True)
    #xgboost_test.drop(columns='Unnamed: 0',inplace=True)
    #xgboost_test4.drop(columns='Unnamed: 0',inplace=True)
    
    
    print('=================== renaming columns ==========================')
    catboost_test.rename(columns={"ctbpred_test_catboost_useful": "ctbpred_test"}, inplace=True)
    catboost_test5.rename(columns={"ctbpred_test_catboost_useful5": "ctbpred_test5"}, inplace=True)
    lightbgm_test.rename(columns={"lgbmpred_test_lighgbm_useful": "lgbpred_test"}, inplace=True)
    lightbgm_test4.rename(columns={"lgbmpred_test_lightgbm_useful4": "lgbpred_test4"}, inplace=True)
    xgboost_test.rename(columns={"xgbpred_test_xgboost_useful": "xgbpred_test"}, inplace=True)
    xgboost_test4.rename(columns={"xgbpred_test_xgboost_useful4": "xgbpred_test4"}, inplace=True)


    print('=================== saving data to csv file ==========================')
    catboost_test.to_csv(f"predictions/catboost_useful_test_pred.csv", index=False)
    catboost_test5.to_csv(f"predictions/catboost_useful5_test_pred.csv", index=False)
    lightbgm_test.to_csv(f"predictions/lighgbm_useful_test_pred.csv", index=False)
    lightbgm_test4.to_csv(f"predictions/lightgbm_useful4_test_pred.csv", index=False)
    xgboost_test.to_csv(f"predictions/xgboost_useful_test_pred.csv", index=False)
    xgboost_test4.to_csv(f"predictions/xgboost_useful4_test_pred.csv", index=False)


    #==================================================================

    p = Path()
    files = p.glob("predictions/*_test_pred.csv")
    print("=====================================================================")
    print("=====================================================================")
    filename_list = [file.name for file in files]
    #print(filename_list)
    print("=====================================================================")
    print("=====================================================================")
    df = pd.concat([pd.read_csv(f'predictions/{df}') for df in filename_list], axis=1)
    #print(df.shape)
    #print(df.columns)
    df.to_csv('final_pred/models_pred.csv', index=False)
            
    
    print("=====================================================================")
    

    print("========================rank average================================")

    avg_pred = np.mean(df.values, axis=1)
    avg_pred = pd.DataFrame(avg_pred, columns=['avg_test_pred'])
    print(avg_pred)
    print(avg_pred.shape)
    avg_pred.to_csv('final_pred/avg_test_pred.csv', index=False)

    print("=====================================================================")


    print("========================weighted average===========================")

    ctbpred = df["ctbpred_test"].values
    ctbpred5 = df["ctbpred_test5"].values
    lgbpred = df["lgbpred_test"].values
    lgbpred4 = df["lgbpred_test4"].values
    xgbpred = df["xgbpred_test"].values
    xgbpred4 = df["xgbpred_test4"].values

    weighted_avg_pred = ((2*ctbpred) + (3*ctbpred5) + (lgbpred) + (lgbpred4) + (2*xgbpred) +(3*xgbpred4))/12
    weighted_avg_pred = pd.DataFrame(weighted_avg_pred, columns=['weighted_test_pred'])
    print(weighted_avg_pred)
    print(weighted_avg_pred.shape)
    weighted_avg_pred.to_csv('final_pred/weighted_test_pred.csv', index=False)
    print("=====================================================================")
    print("YOU CAN TAKE A LOOK AT final_pred DIRECTORY")