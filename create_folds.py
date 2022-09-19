import pandas as pd 
from sklearn.model_selection import StratifiedKFold


if __name__ == "__main__":
    
    train = pd.read_csv("train.csv.zip")
    train.loc[:, "kfold"] = -1
    train = train.sample(frac=1).reset_index(drop=True)

    target = train["target"].values

    # because the data is not balanced, we use stratifiedkold (each k fold wi will have the same sample of 2 classes)
    skf = StratifiedKFold(n_splits=10)
    for fold, (train_id,valid_id) in enumerate(skf.split(X = train, y = target)):
        print(len(train_id),len(valid_id))
        train.loc[valid_id,"kfold"] = fold

    train.to_csv("train_folds.csv", index=False)
