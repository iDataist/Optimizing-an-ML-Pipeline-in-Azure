import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse
from azureml.core.run import Run
import joblib

# Onehot encode non-numerical data and normalize numerical data
def clean_data(df):
    categorical_col = df.loc[:, df.dtypes==object].columns
    numerical_col = [x for x in df.columns if x not in categorical_col]
    categorical_df = pd.get_dummies(df[categorical_col],
                                    columns=categorical_col,
                                    drop_first=True)
    numerical_df = pd.DataFrame(StandardScaler().fit_transform(df[numerical_col]),
                                columns=numerical_col)
    cleaned_df = pd.concat([categorical_df.reset_index(drop=True),
                            numerical_df.reset_index(drop=True)],
                            axis=1)
    return cleaned_df

# Split the data into training and testing sets
def data_split():
    train_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    val_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_validate.csv"
    test_url = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_test.csv"
    df_list = [pd.read_csv(url) for url in [train_url, val_url, test_url]]
    df = clean_data(pd.concat(df_list, axis=0))
    df = shuffle(df, random_state=0)
    train_data = df[:int(0.7*len(df))]
    val_data = df[int(0.7*len(df)):int(0.8*len(df))]
    test_data = df[int(0.8*len(df)):]

    # Sample the training set to make the classes balanced
    train_data_1 = train_data[train_data.y_yes==1]
    train_data_0 = train_data[train_data.y_yes==0].sample(n=len(train_data_1),
                                                          random_state=0)
    train_data = pd.concat([train_data_1, train_data_0],axis=0)
    train_data = shuffle(train_data, random_state=0)
    return train_data, val_data, test_data 

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', 
                        type=float, 
                        default=1.0, 
                        help="Inverse of regularization strength")
    parser.add_argument('--max_iter', 
                        type=int, 
                        default=100, 
                        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # Split data into features and targets
    train_data, val_data, test_data = data_split()
    data = pd.concat([train_data, val_data], axis=0)
    X_train = data.drop(columns=['y_yes'])
    y_train = data['y_yes']
    X_test = test_data.drop(columns=['y_yes'])
    y_test = test_data['y_yes']

    # Train and test the model 
    model = LogisticRegression(C=args.C, 
                               max_iter=args.max_iter).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob[:, 1], average="weighted")
    
    # Save the model
    joblib.dump(model, filename="./outputs/hyperdrive_model.joblib")
    
    # logging
    run = Run.get_context()
    run.log("Inverse of regularization strength:", args.C)
    run.log("Maximum number of iterations:", args.max_iter)
    run.log("Weighted AUC", auc)

if __name__ == '__main__':
    main()


