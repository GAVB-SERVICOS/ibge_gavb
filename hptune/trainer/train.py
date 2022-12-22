import pandas as pd
import argparse
import hypertune
from google.cloud import bigquery
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def get_args():
    """
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--eta0', required=True, type=float, help='eta0')
    parser.add_argument('--max_iter', required=True, type=int, help='max_iter')
    args = parser.parse_args()
    return args


def create_dataset():
    """
    
    """

    bq = bigquery.Client(project="gavb-poc-bu-mlops-f-store")

    table = "gavb-poc-bu-mlops-f-store.vertexai_teste.iris-vertexai_transformed"
    query=f"SELECT * FROM `{table}`"
    data = (
            bq.query(query)
            .result()
            .to_dataframe(create_bqstorage_client=True)
        )

    print("Splitting os dados")
    train_data = data.loc[data["splits"]=="TRAIN", :].drop(["splits"], axis=1)
    test_data = data.loc[data["splits"]=="TEST", :].drop(["splits"], axis=1)

    # Spliting in X and y
    X_train= train_data.iloc[:, :-1]
    y_train= train_data.iloc[:, -1]
    y_train = y_train.astype('int')

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    y_test = y_test.astype('int')
    
    return X_train, y_train, X_test, y_test

def create_model(max_iter, eta0):

    """
    
    """
    
    # Training the model
    print("Treinando modelo")
    modelo = Perceptron(max_iter=max_iter, eta0=eta0, random_state=42)

    return modelo


def main():
    args = get_args()

    # Getting data
    x_train, y_train, x_test, y_test = create_dataset()

    # Creating model
    model = create_model(max_iter=args.max_iter, eta0=args.eta0)

    # Fitting
    model.fit(x_train, y_train)

    #Define Metric
    y_pred = model.predict(x_test)
    hp_metric = accuracy_score(y_test, y_pred)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value = hp_metric,
        #global_step=NUM_EPOCHS
    )

if __name__ == "__main__":
    main()

