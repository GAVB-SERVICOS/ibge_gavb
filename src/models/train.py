from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Artifact,
    Output,
    Metrics,
    ClassificationMetrics,
    Dataset,
    Input
)

@component(
    packages_to_install=[
        "scikit-learn",
        "pandas",
        "joblib",
        "google-cloud-bigquery",
        "db-dtypes",
        "pyarrow",
    ]
)
def train(
    project_id: str,
    dataset: Input[Dataset],
    model_path: Output[Artifact],
    scalar_metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics],
    eta0: float = 0.01,
    max_iter: int= 40
):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
    from sklearn.linear_model import Perceptron
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import roc_curve
    from sklearn.pipeline import Pipeline
    from google.cloud import bigquery
    import pandas as pd
    import joblib

    table = dataset.metadata['bigquery_table']
    data = (
        bigquery.Client(project=project_id)
        .query(f"SELECT * EXCEPT (data_split) FROM `{table}` WHERE data_split = 'train'")
        .result()
        .to_dataframe()
    )

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # transform string labels into numeric
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    # train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    pipe = Pipeline(
        [
            ("Scaler", StandardScaler()),
            ("Model", Perceptron(max_iter=max_iter, eta0=eta0, random_state=2)),
        ]
    )

    pipe.fit(X_train, y_train)

    # testing
    y_pred = pipe.predict(X_test)

    scalar_metrics.log_metric("accuracy", accuracy_score(y_test, y_pred))

    y_score = pipe.decision_function(X_test)

    # micro-average roc
    fpr, tpr, thresholds = roc_curve(label_binarize(y_test, classes=[0,1,2]).ravel(), y_score.ravel())

    classification_metrics.log_roc_curve(fpr, tpr, thresholds)

    confusion_matrix_r = confusion_matrix(y_test, y_pred)
    classification_metrics.log_confusion_matrix(le.classes_.tolist(), confusion_matrix_r.tolist())

    model_path.uri += ".joblib"
    joblib.dump(pipe, model_path.path)