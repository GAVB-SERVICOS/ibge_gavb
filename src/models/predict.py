from kfp.v2.dsl import component
from kfp.v2.dsl import (
    Output,
    Input,
    Model,
    Dataset
)

@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud",
        "google-cloud-bigquery",
        "db-dtypes",
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "scikit-learn",
        "pandas",
        "joblib",
        "pyarrow",
    ],
)
def predict(
    project_id: str, source_table: Input[Dataset], output_table: str, model: Input[Model], results: Output[Dataset]
):
    from google.cloud import bigquery, storage, aiplatform
    import pandas as pd
    import joblib

    client = bigquery.Client(project=project_id)

    table = source_table.metadata['bigquery_table']
    data = (
        client
        .query(f"SELECT * FROM `{table}` WHERE data_split = 'test'")
        .result()
        .to_dataframe()
    )

    X = data.iloc[:, :-2]
    y = data.iloc[:, -2]

    # download model
    model_local_path = "model.joblib"

    if model.uri.startswith("gs://"):
        pipe = joblib.load(model.path)
    elif model.uri.startswith("aiplatform://v1"): # Vertex Model
        model_uri = model.metadata.get('model_artifact_uri')

        if model_uri is None:
            model_client = aiplatform.gapic.ModelServiceClient(
                client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
            )
            registed_model = model_client.get_model(model.metadata['resourceName'])
            model_filename = "model_path.joblib"
            model_uri = registed_model.artifact_uri + "/" + model_filename

        with open(model_local_path, 'wb') as m:
            storage.Client().download_blob_to_file(model_uri, m)

        pipe = joblib.load(model_local_path)
    else:
        raise Exception("Unrecognized model type")

    # testing
    y_pred = pipe.predict(X)

    output = X
    output['species_predicted'] = y_pred

    table = client.get_table(output_table)
    client.insert_rows_from_dataframe(table, output)

    results.metadata['type'] = 'result'
    results.metadata['bigquery_table'] = output_table