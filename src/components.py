from venv import create
from numpy import average
import pandas as pd
from typing import NamedTuple
from kfp.v2.dsl import component
from collections import namedtuple
from kfp.v2.dsl import (
    Artifact,
    Output,
    OutputPath,
    Metrics,
    ClassificationMetrics,
    Dataset,
    Input,
    Model
)


@component(
    base_image="us-central1-docker.pkg.dev/gavb-poc-bu-mlops-f-store/vertexai-teste/vertexai-iris-teste:latest",
    packages_to_install=["google-cloud-bigquery", 
                         "pandas",
                         "db-dtypes"]
)
def collect_split_component(project_id: str,
                            dataset_name: str,
                            source_table_name: str,
                            splited_table_name: str,
                            data:Output[Dataset]):

    """Kubeflow component that splits data between train 80%, test 10%, valdiate 10%

    parameters:
        project_id: The id of GCP project.
        dataset_name: The name of the dataset where raw data are and splitted data will be saved.
        source_table_name: The name of the raw data table.
        splitted_table_name: The name of splitted data will be persisted.
        data: Dataset output, bigquery address were splitted data was persisted.

    Return: None
    """                        

    import pandas as pd
    from google.cloud import bigquery

    query=f"""
        CREATE OR REPLACE TABLE `{project_id}.{dataset_name}.{splited_table_name}` AS
        WITH add_id AS(SELECT *, GENERATE_UUID() transaction_id FROM `{project_id}.{dataset_name}.{source_table_name}`)
        SELECT *,
            CASE 
                WHEN MOD(ABS(FARM_FINGERPRINT(transaction_id)),10) < 8 THEN "TRAIN" 
                WHEN MOD(ABS(FARM_FINGERPRINT(transaction_id)),10) < 9 THEN "VALIDATE"
                ELSE "TEST"
            END AS splits
        FROM add_id """
    
    #Instanciate Client 
    bq = bigquery.Client(project = project_id)
    
    # Executing the job
    job = bq.query(query=query)
    job.result()

    data.metadata["tags"] = "Dataset dividido em Train, Test e Validacao"
    data.metadata["bigquery_table"] = f"{project_id}.{dataset_name}.{splited_table_name}"


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn",
                         "pandas",
                         "pandas-gbq",
                         "db-dtypes",
                         "google-cloud-bigquery"])

def transform_component(project_id: str,
                        dataset_name: str, 
                        destination_table: str,
                        data_input: Input[Dataset],
                        data_output: Output[Dataset],
                        ):
    
    """Kubeflow component of transformations to our example problem.

    parameters:
        project_id: The id of GCP project.
        dataset_name: The name of the dataset where raw data are and splitted data will be saved.
        destination_table: The name of splitted data will be persisted.
        data_input: Metadata from collect_split_component were splitted data was persisted.
        data_output: Metadata output bigquery address were transformed data was persisted.
    
    Return: None
    """
                        

    import pandas as pd
    from typing import NamedTuple
    from google.cloud import bigquery
    from collections import namedtuple
    from sklearn.preprocessing import StandardScaler, LabelEncoder


    #Instanciate Client 
    bq = bigquery.Client(project = project_id)

    table = data_input.metadata["bigquery_table"]
    query=f"SELECT * FROM `{table}`"
    data = (
            bq.query(query)
            .result()
            .to_dataframe(create_bqstorage_client=True)
        ) 

    # Transform the labels
    le = LabelEncoder()
    le.fit(data['Species'])
    data['Species'] = le.transform(data['Species'])

    # Spliting in train, test, validate
    train_data = data.loc[data["splits"]=="TRAIN", :].drop(["Id","transaction_id","splits"], axis=1)
    test_data = data.loc[data["splits"]=="TEST", :].drop(["Id","transaction_id","splits"], axis=1)
    validate_data = data.loc[data["splits"]=="VALIDATE", :].drop(["Id","transaction_id","splits"], axis=1)

    # Spliting in X and y
    X_train= train_data.iloc[:, :-1]
    y_train= train_data.iloc[:, -1]
    y_train = y_train.reset_index().drop('index', axis=1)

    X_test= test_data.iloc[:, :-1]
    y_test= test_data.iloc[:, -1]
    y_test = y_test.reset_index().drop('index', axis=1)

    X_val= validate_data.iloc[:, :-1]
    y_val= validate_data.iloc[:, -1]
    y_val = y_val.reset_index().drop('index', axis=1)

    # Standard Scaler
    scaler = StandardScaler()

    # Treino
    scaler.fit(X_train)
    colunas = list(X_train.columns.values)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=colunas)
    X_train["splits"] = "TRAIN" 
    
    # Teste 
    scaler.fit(X_test)
    colunas = list(X_test.columns.values)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=colunas)
    X_test["splits"] = "TEST"


    # Validacao
    scaler.fit(X_val)
    colunas = list(X_val.columns.values)
    X_val = scaler.transform(X_val)
    X_val = pd.DataFrame(X_val, columns=colunas)
    X_val["splits"] = "VAL"

    # Unificando os datasets
    data_train = pd.concat([X_train, y_train], axis=1)
    data_test = pd.concat([X_test, y_test], axis=1)
    data_val = pd.concat([X_val, y_val], axis=1)

    data = pd.concat([data_train, data_test, data_val], ignore_index=True)


    # Persistindo no bq

    data.to_gbq(destination_table=f"{project_id}.{dataset_name}.{destination_table}", project_id=project_id, if_exists="replace")

    data_output.metadata["tags"] = "Dataset transformado"
    data_output.metadata["bigquery_table"] = f"{project_id}.{dataset_name}.{destination_table}"


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn",
                         "pandas",
                         "db-dtypes",
                         "gcsfs",
                         "google-cloud-bigquery"])

def training_component(data_input: Input[Dataset],
                       params: Input[Artifact],
                       project_id: str,
                       model: Output[Artifact],
                       #eta0: float=0.01,
                       #max_iter: int=40
                      ):
    
    """Kubeflow component to training model to our example problem.

    parameters:
        data_input: Metadata from transform_component were transform data was persisted.
        project_id: The id of GCP project.
        model: Metadata output of the model trained as joblib.
        eta0: Constant by which the updates are multiplied.
        max_iter: The maximum number of passes over the training data (aka epochs). 
                  It only impacts the behavior in the fit method, and not the partial_fit method.
    
    Return: None
    """
    import json
    import pandas as pd
    import numpy as np
    import gcsfs
    from google.cloud import bigquery
    from sklearn.linear_model import Perceptron
    import joblib

    # Instanciamento do cliente.
    bq = bigquery.Client(project = project_id)

    # Capturando os dados.
    table = data_input.metadata["bigquery_table"]
    query=f"SELECT * FROM `{table}`"
    data = (
            bq.query(query)
            .result()
            .to_dataframe(create_bqstorage_client=True)
        )
    
    # Loading params
    if params.uri.startswith("gs://"):
         gcs_file_system = gcsfs.GCSFileSystem()
         gcs_json_path = params.uri
         with gcs_file_system.open(gcs_json_path) as f:
         #f = open(params.uri, "r")  #.replace("json_path.json", "")
            parameters = json.load(f)
    
    print(f"Eta0: {parameters[1]}")
    print(f"max_iter: {parameters[3]}")
    eta0 = parameters[1]
    max_iter = parameters[3]
    
    
    
    # Spliting in train, test, validate
    print("Splitting os dados")
    train_data = data.loc[data["splits"]=="TRAIN", :].drop(["splits"], axis=1)

    # Spliting in X and y
    X_train= train_data.iloc[:, :-1]
    y_train= train_data.iloc[:, -1]
    y_train = y_train.astype('int')

    # Training the model
    print("Treinando modelo")
    modelo = Perceptron(max_iter=max_iter, eta0=eta0, random_state=42)
    modelo.fit(X_train, y_train)
    f.close()

    # Dumping model
    model.uri +=".joblib"
    joblib.dump(modelo,model.path)

    # Metadata
    model.metadata["framework"] = "Perceptron-sklearn"
    model.metadata["model_path"] = model.path
    model.metadata["param_eta0"] = eta0
    model.metadata["param_max_iter"] = max_iter


@component(base_image="python:3.9",
           packages_to_install=[        
                   "google-cloud",
                   "google-cloud-bigquery",
                   "db-dtypes",
                   "pandas-gbq",
                   "google-cloud-storage",
                   "google-cloud-aiplatform",
                   "scikit-learn",
                   "pandas",
                   "joblib",
                   "pyarrow"
                   ])

def predict_component(
        project_id: str,
        dataset_name: str,
        source_table: Input[Dataset],
        raw_table: Input[Dataset],
        output_table: str,
        model_input: Input[Model],
        model_output: Output[Artifact],
        results: Output[Dataset]
    ):
        """Kubeflow component to predicting by loading from a .joblib in a GCP bucket.

        """

        from google.cloud import bigquery
        import pandas as pd
        import joblib

        # data client
        client = bigquery.Client(project=project_id)

        # Tabelas
        table = source_table.metadata['bigquery_table']
        table_output = f"{project_id}.{dataset_name}.{output_table}"

        # Coletando dados
        data = (
            client
            .query(f"SELECT * FROM `{table}` WHERE splits = 'VAL'")
            .result()
            .to_dataframe()
        )

        table_raw = raw_table.metadata['bigquery_table']

        # Coletando dados
        data_raw = (
            client
            .query(f"SELECT * FROM `{table_raw}` WHERE splits = 'VALIDATE'")
            .result()
            .to_dataframe()
        )


        if model_input.uri.startswith("gs://"):
            model = joblib.load(model_input.path)
        elif model_input.uri.startswith("aiplatform://v1"):
            model_uri = model_input.metadata.get('model_artifact_uri')

        data=data.drop("splits", axis=1)
        X = data.iloc[:, :-1]
       

        data_raw = data_raw.drop("splits", axis=1)
        X_raw = data_raw.iloc[:, :-1]
        #y = data_raw.iloc[:, -1]
        
        # Perdicting
        y_pred = model.predict(X)
        output = X_raw
        output['predicted'] = y_pred

        # Persisting on 
        output.to_gbq(destination_table=table_output, project_id=project_id, if_exists="replace")


        # Logs
        results.metadata['type'] = 'result'
        results.metadata["bigquery_table"] = table_output


@component(
    base_image="python:3.9",
    packages_to_install = [
        "scikit-learn",
        "pandas",
        "joblib",
        "db-dtypes",
        "google-cloud-bigquery",
        "google-cloud-storage"])

def classification_model_eval_component(
    project_id: str,
    test_set: Input[Dataset],
    model: Input[Model],
    threshold_dict_str: str,
    classification_metrics: Output[ClassificationMetrics],
    scalar_metrics: Output[Metrics]

)-> NamedTuple("output", [("deploy", str)]):

    """Kubeflow component to evaluation of classification models

    parameters:
        project_id: The id of GCP project.
        test_set: Metadata input with test data from transform_component. 
        model: Metada input model from training_component.
        threshold_dict_str: Threshold to decide if the trained model will be deployed or not.
        classification_metrics: Metadata Confusion metric and ROC Curve
        scalar_metrics: Metadata precision_score, recall_score, accuracy_score, f1-score
    
    Return: NamedTuple() with the bool value if the model should be deployed or not
    
    
    """

    import json
    import typing
    import joblib
    import logging
    import pandas as pd
    from collections import namedtuple
    from google.cloud import bigquery
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2:
            cond = "true"
        return cond

    # data client
    client = bigquery.Client(project=project_id)
    
    # Coletando dados    
    table = test_set.metadata['bigquery_table']

    data = (
        client
        .query(f"SELECT * FROM `{table}` WHERE splits = 'TEST'")
        .result()
        .to_dataframe()
    )


    # Carregando o modelo treinado
    if model.uri.startswith("gs://"):
        modelo = joblib.load(model.path)
    elif model.uri.startswith("aiplatform://v1"):
        modelo_uri = model.metadata.get('model_artifact_uri')

    # 
    data=data.drop("splits", axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.astype('int')

    # Testing on test data
    y_pred = modelo.predict(X)
    scalar_metrics.log_metric("accuracy", accuracy_score(y, y_pred, ))
    scalar_metrics.log_metric("recall", recall_score(y,y_pred, average='micro') )
    scalar_metrics.log_metric("precision", precision_score(y,y_pred, average='weighted'))
    scalar_metrics.log_metric("f1-score", f1_score(y,y_pred, average='weighted'))
    scalar_metrics.log_metric("threshold", threshold_dict_str)

    # y-score
    y_score = modelo.decision_function(X)

    # micro-average roc
    fpr, tpr, thresholds = roc_curve(label_binarize(y, classes=[0,1,2]).ravel(), y_score.ravel())
    classification_metrics.log_roc_curve(fpr, tpr, thresholds)

    # confusion metrics
    confusion_matrix_m = confusion_matrix(y, y_pred)
    classification_metrics.log_confusion_matrix(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], confusion_matrix_m.tolist())

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    threshold_dict = json.loads(threshold_dict_str)
    deploy = threshold_check(float(accuracy), float(threshold_dict['roc']))


    # namedtuple
    deploy_eval = namedtuple("output", ["deploy"])

    return deploy_eval(deploy=deploy)

@component(base_image="python:3.9",
           packages_to_install=["google-cloud-aiplatform", "scikit-learn",  "kfp"])
def deploy_model_component(
    model: Input[Model],
    project_id: str,
    region: str,
    serving_container_image_uri: str,
    model_version: str,
    endpoint_prefix: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]

):

    """Kubeflow component to deploy the model in an endpoint.

    parameters:
        model: Metada input model from training_component.
        project_id: The id of GCP project.
        region: Region were the endpoint will be created.
        serving_container_image_uri: Docker image to serve the predictions.
        model_version: Model version deployed.
        endpoint_prefix: Prefix to the endpoint name.
        vertex_endpoint: Metadata with info about the endpoint.
        vertex_model: Metadata with info about the model created.



    Return: None
    """

    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    
    #DISPLAY_NAME="vertexai-test-iris"
    MODEL_NAME=f"vertexai-test-{model_version}"
    ENDPOINT_NAME=f"{endpoint_prefix}-endpoint"
    
    def create_endpoint():
        endpoints = aiplatform.Endpoint.list(
            filter='',
            #order_by='create_time_desc',
            project=project_id,
            location=region
        )
        if ENDPOINT_NAME not in endpoints:
                endpoint = aiplatform.Endpoint.create(
                    display_name=ENDPOINT_NAME, project=project_id, location=region
            )
        else: 
            return "Endpoint existente!"


    # Import a model
    model_upload = aiplatform.Model.upload(
        display_name= MODEL_NAME,
        artifact_uri = model.uri.replace("/model.joblib", ""),
        serving_container_image_uri = serving_container_image_uri,
        serving_container_health_route=f"/v1/models/{MODEL_NAME}",
        serving_container_predict_route=f"/v1/models/{MODEL_NAME}:predict",
        )

    # find endpoints
    for e in aiplatform.Endpoint.list():
        if e.display_name.startswith(endpoint_prefix): 
            endpoint_listed = e
    print(endpoint_listed.display_name)

    models = endpoint_listed.list_models()
    if len(models) == 1:
        oldmodel = models[0]
    
        # deploy no endpoint
        endpoint_listed.deploy(
            model=model_upload,
            deployed_model_display_name=MODEL_NAME,
            traffic_percentage=100,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1
        )

        # undeploy o modelo velho
        endpoint_listed.undeploy(
            deployed_model_id = oldmodel.id
        )
    else:
        create_endpoint()

        for e in aiplatform.Endpoint.list():
            if e.display_name.startswith(endpoint_prefix): 
                endpoint_listed = e
        
        # deploy no endpoint
        endpoint_listed.deploy(
            model=model_upload,
            deployed_model_display_name=MODEL_NAME,
            traffic_percentage=100,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=1
        )

    vertex_model.metadata["model_deployed"] = MODEL_NAME
    vertex_endpoint.metadata["endpoint_name"] = ENDPOINT_NAME
    

@component(
    base_image="python:3.9",
    packages_to_install = [
        "scikit-learn",
        "pandas",
        "joblib",
        "db-dtypes",
        "google-cloud-bigquery",
        "google-cloud-storage"])

def regression_model_eval_component(
    project_id: str,
    test_set: Input[Dataset],
    model: Input[Model],
    threshold_dict_str: str,
    scalar_metrics: Output[Metrics]) -> NamedTuple("output", [("deploy", str)]):

    import json
    import typing
    import joblib
    import logging
    import pandas as pd
    from collections import namedtuple
    from google.cloud import bigquery
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

    def threshold_check(val1, val2):
        cond = "false"
        if val1 >= val2:
            cond = "true"
        return cond

    # data client
    client = bigquery.Client(project=project_id)
    
    # Coletando dados    
    table = test_set.metadata['bigquery_table']

    data = (
        client
        .query(f"SELECT * FROM `{table}` WHERE splits = 'TEST'")
        .result()
        .to_dataframe()
    )

        # Carregando o modelo treinado
    if model.uri.startswith("gs://"):
        modelo = joblib.load(model.path)
    elif model.uri.startswith("aiplatform://v1"):
        modelo_uri = model.metadata.get('model_artifact_uri')

    # 
    data=data.drop("splits", axis=1)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    y = y.astype('int')

    # Testing on test data
    y_pred = modelo.predict(X)
    scalar_metrics.log_metric("MAE", mean_absolute_error(y, y_pred))
    scalar_metrics.log_metric("MSE", mean_squared_error(y,y_pred) )
    scalar_metrics.log_metric("MAPE", mean_absolute_percentage_error(y,y_pred))
    scalar_metrics.log_metric("threshold", threshold_dict_str)

    # MAPE
    mape = mean_absolute_percentage_error(y, y_pred)
    threshold_dict = json.loads(threshold_dict_str)
    deploy = threshold_check(float(mape), float(threshold_dict['roc']))

    model.metadata["accuracy"]=float(mape)
    model.metadata["deploy"]=deploy

    # namedtuple
    deploy_eval = namedtuple("output", ["deploy"])

    return deploy_eval(deploy=deploy)



@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas",
        "db-dtypes",
        "google-cloud-bigquery",
        "tensorflow-data-validation"])

def generate_stats_component(project_id: str,
                            table_name: Input[Dataset],
                            results: Output[Artifact],
                            output_dir: str,
                            training: bool = False,
                            ):
    
    import pandas as pd
    from google.cloud import bigquery
    import tensorflow_data_validation as tfdv

    # data client
    client = bigquery.Client(project=project_id)

    # Metadata tabela
    table = table_name.metadata['bigquery_table']

    if training == True:

        data_train = (
        client
        .query(f"SELECT * FROM `{table}` WHERE splits = 'TRAIN'")
        .result()
        .to_dataframe()
        )

        data=data_train.drop('splits', axis=1)
    
    else:
    
        all_data = (
        client
        .query(f"SELECT * FROM `{table}`")
        .result()
        .to_dataframe()
        )
        data =all_data.drop('splits', axis=1)

    # Gerando os stats
    data_stats = tfdv.generate_statistics_from_dataframe(data)
    stats_path = output_dir

    # Saving
    tfdv.write_stats_text(data_stats, results.path)

    results.metadata["stats_dir"]= results.path



@component(base_image="us-central1-docker.pkg.dev/gavb-poc-bu-mlops-f-store/vertexai-teste/hptune-vertxai",
           packages_to_install=["google-cloud-aiplatform"])
def hptune_component(data_input: Input[Dataset],
                     json_path: Output[Artifact],
                     max_trial_count: int,
                     parallel_trial_count: int,
                     min_eta0: float,
                     max_eta0: float,
                     max_iter_space: list):

                     """Kubeflow component to trigger a Hyperparamenter Tunning job from a vertexai pipeline.

                     Params:
                        data_input: Data from transform componente, just to create dependency.
                        output_path: Path were the file .json will be saved.
                        max_trial_count: Max number of trial to tune the parameters.
                        parallel_trial_count: 
                        min_eta0: Min value of eta0 parameter.
                        max_eta0: Max value of eta0 parameter.
                        max_iter_space: Space for max_iter parameter.



                     Return:
                        output_path: The path to .json file with the best params found on the hyperparanmeter job.
                     
                     """

                     import json
                     from google.cloud import aiplatform
                     from google.cloud.aiplatform import hyperparameter_tuning as hpt

                     # worker pool specs
                     worker_pool_specs = [{
                         "machine_spec": {
                             "machine_type": "n1-standard-4",
                             "accelerator_type": "NVIDIA_TESLA_V100",
                             "accelerator_count": 2
                         },
                         "replica_count":1,
                         "container_spec":{
                             "image_uri": "us-central1-docker.pkg.dev/gavb-poc-bu-mlops-f-store/vertexai-teste/hptune-vertxai"
                         }
 
                     }]
 
                     # parameter spec
                     parameter_spec = {
                         "eta0": hpt.DoubleParameterSpec(min=min_eta0, max=max_eta0, scale="log"),
                         "max_iter": hpt.DiscreteParameterSpec(values=max_iter_space, scale=None)
                     }
 
                     # metric spec
                     metric_spec={'accuracy':'maximize'}
 
                     # INstanciate the job
                     iris_job = aiplatform.CustomJob(display_name="iris-hptune-test",
                                                     worker_pool_specs=worker_pool_specs,
                                                     staging_bucket='gs://hptune-bucket')
 
 
                     # Running the job
                     hp_job = aiplatform.HyperparameterTuningJob(
                         display_name='iris-hptune-test',
                         custom_job=iris_job,
                         metric_spec=metric_spec,
                         parameter_spec=parameter_spec,
                         max_trial_count=max_trial_count,
                         parallel_trial_count=parallel_trial_count,
                         search_algorithm=None)
 
                     hp_job.run()
 
                     #Getting the best parameters
                     best = sorted(hp_job.trials, key=lambda x: x.final_measurement.metrics[0].value)[0]
 
                     best_params = []
                     for param in best.parameters:
                         best_params.append(f'--{param.parameter_id}')
                         best_params.append(param.value)
                      
                     json_path.uri +=".json"
                     
                     with open(json_path.path, 'w') as f:
                         json.dump(best_params, f)
                    

                     json_path.metadata["json_path"]= json_path.path
 
   


    

    

   







   

    





