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
    "eta0": hpt.DoubleParameterSpec(min=0.001, max=0.1, scale="log"),
    "max_iter": hpt.DiscreteParameterSpec(values=[20,40,60], scale=None)
}

# metric spec
metric_spec={'accuracy':'maximize'}

# job
iris_job = aiplatform.CustomJob(display_name="iris-hptune-test",
                              worker_pool_specs=worker_pool_specs,
                              staging_bucket='gs://hptune-bucket')


hp_job = aiplatform.HyperparameterTuningJob(
    display_name='iris-hptune-test',
    custom_job=iris_job,
    metric_spec=metric_spec,
    parameter_spec=parameter_spec,
    max_trial_count=6,
    parallel_trial_count=2,
    search_algorithm=None)

hp_job.run()




