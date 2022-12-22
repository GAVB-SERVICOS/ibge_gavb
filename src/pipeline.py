import os
import yaml
import argparse
from kfp.v2 import dsl
from pathlib import Path
from components import *
from kfp.v2 import compiler
from datetime import datetime
from google.cloud import aiplatform
#from kfp.v2.dsl import component, ExitHandler
#from typing import List, Tuple, Dict, NamedTuple

parser = argparse.ArgumentParser()
parser.add_argument("--build-dir", "-b", type=Path, default="./build")
parser.add_argument("--pipeline-version", "-p", type=str, default="Pipeline-teste" )
parser.add_argument("--model-version", "-m", type=str, default="Model-teste" )
parser.add_argument("--endpoint-prefix", "-e", type=str, default="vertexai-teste" )

args = parser.parse_args()

# load config
with open('../project.yaml') as f:
    config = yaml.safe_load(f)

# Env Vars
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
PIPELINE_ROOT=f"{config['dev']['bucket']}/{config['config']['model_name']}/"

"""
Pipeline Definition
"""
@dsl.pipeline(name=args.pipeline_version, pipeline_root=PIPELINE_ROOT)
def pipeline(
    project_id: str = config['dev']['workspace']['project_id'],
    dataset_name: str = config['dev']['source']['dataset'],
    source_table_name: str = config['dev']['source']['table'],
    splited_table_name: str = config['dev']['destination']['splited'],
    destination_table: str =  config['dev']['destination']['transformed'],
    thresholds_dict_str: str='{"roc":0.6}',
    serving_container_image_uri: str="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest" 
            ):

    train_test_split_op = collect_split_component(project_id=project_id,
                                                  dataset_name=dataset_name,
                                                  source_table_name=source_table_name,
                                                  splited_table_name=splited_table_name)

    generate_stats_op = generate_stats_component(project_id=project_id,
                                                 table_name=train_test_split_op.output,
                                                 output_dir=config['dev']['bucket'],
                                                 training=True)
    
    
    transform_component_op = transform_component(project_id=project_id,
                                                 dataset_name=dataset_name,
                                                 destination_table=destination_table,
                                                 data_input=train_test_split_op.output)

    hptune_component_op = hptune_component(data_input=transform_component_op.output,
                                           #output_path=PIPELINE_ROOT+"hptune/",
                                           max_trial_count=6,
                                           parallel_trial_count=2,
                                           min_eta0=0.001,
                                           max_eta0=0.1,
                                           max_iter_space=[20,40,60])

    training_op = training_component(project_id=project_id,
                                     params=hptune_component_op.output,
                                     data_input=transform_component_op.output)

    model_evaluation_op = classification_model_eval_component(project_id=project_id,
                                                 test_set=transform_component_op.output,
                                                 model=training_op.outputs["model"],
                                                 threshold_dict_str=thresholds_dict_str)

    
    with dsl.Condition(model_evaluation_op.outputs["deploy"] == "true",
                       name="deploy-vertexai-teste"):
    
            deploy_model_op = deploy_model_component(
                           model=training_op.outputs["model"],
                           project_id=project_id,
                           region="us-central1",
                           serving_container_image_uri=serving_container_image_uri,
                           model_version=args.model_version,
                           endpoint_prefix=args.endpoint_prefix)
        


if __name__=="__main__":

    """
    Compile and Run
    """

    pipeline_name = args.pipeline_version

    # create build dir
    os.makedirs(args.build_dir, exist_ok=True)

    # compile pipeline
    build_path = str(args.build_dir / config['config']['template_file'])
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=build_path)

    # Submit Job
    aiplatform.PipelineJob(
        display_name=pipeline_name, 
        template_path=build_path, 
        pipeline_root=PIPELINE_ROOT, 
        enable_caching=False
    ).submit()