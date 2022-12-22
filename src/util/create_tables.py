from typing import List, Tuple, Optional, Union
from kfp.v2 import dsl
from kfp.v2.dsl import component
from matplotlib.pyplot import isinteractive


def create_tables(
    env: str,
    pipeline_version: str,
    config_file: str = "project.yaml",
    tables_dir: str = "tables",
    timeout: int = 300,
    wait_for_creation: bool = True,
    github_config: dict = {},
    run_local:bool = False
):
    import os
    import yaml

    with open(config_file, "r") as filename:
        config = yaml.safe_load(filename)

    file_contents = []

    for root, dirs, files in os.walk(tables_dir):
        type_of_table = root.rsplit("/", maxsplit=1)[-1]

        if type_of_table not in ("workspace", "destination"):
            continue

        for filename in files:
            if not filename.endswith(".yaml"):
                continue
            with open(os.path.join(root, filename), "r") as f:
                contents = yaml.safe_load(f)

            file_contents.append((type_of_table, filename, contents))

    @component(
        base_image="python:3.9",
        output_component_file="build/create-tables.yaml",
        packages_to_install=[
            "PyGithub",
            "google-cloud-secret-manager",
            "pyyaml",
            "google-cloud-bigquery",
            "cryptography"
        ]
    )
    def create_tables_op(
        env: str,
        pipeline_version: str,
        config: dict,
        file_contents: List[Tuple[str, str, dict]],
        job_name: str = dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
        job_id: str = dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        timeout: int = 300,
        wait_for_creation: bool = True,
        github_config: dict = {},
        run_local:bool=False
    ) -> dict:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
        from github import Github
        import yaml
        import time
        from datetime import datetime
        import json
        from io import BytesIO

        from google.cloud import secretmanager

        from github import Github, GithubIntegration

        # yaml dumper to fix indentation
        class Dumper(yaml.Dumper):
            def increase_indent(self, flow=False, *args, **kwargs):
                return super().increase_indent(flow=flow, indentless=False)

        def get_config(env, config:Union[dict, list], table_config:dict):
            if isinstance(config, list):
                config = {
                    c['tableConfigId'] : c
                    for c in config
                }

            if config_id := table_config.get('tableConfigId'):
                if config_id not in config:
                    raise Exception("Table {} has tableConfigId that does not exist in project config file for env {}".format(table_config.get('tableName'), env))
                return config[config_id]
            else:
                if 'project_id' in config and 'dataset' in config:
                    return config
                elif 'default' in config:
                    return config['default']
                else:
                    raise Exception("Table {} has no tableConfigId and project config has no default project or dataset for env {}".format(table_config.get('tableName'), env))

        def reorder_dict(table_config):
            new_dict = {
                "tableId": None,
                "tableName": None,
                "zone": None,
                "idProject": None,
                "environment": None,
                "dataset": None
            }

            for k, v in table_config.items():
                new_dict[k] = v

            return new_dict

        today = datetime.now().strftime("%Y%m%d")

        name_template = "{table_name}_{version}_{job_id}_{date}"

        files_to_create = []

        for type_of_table, filename, table_config in file_contents:
            # check if table has alias
            prepared_config = get_config(env, config[env][type_of_table], table_config)
            project = prepared_config["project_id"]
            dataset = prepared_config["dataset"]
            # project = config[type_of_table]["project_id"]
            # dataset = config[type_of_table]["dataset"]

            table_config["idProject"] = project
            table_config["environment"] = env
            table_config["zone"] = project

            if table_config.get("dataset") is None:
                table_config["dataset"] = dataset

            table_config['tableId'] = table_config["tableName"]
            table_config["tableName"] = name_template.format(
                table_name=table_config["tableName"],
                version=pipeline_version,
                job_id=job_id,
                date=today,
            )
            reordered_dict = reorder_dict(table_config)
            reordered_dict.pop("tableConfigId", None)
            file_content = yaml.dump(reordered_dict, sort_keys=False, Dumper=Dumper)
            filename = table_config["tableName"] + ".yaml" # filename.rsplit(".", maxsplit=1)[0] + f"-{job_id}.yaml"
            full_filename = f"{project}/{dataset}/{filename}"
            table_config["tableFullName"] = ".".join(
                [project, dataset, table_config["tableName"]]
            )
            files_to_create.append([table_config, full_filename, f"+ {full_filename}", file_content])
        
        table_names = '\n'.join(["- " + c[2].get('tableFullName') for c in file_contents])

        if not run_local:
            if env == "dev":
                project_id = "ml-tools-developer"
                secret_id = "github_app_private_key"
                version_id = github_config.get("secret_version") or "1"
                # repoName = "grupoboticario/ds-mlops-vertex-reference"
                repoName = github_config.get("repo_name") or "grupoboticario/dataops-gcp-platform-bigquery"
                source_branch = "develop"
            elif env == "prod":
                project_id = "ml-tools-production"
                secret_id = "github_app_private_key"
                version_id = github_config.get("secret_version") or "1"
                repoName = github_config.get("repo_name") or  "grupoboticario/dataops-gcp-platform-bigquery"
                source_branch = "main"
            else:
                raise Exception("Env must be either 'dev' or 'prod'")

            config = config[env]

            secret_client = secretmanager.SecretManagerServiceClient()
            secret_name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
            response = secret_client.access_secret_version(request={"name": secret_name})
            private_key = response.payload.data.decode("UTF-8")

            GITHUB_APP_ID = "172241"
            integration = GithubIntegration(
                GITHUB_APP_ID, private_key)

            owner = "grupoboticario"
            repository = repoName.split('/')[-1]

            install = integration.get_installation(owner, repository)

            access = integration.get_access_token(install.id)

            g = Github(access.token)

            repo = g.get_repo(repoName)
            sb = repo.get_branch(source_branch)

            if env == 'prod':
                target_branch = job_id
                repo.create_git_ref(ref="refs/heads/" + target_branch, sha=sb.commit.sha)
            else:
                target_branch = source_branch
        
            for _, *f in files_to_create:
                repo.create_file(
                    *f,
                    target_branch
                    # full_filename, f"+ {full_filename}", file_content, target_branch
                )

            if env == 'prod':
                print("Env = prod, creating PR")
                pr = repo.create_pull(
                    title=f"Vertex Pipeline: {job_name} - Create Sharded Tables - Run Id: {job_id}",
                    body="Tables to be created: \n" + table_names,
                    head=target_branch,
                    base=source_branch,
                )

                print("Pull Request para criação de tabelas realizado: ", pr.html_url + "/files")

            client = bigquery.Client()
            start_time = datetime.now()

            while (datetime.now() - start_time).seconds < timeout:
                if not wait_for_creation:
                    print("Don't wait for table creation")
                    break
                tables_created = []
                for *_, table_config in file_contents:
                    table_id = f"{table_config['idProject']}.{table_config['dataset']}.{table_config['tableName']}"
                    print(f"Checking if {table_id} exists")
                    try:
                        client.get_table(table_id)  # Make an API request.
                        tables_created.append(True)
                        print("Table {} already exists.".format(table_id))
                    except NotFound:
                        print("Table {} is not found.".format(table_id))
                        tables_created.append(False)
                        break
                if all(tables_created):
                    print("All tables created")
                    break
                else:
                    print("Tables NOT created. Sleeping for 30 seconds")
                    time.sleep(30)
            else:
                raise TimeoutError("Wait for tables created timeout")

        else:
            client = bigquery.Client()
            for config, *_ in files_to_create:
                print("-------------------------------------------------------------")
                print("Table Config:" , config)
                schema_bin = BytesIO(json.dumps(config["metadata"]).encode())
                schema = client.schema_from_json(schema_bin)

                table_name = f"{config['idProject']}.{config['dataset']}.{config['tableName']}"
                table_ref = bigquery.Table(
                    table_name, schema=schema
                )

                if 'partitionType' in config and 'partitionBy' in config and 'partitionRule' in config:
                    print("Partition definition found")
                    ruleMap = {
                        "HOUR" : bigquery.TimePartitioningType.HOUR,
                        "DAY" : bigquery.TimePartitioningType.DAY,
                        "MONTH" : bigquery.TimePartitioningType.MONTH,
                        "YEAR" : bigquery.TimePartitioningType.YEAR,
                    }

                    if config['partitionType'] == 'datetime':
                        table_ref.time_partitioning = bigquery.TimePartitioning(
                            type_=ruleMap[config['partitionRule']],
                            field=config['partitionBy'],
                            expiration_ms=7776000000
                        )
                    else:
                        print("Only partition by date is implemented at the moment")

                result_dataset = client.create_dataset(f"{config['idProject']}.{config['dataset']}", exists_ok=True)
                print("Dataset created or ot already exists:", result_dataset)

                result = client.create_table(table_ref, exists_ok=True)
                print("Table created: ", result)

        return {
            table_config["tableId"]: table_config["tableFullName"]
            for *_, table_config in file_contents
        }

    return create_tables_op(
        env,
        pipeline_version,
        config,
        file_contents,
        github_config=github_config,
        timeout=timeout,
        wait_for_creation=wait_for_creation,
        run_local=run_local
    )