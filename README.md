# Nome do Projeto

Adicione aqui a descrição do projeto.

## Qual o objetivo deste repositório?

## Organização deste repositório

```
produtos_digitais_template
├── models
│   └── exemplo.joblib
├── notebooks
│   └── exemple.ipynb
├── src
│   ├── models
│   │   └── __init__.py
│   │   └── predict.py
│   │   └── train.py
│   ├── queries
│   │   └── train_test_split.sql
│   ├── util
|   |   └── __init__.py
│   │   └── create_tables.py
│   └── __init__.py
|   └── pipeline.py
├── tables
|   ├── destination
|       └── table-predicted.yaml
|   ├── workspace
|       └── dataset-prepared.yaml
├── README.md
├── project.yaml
├── requirements.txt
```

Onde

- `./models` contém os arquivos binários de modelos (apenas para desenvolvimento local)
- `./notebooks` contém os notebooks de experimentação (assim como exemplos)
- `./src` contém os módulos python da aplicação
- `./src/models` contém os módulos de treinamento e scoring (`train.py` e `predict.py`)
- `./src/queries` contém os arquivos .sql do projeto.
- `./src/utils` contém as scripts utilitarios.
- `./tables` 
- `./project.yaml` contém configuração do projeto.(Variáveis de dev e prod)
- `./requirements.txt` Libs requeridas para o projeto.

## Setup
Se necessário, adicione aqui os passos para fazer o setup do ambiente, instalação do projeto e dependências

## CI/CD
Link da ferramenta de CI e CD
