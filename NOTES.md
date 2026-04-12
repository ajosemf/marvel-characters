# MLOps with Databricks Free Edition

## References
* Curso [MLOps with Databricks Free Edition](https://www.youtube.com/playlist?list=PL_MIDuPM12MOcQQjnLDtWCCCuf1Cv-nWL), no YouTube.

## Informações adicionais
Data de publicação no YouTube: 29/08/2025

O curso é uma playlist de 10 vídeos, com duração total de aproximadamente 3 horas.

O curso se baseia no desenvolvimento de um modelo LightGBM para prever se um personagem da Marvel irá sobreviver ou não.

## Requisitos
* [Instalação do databricks-cli](https://docs.databricks.com/aws/en/dev-tools/cli/install#curl-installation-for-linux-macos-and-windows)
* [Gerenciador de pacotes UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
* Instalação e configuração do Databricks extension no VS Code
    * Extension ID: `databricks.databricks`

## Forked repository
O código fonte deste curso é um forked do repositório original, disponível em: https://github.com/marvelousmlops/marvel-characters

O repositório no meu GitHub é: https://github.com/ajosemf/marvel-characters

## Principais arquivos do repositório
* `project_config_marvel.yml`: arquivo de configuração do projeto. Define
    * Envs de produção, acceptance e desenvolvimento
    * Parâmetros de treinamento do modelo
    * Features numéricas e categóricas
    * Target
* `databricks.yml`: arquivo de configuração do Databricks Asset Bundles. Define os workspaces, root paths e variáveis para os targets de desenvolvimento, acceptance e produção.
* `pyproject.toml`: arquivo de configuração do projeto Python. Define as dependências do projeto, bem como as dependências opcionais conforme o target (dev, acc, prd), além da qualidade do código por meio da configuração da ferramenta `ruff` na seção `[tool.ruff]`.
* `src/marvel_characters/config.py`: módulo de configuração do projeto. Define a classe `ProjectConfig`, que implementa o método `from_yaml` para carregar a configuração do projeto a partir do arquivo `project_config_marvel.yml`.

## Description
Databricks recently introduced Free Edition, which opened the door for us to create a free hands-on course on MLOps with Databricks.

In this course series, we walk through the tools, patterns, and best practices for building and deploying machine learning workflows on Databricks:

Lecture 1: Introduction to MLOPs
Lecture 2: Developing on Databricks
Lecture 3: Getting started with MLflow
Lecture 4: Log and register model with MLflow
Lecture 5: Model serving architectures
Lecture 6: Deploying model serving endpoint
Lecture 7: Databricks Asset Bundles
Lecture 8: CI/CD and deployment strategies
Lecture 9: Intro to monitoring
Lecture 10: Lakehouse monitoring

# Setup do Projeto

## Setup Inicial no VSCode
No VSCode, abrir um terminal integrado na raiz do projeto.

No terminal, executar o login, substituindo o [PROFILE_NAME] pelo nome que aparece na URL do seu workspace Databricks. Após o comando de login, tecle Enter para confirmar o nome do profile e na janela do navegador que abrir, faça login com a sua conta Databricks.
```bash
    databricks auth login --host https://[PROFILE_NAME].cloud.databricks.com
    ✔ Databricks profile name [PROFILE_NAME]: █
    Profile [PROFILE_NAME] was successfully saved
```

Clicar no ícone do Databricks na barra lateral esquerda do VS Code. Em `Configuration > Local Folder` você verá `"Invalid host for target dev"`. Será necessário configurar o arquivo `databricks.yml`, que informa à extensão do Databricks qual host deve ser usado para diferentes `targets` (dev, acc, prd). Configure os hosts com `https://[PROFILE_NAME].cloud.databricks.com`, substituindo o [PROFILE_NAME] pelo nome do seu profile. Se for solicitado o login, use OAuth e faça login com a sua conta Databricks. Você pode checar a configuração clicando no ícone de engrenagem ao lado de `Target > Auth Type` e no menu suspenso, clicar em `Edit Databricks profiles`. Você verá o conteúdo do arquivo `.databrickscfg` criado em seu diretório home. Este arquivo é onde o comando `databricks auth login` salva as informações de autenticação.

Criação do ambiente virtual com UV
* O arquivo `pyproject.toml` define as dependências do projeto, bem como as dependências opcionais conforme o target (dev, acc, prd), além da qualidade do código por meio da configuração da ferramenta `ruff` na seção `[tool.ruff]`. `Ruff` é uma ferramenta de linting e formatação de código Python, que ajuda a manter a qualidade do código, identificando erros, problemas de estilo e aplicando formatação automática. Ele é configurado no arquivo `pyproject.toml` para garantir que o código do projeto siga as melhores práticas e padrões de estilo.
* Na raiz do projeto, executar:
```bash
    uv sync --extra dev
```
* Após a instalação das dependências, configurar o ambiente correto na extensão do Databricks no VS Code. Clicar no ícone do Databricks na barra lateral esquerda, em `Configuration > Python Environment`, clicar no ícone de engrenagem ao lado de `Activate Environment` e selecionar o ambiente virtual criado pelo UV (`3.12.3 marvel-characters`).

Em `Configuration > Python Environment`, clicar em `Attach a cluster` e selecionar o cluster `Serverless`, o único disponível na Free Edition.

Saia do terminal integrado e abra um novo terminal integrado para garantir que o ambiente virtual esteja ativado (marvel-characters).

## Configurando e executando no Databricks
* No navegador, acesse o seu Databricks workspace e certifique-se de estar em `Workspace > Users > [seu email]`. No canto superior clique em `Create > Git Folder`. Em `Git repository URL` cole a URL do repositório do projeto (`https://github.com/seu-usuario/marvel-characters`) e clique em `Create`. Você pode obter a URL do repositório no GitHub clicando no botão `Code` e copiando a URL HTTPS e apenas remover o `.git` do final da URL.

No navegador, no Databricks workspace, abra o notebook `/Workspace/Users/ajosemf@gmail.com/marvel-characters/notebooks/lecture2.marvel_data_preprocessing`. No painel lateral direito, clique em `Environment` e certifique-se de que o `Base environment` utilize-se da versão 3 (Operating System: Ubuntu 24.04.2 LTS; Python: 3.12.3; Release date: June 13, 2025; End-of-support data: June 13, 2028). Você pode ver informações detalhadas sobre as versões de ambiente serverless disponíveis na Free Edition em https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/ . Em seguida clique em `Apply` para aplicar o ambiente ao notebook. Se aparecer a mensagem `Environment configurations are not saved Environment configurations are not persisted in source-format notebooks. Enable Include in exports in the environment panel to preserve your environment.`, clique em `Include in exports` para salvar a configuração do ambiente no notebook.

Executar a primeira célula do notebook que instala as dependências do projeto. Ignore a mensagem `Core Python package version(s) changed ...`. Em seguida, é necessário descomentar e executar o código da célula 2, que ajusta o system path. Isso evita o erro na célula 3 `ModuleNotFoundError: No module named 'marvel_characters'`. Estas duas primeiras células não são recomendadas em um ambiente de produção e só existem nesse momento para fins didáticos. A forma correta é apresentada nas seções [Gerando um pacote wheel do projeto](#gerando-um-pacote-wheel-do-projeto) e [Usando o pacote wheel no Databricks](#usando-o-pacote-wheel-no-databricks).

## Executando no VSCode
Abrir o notebook `notebooks/lecture2.marvel_data_preprocessing` no VS Code. Repare que o arquivo não é do tipo `.ipynb`, mas sim um arquivo de código Python com marcações de células. Essas marcações são usadas pela extensão do Databricks para identificar as células do notebook. As marcações de células podem ser `# COMMAND ----------`, `# Databricks notebook source` ou `# %%` (entre outras). Para configurar a extensão do Databricks para reconhecer essas marcações, é necessário ajustar a configuração `jupyter.interactiveWindow.cellMarker.codeRegex` no arquivo `.vscode/settings.json`. Após configurar a extensão, é possível executar as células do notebook diretamente no VS Code, usando os comandos de execução de células do Jupyter.

No VSCode, procure pela célula abaixo, que é a célula 3 do notebook, e execute-a. Acima dessa célula aparecem os comandos clicáveis `Run Cell | Run Above ...`. Clique em `Run Cell`. O VSCode irá se conectar ao Databricks e processar o código usando o Serverless configurado. O resultado será exibido em um painel à direita do código, onde é possível ver os logs de execução e o output do código. Se a célula for executada com sucesso, você verá a mensagem `Configuration loaded:` seguida da configuração do projeto em formato YAML.
```bash
# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
from pyspark.sql import SparkSession

from marvel_characters.config import ProjectConfig
from marvel_characters.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config_marvel.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
```

Todas as demais células do notebook poderão ser executadas da mesma forma. Como resultado final, espera-se que os tables `mlops_dev.marvel_characters.train_set` e `mlops_dev.marvel_characters.test_set` sejam criados no catálogo do Databricks.


# Anotações

## Monitoramento de modelos
O monitoramento pode ser executado do ponto de vista de "data drift" ou "model drift". No entanto, é mais comum que o monitoramento do modelo seja customizado conforme as necessidades do negócio.

## Desvantagens do desenvolvimento baseado em notebooks
* Dificulta a implementação de código modular e reutilizável.
* Dificulta a implementação de testes automatizados.
* Reduz a qualidade do código.

## Databricks Connect
* O Databricks Connect é uma ferramenta que permite que você se conecte a um cluster do Databricks a partir do seu ambiente de desenvolvimento local, como o VS Code. Ele permite que você execute código Python localmente, mas que seja processado no cluster do Databricks, o que é útil para desenvolvimento e depuração de código.
* É possível checar no terminal integrado do VS Code que o pyspark não está instalado no ambiente virtual local. Com o ambiente virtual ativado (marvel-characters), execute `uv pip freeze` e veja que o pyspark não está listado. 
* Exatamente por isso, não se deve instalar o pyspark localmente, pois isso pode causar conflitos de versão com o Databricks Connect, que já inclui uma versão do pyspark compatível com o cluster do Databricks.
* O Databricks Connect é uma dependência que está listada no arquivo `pyproject.toml` na seção abaixo:
```toml
[project.optional-dependencies]
dev = ["databricks-connect>=16.0, <17",
       "ipykernel>=6.29.5, <7",
       "pip>=25.0.1, <26",
       "pre-commit>=4.1.0, <5"]
```

## Sincronizando VSCode com Databricks
* Na extensão do Databricks no VS Code, em `Configuration > Remote Folder`, clicar no ícone de `sync` para sincronizar os arquivos do projeto com o Databricks. Isso irá criar uma pasta no Databricks com o nome do projeto, onde os arquivos do projeto serão sincronizados. O caminho da pasta no Databricks será algo como `/Workspace/User/[seu email]/.bundle/dev/marvel-characters/files`, onde `dev` é o target de desenvolvimento configurado no arquivo `databricks.yml`. Após a sincronização, os arquivos do projeto estarão disponíveis no Databricks e poderão ser acessados e executados a partir do Databricks.
* A sincronização é especialmente importante quando há necessidade de executar notebooks a partir do Databricks, uma vez que algums comandos não poder ser executados a partir do VS Code, como por exemplo:
    * Comandos de delta tables, como `spark.sql("CREATE TABLE ...")`
    * Feature engineering packages

## Gerando um pacote wheel do projeto
* No terminal integrado do VS Code, com o ambiente virtual ativado, executar o comando `uv build` para gerar um pacote wheel do projeto. O pacote wheel é um formato de distribuição de pacotes Python que pode ser facilmente instalado em outros ambientes. O comando `uv build` irá criar um arquivo `.whl` na pasta `dist` do projeto (que também será criada). Este arquivo permite a instalação do projeto em outros ambientes.
* O diretório `dist` não é versionado.

## Usando o pacote wheel no Databricks
* Uma vez que o pacote wheel do projeto foi gerado, é possível instalá-lo no Databricks usando o comando `%pip install [caminho do arquivo .whl]`.
* Para fins didáticos, no VSCode, copie o arquivo whl em dist para a pasta `notebooks`. Em um ambiente de produção, o ideal seria publicar o pacote em um repositório de pacotes privado, como o Azure Artifacts ou o AWS CodeArtifact, e instalar o pacote a partir do repositório usando o comando `%pip install --index-url [URL do repositório] [nome do pacote]`.
* No navegador, no Databricks workspace, as células abaixo
```python
%pip install -e ..
%restart_python
```
```python
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))
```
Agora podem ser substituídas por uma única célula:
```python
%pip install [caminho do arquivo .whl]
%restart_python
```
