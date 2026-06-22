The data used in this project come from the **“Appliances Energy Prediction”** dataset, published by:

> **Candanedo, L. (2017).**  
> *Appliances Energy Prediction* [Dataset].  
> **UCI Machine Learning Repository.**  
> DOI: [10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G)

The dataset is publicly available at the **UCI Machine Learning Repository**:  
🔗 [https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

**License:**  
This dataset is licensed under a **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
This allows for **sharing** and **adapting** the dataset for any purpose, even commercially, provided that appropriate credit is given to the original author.

# Branchs
- main: É a versão atual.
- one-tree/0.1.1: É a versão utilizada no artigo FEXT-DP: An Approach for Differentially Private and Explainable Federated Learning.

# Before Run

### Clonar o scikit-learn modificado
Realizar o download do scikit-learn-dp, essa é a versão do scikit-learn modificada, que foi utilizada no projeto, irei adicionar posteriormente uma descrição do que foi alterado, mas por enquanto é possível visualizar somente pelos commits. Para clonar use:
```
https://github.com/Julio-C-Oliveira/scikit-learn-dp.git
```
Atualmente está na versão 1.9.dev0+dp

### Adicionar o path pro scikit-learn modificado no pyproject
Dentro do repositório do Fedt existe o arquivo pyproject.toml, vá para o arquivo, na sessão project, dentro da váriavel dependencies altere essa linha:
```
"scikit-learn @ file:///home/julio/documents/github/scikit-learn-dp",
```
Pegue o caminho absoluto para a sua pasta do scikit-learn-dp, e são 3 barras no inicio mesmo.

Para instalar todas as dependências do projeto use o seguinte comando na raiz do projeto:
```
pip install .
```
Caso você modifique algo, as vezes o pip não reconhece as modificações e utiliza os dados em cache, para contornar isso use:
```
pip install --no-cache-dir .
```

# How to Run

**Para rodar as simulações do FedT:**

- python -m fedt.simulation.run_server -s
- python -m fedt.simulation.run_clients -n 20 -i 0

**Para rodar os Ataques:**

- python -m fedt.side_tests.data_reconstruction_attack
- python -m fedt.side_tests.membership_inference_attack

**Para rodar o Shap:**

- python -m fedt.side_tests.shap

**Para rodar o teste de Ensemble:**

- python -m fedt.side_tests.ensemble_analysis

**Para gerar os gráficos:**

- python -m fedt.scripts_for_graphics.graphics

# Folders

## App
No pasta app reside a lógica do aprendizado federado.

## Simulation
Na pasta simulation reside as simulações.

## Service
Na pasta service estão os arquivos relacionados à interface de comunicação entre o servidor e os clientes.

No fedT.proto estão as definições de funções aceitas e dos tipos de mensagem.

Os arquivos .py são compilados a partir do fedT.proto.

## Side Tests
Aqui ficam os outros testes, como o ataque de reconstrução de label e o shap.

# Others



## Compile the proto file
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    --mypy_out=. \
    ./fedT.proto

## To-Do
- [x] Montar o teste com shap
- [x] Adicionar o ataque de inferência de pertencimento (MIA).
- [x] Adicionar um modo de avaliar como a perda de desempenho do modelo global é mitigada em decorrência do aumento do número de árvores no modelo global.

# Esquema de Commits
- feat: Pra adição de funcionalidade.
- fix: Pra correção de bugs.
- docs: Pra alterações de documentação.
- style: Mudança de formatação (espaços, vígulas e etc...) coisas que não afetam o código.
- refactor: Refatoração do código, sem adicionar novas funcionalidades ou corrigir bugs.
- test: Adição ou ajuste de testes.
- chore: Manutenção, atualização de dependências, configurações e etc...