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

# Before Run

### Clonar o scikit-learn modificado
Realizar o download do scikit-learn-dp, essa é a versão do scikit-learn modificada, que foi utilizada no projeto, irei adicionar posteriormente uma descrição do que foi alterado, mas por enquanto é possível visualizar somente pelos commits. Para clonar use:
```
https://github.com/Julio-C-Oliveira/scikit-learn-dp.git
```
Atualmente está na versão 1.9.dev0+dp

### Adicionar o path pro scikit-learn modificado no pyproject
Dentro do repositŕoio do Fedt existe o arquivo pyproject.toml, vá para o arquivo, na sessão project, dentro da váriavel dependencies altere essa linha:
```
"scikit-learn @ file:///home/julio/documents/github/scikit-learn-dp",
```
Pegue o caminho absoluto para a sua pasta do scikit-learn-dp, e são 3 barras no inicio mesmo.

# How to Run



# Folders



# Others



## Compile the proto file
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    --mypy_out=. \
    ./fedT.proto