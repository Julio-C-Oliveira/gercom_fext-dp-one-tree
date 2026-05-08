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



# Folders



# Others



## Compile the proto file
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    --mypy_out=. \
    ./fedT.proto

## To-Do
- [ ] Ajustar os colaterais de modificar a função rpc aggregate_trees, Client_Tree é uma mensagem única ao invés de um stream agora.
    - [x] Ajustar os imports dos arquivos gerados pelo proto.
    - [x] Ajustar o tratamento da mensagem do cliente no servidor.
        - [x] Retirar a função get_number_of_trees_per_client, não é mais necessária, agora cada cliente treina somente uma árvore. Medida paliativa: trees_by_client=1.
        - [x] Alterar o argumento request_iterator do aggregate_trees no server para request. 
        - [x] Alterar o tratamento do request.
    - [x] Ajustar a get_server_settings, ela vai ser responsável por passar os parâmetros de privacidade.
    - [x] Ajustar os imports do server
    - [ ] Ajustar o Cliente
        - [x] Imports
        - [x] Modo de envio da árvore para o servidor. De iterator, para request único, no aggregate trees.
        - [x] Alterar HouseClient, para Client.
        - [x] Ajustar o get_server_settings no cliente.
        - [x] Substituir as florestas por árvores.
        - [x] Remover o self.trees
        - [x] O evaluate foi substituido por choose_model, ajustar o cliente.
        - [x] O return do choose também é diferente do evaluate.
    - [ ] Escrever o simulation.toml
- [ ] Adicionar reprodutibilidade na geração dos datasets para os clientes. 
- [ ] Analisar a possibilidade de remover a inicialização de um modelo de floresta do lado do servidor.  

- [ ] Remover as funções mortas
    - [x] Retirar o gerar_funcao_logaritmica do utils, se o server não vai utilizar mais ela se tornou inútil.

# Esquema de Commits
- feat: Pra adição de funcionalidade.
- fix: Pra correção de bugs.
- docs: Pra alterações de documentação.
- style: Mudança de formatação (espaços, vígulas e etc...) coisas que não afetam o código.
- refactor: Refatoração do código, sem adicionar novas funcionalidades ou corrigir bugs.
- test: Adição ou ajuste de testes.
- chore: Manutenção, atualização de dependências, configurações e etc...