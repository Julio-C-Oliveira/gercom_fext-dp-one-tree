import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Importa a função modificada do seu arquivo utils.py 
from fedt.app.utils import load_house_client
from fedt.app.utils import load_dataset

# Configurações do experimento
num_clients = 10
alpha_value = 0.1  # Altere este valor para testar o nível de Non-IID! (ex: 0.1, 0.5, 5.0)
base_seed = 42
bins = 100

all_clients_data = []

print("Coletando e processando os dados dos 10 clientes...")

for client_id in range(num_clients):
    # Simulando a criação do seed único por cliente [cite: 27]
    seed = base_seed + client_id
    
    # Carrega os dados do cliente 
    # (Certifique-se de que a sua função load_house_client no utils.py aceita o parâmetro alpha)
    X_train, y_train, _, _ = load_house_client(seed=seed, alpha=alpha_value, bins=bins)
    
    # Armazena o alvo (Appliances) marcando a quem ele pertence
    df_client = pd.DataFrame({
        'Appliances': y_train,
        'Cliente': f'Cliente {client_id}'
    })
    
    all_clients_data.append(df_client)

# Concatena os dados de todos os clientes em um único DataFrame para o Seaborn
df_total = pd.concat(all_clients_data, ignore_index=True)

# --- CONFIGURAÇÃO DO GRÁFICO ---
plt.figure(figsize=(14, 6))
sns.set_theme(style="whitegrid")

# O Violin Plot mostra a distribuição de densidade de cada cliente de forma comparativa
sns.violinplot(
    x='Cliente', 
    y='Appliances', 
    data=df_total, 
    palette='viridis', 
    hue='Cliente', 
    legend=False
)

# Customização visual
plt.title(f'Distribuição Non-IID de "Appliances" entre 10 Clientes (Dirichlet $\\alpha$ = {alpha_value})', fontsize=16, fontweight='bold')
plt.xlabel('Clientes (Nós Federados)', fontsize=13)
plt.ylabel('Consumo de Energia (Appliances)', fontsize=13)
plt.xticks(rotation=30)
plt.tight_layout()

# Exibe o gráfico
plt.savefig('distribuicao_non_iid.pdf', dpi=300, bbox_inches='tight')

#########################################

# 1. Carrega os dados brutos
X, y = load_dataset()

# 2. Configura o gráfico
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 3. Histograma com 10 bins e a curva de densidade (KDE)
# O parâmetro bins=10 define a quantidade de barras
sns.histplot(y, bins=100, kde=True, color='skyblue', stat="density")

plt.title('Distribuição Original (Global) de "Appliances"', fontsize=16, fontweight='bold')
plt.xlabel('Consumo de Energia (Appliances)', fontsize=13)
plt.ylabel('Densidade', fontsize=13)

# 4. Salva com alta resolução
plt.savefig('distribuicao_original_global.pdf', dpi=300, bbox_inches='tight')