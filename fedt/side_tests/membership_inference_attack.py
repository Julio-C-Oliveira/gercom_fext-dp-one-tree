import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from fedt.app.settings import paths, dataset, settings
from fedt.scripts.settings import graphics
from fedt.simulation.settings import simulation
from fedt.app.utils import load_house_client

def print_section(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

# =====================================================================
# ETAPA 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS REAIS (FEDT)
# =====================================================================
print_section("ETAPA 1: Carregando e Dividindo Dados Reais (Cliente vs Shadow)")

# Usando a primeira seed da simulação para extrair o dataset real
current_seed = simulation.seeds[0] 
client_data_divisor = 25 

X_train_raw, y_train_raw, _, _ = load_house_client(
    seed=current_seed, 
    alpha=settings.client.dirichlet_alpha, 
    bins=settings.client.number_of_bins_for_dirichlet,
    percentage_value_of_samples_per_client=dataset.percentage_value_of_samples_per_client / client_data_divisor
)

X_real = X_train_raw.to_numpy()
y_real = y_train_raw.to_numpy()

# Aplicação da Normalização MinMax exatamente como no seu segundo script
X_min = X_real.min(axis=0)
X_max = X_real.max(axis=0)
X_max[X_max == X_min] += 1e-8  # Evita divisão por zero
X_real_norm = (X_real - X_min) / (X_max - X_min)

print(f"Dataset Real Carregado (Seed {current_seed}) | Shape Total: {X_real_norm.shape}")

# Divisão Principal: 50% dos dados reais para o Cliente (Target), 50% para o Atacante (Shadow)
X_target, X_shadow, y_target, y_shadow = train_test_split(
    X_real_norm, y_real, test_size=0.5, random_state=current_seed
)

# Ambiente do Cliente: Membros (Treino) e Não-Membros (Teste)
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, test_size=0.5, random_state=current_seed
)

# Ambiente Sombra do Atacante: Dados para treinar o classificador do MIA
X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
    X_shadow, y_shadow, test_size=0.5, random_state=current_seed
)

print(f"Ambiente Cliente -> Treino (Membros): {X_target_train.shape[0]} | Teste (Não-Membros): {X_target_test.shape[0]}")
print(f"Ambiente Sombra  -> Treino (Membros): {X_shadow_train.shape[0]} | Teste (Não-Membros): {X_shadow_test.shape[0]}")

# =====================================================================
# ETAPA 2: TREINAMENTO DOS REGRESSORES (TARGET E SHADOW)
# =====================================================================
print_section("ETAPA 2: Treinamento dos Modelos de Árvore com Parâmetros Customizados")

epsilon_attack = -1.0

# Inicialização com os hiperparâmetros do seu fit_client
target_model = DecisionTreeRegressor(
    max_depth=settings.differential_privacy.tree_max_depth, 
    splitter="best", 
    random_state=current_seed
)
target_model.fit(
    X_target_train, y_target_train,
    global_max_target=1200,
    global_min_target=0,
    epsilon_global_budget=epsilon_attack,
    balancing_coefficient=0.37
)

shadow_model = DecisionTreeRegressor(
    max_depth=settings.differential_privacy.tree_max_depth, 
    splitter="best", 
    random_state=current_seed
)
shadow_model.fit(
    X_shadow_train, y_shadow_train,
    global_max_target=1200,
    global_min_target=0,
    epsilon_global_budget=epsilon_attack,
    balancing_coefficient=0.37
)

print(f"R² Target (Treino): {target_model.score(X_target_train, y_target_train):.2f}")
print(f"R² Target (Teste) : {target_model.score(X_target_test, y_target_test):.2f}")

# =====================================================================
# ETAPA 3: OBTENDO CARACTERÍSTICAS PARA REGRESSÃO
# =====================================================================
def extract_regression_attack_features(model, X, y_true):
    tree_ = model.tree_
    leaf_ids = model.apply(X)
    
    # 1. Profundidade do nó de decisão
    decision_paths = model.decision_path(X)
    depths = np.array(decision_paths.sum(axis=1)).flatten() - 1 
    
    # 2. Número de amostras na folha correspondente
    n_samples = tree_.n_node_samples[leaf_ids]
    
    # 3. Impureza da folha (MSE do Nó)
    node_mse = tree_.impurity[leaf_ids]
    
    # 4. Valor predito
    preds = model.predict(X)
    
    # 5. Erro Residual Absoluto (Métrica chave)
    residuals = np.abs(y_true - preds)
    
    return np.column_stack([depths, n_samples, node_mse, preds, residuals])

# =====================================================================
# ETAPA 4: TREINAMENTO DO MODELO ATACANTE
# =====================================================================
print_section("ETAPA 4: Treinando o Classificador do Atacante (MIA Model)")

attack_features_train = extract_regression_attack_features(shadow_model, X_shadow_train, y_shadow_train)
attack_labels_train = np.ones(len(X_shadow_train)) 

attack_features_test = extract_regression_attack_features(shadow_model, X_shadow_test, y_shadow_test)
attack_labels_test = np.zeros(len(X_shadow_test))  

Attack_X = np.vstack((attack_features_train, attack_features_test))
Attack_y = np.concatenate((attack_labels_train, attack_labels_test))

attack_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=current_seed)
attack_model.fit(Attack_X, Attack_y)
print("Modelo Atacante treinado nos dados da Shadow Tree.")

# =====================================================================
# ETAPA 5: EXECUÇÃO DO ATAQUE E AVALIAÇÃO
# =====================================================================
print_section("ETAPA 5: Executando o Ataque contra o Modelo Target (Dados do Cliente)")

eval_features_members = extract_regression_attack_features(target_model, X_target_train, y_target_train)
eval_labels_members = np.ones(len(X_target_train))

eval_features_non_members = extract_regression_attack_features(target_model, X_target_test, y_target_test)
eval_labels_non_members = np.zeros(len(X_target_test))

Eval_X = np.vstack((eval_features_members, eval_features_non_members))
Eval_y = np.concatenate((eval_labels_members, eval_labels_non_members))

mia_preds = attack_model.predict(Eval_X)
mia_probs = attack_model.predict_proba(Eval_X)[:, 1]

# Métricas de Sucesso
auc_roc = roc_auc_score(Eval_y, mia_probs)
accuracy = accuracy_score(Eval_y, mia_preds)

print(f"Acurácia do Ataque MIA: {accuracy * 100:.2f}%")
print(f"AUC-ROC do Ataque:      {auc_roc:.4f}\n")

print("Relatório de Classificação:")
print(classification_report(Eval_y, mia_preds, digits=4))

feature_names = ["Profundidade do Nó", "Num. Amostras", "Impureza do Nó (MSE)", "Valor Predito", "Erro Residual Absoluto"]
importances = attack_model.feature_importances_
print("\nImportância das Features no Dataset Real:")
for name, imp in zip(feature_names, importances):
    print(f"- {name}: {imp*100:.1f}%")