import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def print_section(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

# =====================================================================
# ETAPA 1: PREPARAÇÃO DOS DADOS (REGRESSÃO)
# =====================================================================
print_section("ETAPA 1: Gerando Ambientes do Cliente e do Atacante (Regressão)")

# Gerando um dataset contínuo (Regressão)
X, y = make_regression(n_samples=4000, n_features=5, noise=0.5, random_state=42)

# Divisão Principal: Metade para o Cliente (Target), Metade para o Atacante (Shadow)
X_target, X_shadow, y_target, y_shadow = train_test_split(X, y, test_size=0.5, random_state=42)

# Ambiente do Cliente: Membros (Treino) e Não-Membros (Teste)
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, test_size=0.5, random_state=42)

# Ambiente Sombra do Atacante: Dados para treinar o MIA
X_shadow_train, X_shadow_test, y_shadow_train, y_shadow_test = train_test_split(
    X_shadow, y_shadow, test_size=0.5, random_state=42)

print(f"Ambiente Cliente -> Treino (Membros): {X_target_train.shape[0]} | Teste (Não-Membros): {X_target_test.shape[0]}")
print(f"Ambiente Sombra  -> Treino (Membros): {X_shadow_train.shape[0]} | Teste (Não-Membros): {X_shadow_test.shape[0]}")

# =====================================================================
# ETAPA 2: O MODELO ALVO E O MODELO SOMBRA (REGRESSORES)
# =====================================================================
print_section("ETAPA 2: Treinamento dos Modelos de Árvore (Target e Shadow)")

# Treinando o modelo de Regressão do Cliente. 
# Permitimos crescimento profundo para induzir a memorização (overfitting).
target_model = DecisionTreeRegressor(random_state=42)
target_model.fit(
    X_target_train, y_target_train,
    global_max_target=1200,
    global_min_target=0,
    epsilon_global_budget=-1.0,
    balancing_coefficient=-1.0
)

# O atacante treina um Regressor análogo no seu ambiente sombra.
shadow_model = DecisionTreeRegressor(random_state=42)
shadow_model.fit(
    X_shadow_train, y_shadow_train,
    global_max_target=1200,
    global_min_target=0,
    epsilon_global_budget=-1.0,
    balancing_coefficient=-1.0
)

# Calculando o R² (Coeficiente de Determinação) para ver o overfitting
print(f"R² Target (Treino): {target_model.score(X_target_train, y_target_train):.2f} (Overfitting perfeito)")
print(f"R² Target (Teste): {target_model.score(X_target_test, y_target_test):.2f}")

# =====================================================================
# ETAPA 3: ENGENHARIA DE CARACTERÍSTICAS PARA REGRESSÃO
# =====================================================================
def extract_regression_attack_features(model, X, y_true):
    """
    Extrai métricas estruturais e de erro de um modelo de REGRESSÃO.
    """
    tree_ = model.tree_
    leaf_ids = model.apply(X)
    
    # 1. Profundidade do nó
    decision_paths = model.decision_path(X)
    depths = np.array(decision_paths.sum(axis=1)).flatten() - 1 
    
    # 2. Número de amostras na folha
    n_samples = tree_.n_node_samples[leaf_ids]
    
    # 3. Impureza da folha (Em regressão, o Scikit-Learn usa o MSE (Erro Quadrático Médio) do nó)
    node_mse = tree_.impurity[leaf_ids]
    
    # 4. Valor predito (A média Y da folha)
    preds = model.predict(X)
    
    # 5. ERRO RESIDUAL ABSOLUTO (Feature crucial para Regressão)
    # Membros do treino terão erro próximo a 0.0. Não-membros terão erro maior.
    residuals = np.abs(y_true - preds)
    
    features = np.column_stack([depths, n_samples, node_mse, preds, residuals])
    return features

# =====================================================================
# ETAPA 4: TREINAMENTO DO MODELO ATACANTE (MIA CLASSIFIER)
# =====================================================================
print_section("ETAPA 4: Treinando o Classificador do Atacante (MIA Model)")

# Construindo o Dataset do Atacante usando o Regressor Sombra
attack_features_train = extract_regression_attack_features(shadow_model, X_shadow_train, y_shadow_train)
attack_labels_train = np.ones(len(X_shadow_train)) # 1 = Membro

attack_features_test = extract_regression_attack_features(shadow_model, X_shadow_test, y_shadow_test)
attack_labels_test = np.zeros(len(X_shadow_test))  # 0 = Não-Membro

Attack_X = np.vstack((attack_features_train, attack_features_test))
Attack_y = np.concatenate((attack_labels_train, attack_labels_test))

# O modelo ATACANTE é SEMPRE um Classificador (prever 0 ou 1)
attack_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
attack_model.fit(Attack_X, Attack_y)
print("Modelo Atacante treinado com sucesso nos vetores de regressão.")

# =====================================================================
# ETAPA 5: EXECUÇÃO DO ATAQUE E AVALIAÇÃO
# =====================================================================
print_section("ETAPA 5: Executando o Ataque contra o Modelo Alvo (Cliente)")

# Vetores dos Membros e Não-Membros REAIS passando pelo Regressor Target
eval_features_members = extract_regression_attack_features(target_model, X_target_train, y_target_train)
eval_labels_members = np.ones(len(X_target_train))

eval_features_non_members = extract_regression_attack_features(target_model, X_target_test, y_target_test)
eval_labels_non_members = np.zeros(len(X_target_test))

Eval_X = np.vstack((eval_features_members, eval_features_non_members))
Eval_y = np.concatenate((eval_labels_members, eval_labels_non_members))

# Predições de Pertencimento
mia_preds = attack_model.predict(Eval_X)
mia_probs = attack_model.predict_proba(Eval_X)[:, 1]

# Resultados
auc_roc = roc_auc_score(Eval_y, mia_probs)
accuracy = accuracy_score(Eval_y, mia_preds)

print(f"Acurácia do Ataque MIA: {accuracy * 100:.2f}%")
print(f"AUC-ROC do Ataque:      {auc_roc:.4f}\n")

print("Relatório de Classificação do Ataque (1 = Membro, 0 = Não-Membro):")
print(classification_report(Eval_y, mia_preds, digits=4))

# Importância das Features para o Atacante em Regressão
feature_names = ["Profundidade do Nó", "Num. Amostras", "Impureza do Nó (MSE)", "Valor Predito", "Erro Residual Absoluto"]
importances = attack_model.feature_importances_
print("\nImportância das Features (O que denuncia a regressão?):")
for name, imp in zip(feature_names, importances):
    print(f"- {name}: {imp*100:.1f}%")