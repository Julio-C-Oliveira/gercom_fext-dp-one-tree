# Revisão Sistemática

## Questões de Pesquisa
- **RQ - 01:** Quais são as arquiteturas de árvore de Decisão Federadas (horizontais, verticais ou hibridas) que atualmente aplicam Privacidade Diferencial?
- **RQ - 02:** Como a Privacidade Diferencial é aplicada nesses modelos?
    - Local, no modelo do cliente:
        - Como a distribuição do budget é realizada?
        - Qual mecanismo é utilizado nos nós de split para aplicar privacidade diferencial?
        - Qual mecanismo é utilizado nos nós folha para aplicar privacidade diferencial?
    - Local, nos dados antes de treinar.
    - Global, no servidor.
   
- **RQ - 03:** Quais são as métricas de trade-off entre utilidade e privacidade utilizadas?
- **RQ - 04:** Quais são as lacunas abertas? 
- **RQ - 05:** Como a aplicação de privacidade diferencial em modelos baseados em árvores afeta os métodos de interpretabilidade, como o SHAP?
- **RQ - 06:** Quais mecanismos de agregação baseados em validação no servidor são utilizados em árvores de decisão federadas? 
- **RQ - 07:** Qual é a eficiência da aplicação de DP local no modelo de árvore contra ataques de reconstrução dos dados e inferência de pertencimento?


## String de Busca
- Explicabilidade + DP em Árvores:
    ``` 
    ("Federated Learning") AND ("Regression Tree" OR "Decision Tree" OR "Random Forest" OR "Regression Forest" OR "GBDT" OR "XGBoost") AND ("Differential Privacy") AND ("SHAP" OR "Shapley" OR "Explainable AI" OR "XAI" OR "LIME" OR "MDI")
    ```
- Ataques + Árvores Privadas
    ```
    ("Federated Learning") AND ("Regression Tree" OR "Decision Tree" OR "Random Forest" OR "Regression Forest" OR "GBDT" OR "XGBoost") AND ("Differential Privacy") AND ("Membership Inference" OR "Label Reconstruction" OR "Inversion Attack" OR "Inference Attack")
    ```

**Caso poucos resultados sejam encontrados:**
- Aprendizado Federado + Árvores de Decisão:
    ```
    ("Federated Learning" OR "Distributed Learning") AND ("Decision Tree" OR "Random Forest" OR "XGBoost" OR "Gradient Boosting" OR "GBDT" OR "Tree-based")
    ```
- Privacidade Diferencial + Árvores de Decisão:
    ```
    ("Differential Privacy" OR "DP") AND ("Decision Tree" OR "Random Forest" OR "XGBoost" OR "Gradient Boosting" OR "GBDT" OR "Tree-based")
    ```


## Critérios de Inclusão e Exclusão

| Critérios de Inclusão | Critérios de Exclusão |
| :-------------------: | :-------------------: |
Artigos que propõem ou avaliam árvores de decisão no contexto de aprendizado federado com DP. | Artigos que usam aprendizado federado e DP, porém com redes neurais.
Estudos que analisam empiricamente ou teoricamente o trade-off de utilidade/privacidade em estruturas de árvores. | Artigos que tratam de privacidade em árvores federadas usando apenas criptografia (SMC/HE) sem aplicar DP.
| | Artigos de DP aplicados a árvores de decisão, mas em ambientes centralizados (não-federados).

## Extração de Dados
Vou ler o titulo e o abstract dos artigos que as strings de busca retornarem, para os que passarem vou ler o texto e montar uma tabela com as seguintes informações:
- Autor e Ano.
- Tipo de Aprendizado Federado: (Horizontal/Vertical).
- Modelo de Base: (XGBoost, Decision Tree ou Random Forest).
- Mecanismo de DP: (Laplace, Gaussiano, Exponencial, EDP, Local ou Central).
- Pontos de Inserção da DP: (Nos dados de treino, nos nós de split ou nos nós folha).
- Mecanismo de Agregação.
- Avaliação de Explicabilidade: (SHAP, LIME ou MDI).
- Ataques de Privacidade Avaliados: (Nenhum, inferência de pertencimento ou ataque de recontrução de label).
- Limitações Apontadas.

## Retorno da Busca
```
("Federated Learning") AND ("Regression Tree" OR "Decision Tree" OR "Random Forest" OR "Regression Forest" OR "GBDT" OR "XGBoost") AND ("Differential Privacy") AND ("SHAP" OR "Shapley" OR "Explainable AI" OR "XAI" OR "LIME" OR "MDI")
```
- **Explainable AI and Federated Learning for Privacy-Preserving Loan Default Prediction in Cooperative Banking Networks:**
    - IEEE Xplorer.
    - 2026 IEEE International Conference for Convergence in Computing Technology (I3CTCON).
    - Shilpa Shivshankar Mathpati; Abhijit Chirputkar.
    - XAI + Federated Learning + Random Forest + DP + Inference Attack.
    - Relação com o nosso: 5.
- **FLAME-X: Federated Learning with Adaptive Model Ensembles and Explainable AI for Real-Time DDoS Detection in IoT Networks:**
    - IEEE Xplorer.
    - 2025 IEEE Guwahati Subsection Conference (GCON).
    - Sanjoy Kumar Ghimire; Rakesh Matam; Ferdous Ahmed Barbhuiya.
    - XAI + Federated Learning + Rede Neural (Apenas cita Decision Tree) + DP.
    - Relação com o nosso: 0.
- **Privacy-Preserving and Explainable AI in Credit Card Fraud Detection: Balancing Accuracy, Transparency and Security:**
    - IEEE Xplorer.
    - 2025 2nd International Conference on Intelligent Systems for Cybersecurity (ISCS).
    - Lakshya Sharma; Ayush Kaushik; Mridul Sharma; Shilpa Gupta.
    - XAI + Federated Learning + Rede Neural (Random Forest apenas para pré processamento) + DP.
    - Relação com o nosso: 0.
- **Novel Federated Learning Approach for Dengue Predictive Models:**
    - IEEE Xplorer.
    - 2025 3rd International Conference on Business Analytics for Technology and Security (ICBATS).
    - Sirajul Muneer Abbas; Kushan Bhareti.
    - XAI (Shap e LIME) + Federated Learning + Não especifica (cita modelos baseados em árvore, usou Flower, suponho que seja rede neural) + DP.
    - Relação com o nosso: 0.
- **Healthcare Security using Federated Learning and Explainable AI with Secure Aggregation:**
    - IEEE Xplorer.
    - 2025 International Conference on Sustainable Communication Networks and Application (ICSCN).
    - Mukhila R; Sundhari M; Nandeta S; S. Pavithra.
    - XAI (TreeSHAP) + Federated Learning + Random Forest + DP (Porém o DP é aplicado em uma rede neural que pré processa os dados).
    - Relação com o nosso: 3.
- **Trustworthy Multimodal Fraud Detection with Federated Learning and Computer Vision:**
    - IEEE Xplorer.
    - 2025 International Conference on Artificial Intelligence, Blockchain, Cloud Computing, and Data Analytics (ICoABCD).
    - Dendy K Pramudito; Jufriadif Na'am; Ferda Ernawan.
    - XAI + Federated Learning (Aplicado a rede neural) + XGBoost e CNN + DP (Aplicada à CNN). 
    - Relação com o nosso: 2.
- **Federated Learning for Privacy-Preserving Employee Performance Analytics:**
    - IEEE Xplorer.
    - IEEE Access ( Volume: 13) - 21 July 2025.
    - Jay Barach.
    - XAI + Federated Learning (Rede Neural) + Rede Neural + DP (na Rede Neural).
    - Relação com o nosso: 0.
- **Federated and Fairness-Aware Learning for Rural Healthcare Risk Prediction Under Data Scarcity:**
    - IEEE Xplorer.
    - 2026 IEEE Bangalore Humanitarian Technology Conference (B-HTC).
    - Dhruti A; Saksham Garg; Panchadip Bhattacharjee; Somyajeet Arukh; Nishanth Shet; Gururaj H L.
    - XAI + Federated Learning + XGBoost e LighGBM + DP (Laplace aplicado nos Gradientes).
    - Relação com o nosso: 4.
- **Fairfedtransnet: Fairness-Aware Transformer-Based Federated Learning for Multimodal Rare Disease Diagnosis:**
    - IEEE Xplorer.
    - 2026 5th International Conference on Communication, Computing and Electronics Systems (ICCCES).
    - Bhavana Jamalpur; KENNEDY JEEVARATHINAM; K. Vetrivel; S.SESHA VIDHYA; Ashok Murugesan; B. Jegajothi.
    - XAI + Federated Learning (Rede Neural) + Rede Neural (Apenas cita modelos Tree based) + DP (Rede Neural).
    - Relação com o nosso: 0.
- **Cross-Silo Prediction of Hospital Readmissions using a Federated Learning Framework:**
    - IEEE Xplorer.
    - 2025 5th International Conference on Evolutionary Computing and Mobile Sustainable Networks (ICECMSN).
    - M.Ayyadurai; Mythreiy Anand; Eniyan P.
    - XAI (SHAP e LIME) + Federated Learning (FedAVG adaptado) + XGBoost + DP (Gaussiano nos gradientes).
    - Relação com o nosso: 4.
- **Federated Learning for Privacy-Preserving Diabetes Detection: A Robust Framework for Heterogeneous Healthcare Data:**
    - IEEE Xplorer.
    - 2025 7th International Conference on Innovative Data Communication Technologies and Application (ICIDCA).
    - C Dhiya; Umesh Gjh; U Kavya; G Sivashankar; S Nagendra Prabhu.
    - XAI + Federated Learning (Rede Neural) + Rede Neural + DP (Aplicado aos gradientes).
    - Relação com o nosso: 0.
- **Enhancing Data Privacy in Multi-Institutional Medical AI: A Secure Vertical Federated Learning Framework:**
    - IEEE Xplorer.
    - 2025 Third International Conference on Industry 4.0 Technology (I4Tech).
    - Samruddhi Prabhune; Balaso Jagdale.
    - XAI (cita como trabalho futuro) + Federated Learning (Vertical) + GBDT + DP (Laplace nos dados).
    - Relação com o nosso: 3.
- **EQAI: Explainable Quantum-Empowered Antispoofing Intelligence for Trustworthy Connected Autonomous Vehicles Communication:**
    - IEEE Xplorer.
    - IEEE Internet of Things Journal ( Volume: 13, Issue: 6, 15 March 2026) - 28 November 2025.
    - Simeon Okechukwu Ajakwe; Dong-Seong Kim.
    - XAI (SHAP e LIME) + Não utiliza (Propõe uma alternativa ao Federated Learning) + Rede Neural Quântica-Classica (Apenas cita modelos Tree-Based) + Não aplica DP (Apenas cita, ele usa Quantum alguma coisa no lugar).
    - Relação com o nosso: 0.
- **Federated Learning Meets Explainable AI: A Comprehensive Review and Roadmap for Privacy-Preserving Cybersecurity:**
    - IEEE Xplorer.
    -  2026 2nd International Conference on Cognitive Computing in Engineering, Communications, Sciences and Biomedical Health Informatics (IC3ECSBHI).
    - Arhina Ghosh; Smriti Jaiswal; Gajra Bhatnagar; Rahul Dobriyal; Neha Tyagi; Raviraj Singh Kurmi.
    - Não propõe nada, é uma revisão da literatura. Pode ser interessante de ler.
    - Relação com o nosso: 0.
- **Differentially Private Deep Learning for Smartphone-Based Human Activity Recognition:**
    - IEEE Xplorer.
    - 2025 28th International Conference on Computer and Information Technology (ICCIT).
    - Indrojit Sarkar; Anjan Kumar Bagchi; Mohammad Sakib Shahriar.
    - XAI (SHAP e Permutation Feature Importance) + Não usa Federado, apenas cita + Rede Neural (Apenas compara com modelos Tree-based) + DP-SGD (Aplicado na rede neural).
    - Relação com o nosso: 0.
- **Leveraging XGBoost for Predictive Analytics in Healthcare: Enhancing Disease Diagnosis:**
    - IEEE Xplorer.
    - 2024 7th International Conference on Contemporary Computing and Informatics (IC3I).
    - Anurag Shrivastava; Arnav Kotiyal; Mohammed I. Habelalmateen; Ajay Rana; V.S Anusuya Devi; Bolleddu Devananda Rao.
    - XAI + Apenas cita Federado + XGBoost + Apenas cita DP.
    - Relação com o nosso: 2.
- **PrivCervBoost: Privacy-Enhanced Federated Gradient Boosting for Cervical Cancer Risk Prediction:**
    - IEEE Xplorer.
    - 2025 IEEE 2nd International Conference on Information Technology, Electronics and Intelligent Communication Systems (ICITEICS).
    - N Meenakshisundaram; Sajiv G.
    - XAI + Federated Learning + XGBoost + Não aplica DP, só cita como trabalho futuro.
    - Relação com o nosso: 3.
- **Federated Graph Neural Networks with Explainable AI for Privacy-Preserving Credit Scoring and Loan Risk Assessment:**
    - IEEE Xplorer.
    - 2026 3rd International Conference on Emerging Trends in Engineering and Medical Sciences (ICETEMS).
    - Shweta Gode; Abhilash Dhote; Aman Giri; Mayank Patel; Pramit Reddy; Nilesh Thamke.
    - XAI (SHAP + GNNExplainer) + Federated Learning (na GNN - Graph Neural Network) + Rede Neural (Apenas cita na revisão da literatura) + Apenas cita privacidade diferencial.
    - Relação com o nosso: 0.
- **A Multi-Modal Federated Graph Learning Approach for Health Insurance Pricing with Attention and Explainability on the Cloud:**
    - IEEE Xplorer.
    - 2025 Third International Conference on Cyber Physical Systems, Power Electronics and Electric Vehicles (ICPEEV).
    - Ganesh Shankar Sargam; Ramprakash Kalapala.
    - XAI + Federated Learning (Rede neural) + Rede Neural (Apenas cita modelos Tree Based) + Apenas cita DP.
    - Relação com o nosso: 0.
- **Machine Learning-Based Approaches for Malicious Insider Threat Detection: a Survey of Recent Advances and Open Issues:**
    - IEEE Xplorer.
    - 2025 International Conference on Intelligent Computing, Information and Control Systems (ICOIICS).
    - Ravi Kumar Tenali; V Vijaya Chamundeeswari.
    - Não propõe nada, é um survey.
    - Relação com o nosso: 0.
- **Federated Forest for Network Anomaly Detection:**
    - IEEE Xplorer.
    - 2025 International Symposium on Networks, Computers and Communications (ISNCC).
    - Flavien Donfack; Otily Toutsop; Tsion M. Yimer.
    - XAI + Federated Learning + Random Forest + Apenas citam privacidade diferencial.
    - Relação com o nosso: 3.
- **A Systematic Study of Machine Learning Frameworks Enabling Scalable Secure and Explainable Artificial Intelligence in Salesforce CRM Platforms:**
    - IEEE Xplorer.
    - 2026 International Conference on Electronic Systems and Intelligent Computing (ICESIC).
    - Achuta Krishna Kishore Varma Alluri.
    - XAI + Apenas cita Federated Learning + XGBoost, LightGBM e Random Forest + Aplica no modelo de regressão logistica, não nos de árvore.
    - Relação com o nosso: 2.
- **Pre-Transaction Fraud Risk Prediction in DeFi Using Explainable AI:**
    - IEEE Xplorer.
    - 2026 Sixth International Conference on Advances in Electrical, Computing, Communications and Sustainable Technologies (ICAECT).
    - Rino Thomas; Savitha K.K.
    - XAI (SHAP e LIME) + Apenas cita Federated Learning + XGBoost integrada com GNNs + Apenas citam privacidade diferencial. 
    - Relação com o nosso: 2.
- **Federated Learning for Privacy-Preserving AI in Healthcare:**
    - IEEE Xplorer.
    - 2025 3rd International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS).
    - Hemanth Dandu.
    - XAI (SHAP e Feature Importance) + Federated Learning (Rede Neural e Regressão logistica) + Rede Neural (Apenas usa modelos de árvore para comparação de desempenho local) + Apenas cita DP.
    - Relação com o nosso: 0.
- **Towards Accountable and Resilient AI-Assisted Networks: Case Studies and Future Challenges:**
    - IEEE Xplorer.
    - 2024 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit).
    - Shen Wang; Chamara Sandeepa; Thulitha Senevirathna; Bartlomiej Siniarski; Manh-Dung Nguyen; Samuel Marchal.
    - XAI (SHAP e LIME) + Federated Learning (Estou em dúvida se usou Tree Based) + Rede Neural, XGboost e LightGBM + Aplica somente na rede neural.
    - Relação com o nosso: 2.
- **IoT-Driven Deep Learning Logic to Identify Cardiovascular Diseases using Electrocardiogram Images:**
    - IEEE Xplorer.
    - 2026 Sixth International Conference on Advances in Electrical, Computing, Communications and Sustainable Technologies (ICAECT).
    - J. Gokulapriya; P. Logeswari; Thiyagarajan Jayaraman; R. Rajesh; Senthil Kumar Sengottaiyan; Yuvaraja Thangavel.
    - XAI (SHAP e LIME) + Federated Learning (Apenas na rede neural) + Rede Neural com XGBoost + Apenas cita DP.
    - Relação com o nosso: 1.
- **A Regulatory Aware AI Framework for Fraud Detection in Indian SMES:**
    - IEEE Xplorer.
    - 2026 International Conference on Computing, Sciences and Communications (ICCSC).
    - Siddhant Gupta; Rishit Nigam; Sahil Bajaj; Ashutosh Joshi; Chelsi Sen.
    - XAI (SHAP e LIME) + Apenas cita Federated Learning + GNN, VAEs e XGBoost + Aplicam apenas no VAE.
    - Relação com o nosso: 1.
- **Weighted Federated Learning with Encryption for Diabetes Classification:**
    - IEEE Xplorer.
    - 2025 Second International Conference on Artificial Intelligence for Medicine, Health and Care (AIxMHC).
    - Puyang Zhao; Zhiyi Yue; Xinhui Liu; Jingjin Wu.
    - XAI (SHAP e LIME) + Federated Learning (Apenas na rede neural) + Rede Neural (Utiliza modelos Tree-Based apenas para baseline) + Apenas cita DP.
    - Relação com o nosso: 0.
- **A Comprehensive Review of Interpretability in AI and Its Implications for Trust in Critical Applications:**
    - IEEE Xplorer.
    - 2024 4th International Conference on Sustainable Expert Systems (ICSES).
    - Ujjwal Singh Kathait; Anamika Rana; Rahul Chauhan; Ruchira Rawat.
    - Não propõe nada, é um survey.
    - Relação com o nosso: 0.
- **Secure AI-Driven Framework for Predicting Drug Toxicity Using Computational Modeling:**
    - IEEE Xplorer.
    - 2026 International Conference on ICT and Photonics (ICTP).
    - Ravi Kumar; Balajee J; Jayapal Lande; Hitendra Garg; Tamilarasi M; Pothuraju Rajarajeswari.
    - XAI + Federated Learning (Apenas com rede neural) + Rede Neural (Apenas cita modelos de árvore) +  Aplicam DP apenas na rede neural.
    - Relação com o nosso: 0.
- **Data Privacy in Machine Learning: A Pipeline for Privacy Risk Assessment:**
    - IEEE Xplorer.
    - 2025 3rd International Conference on Sustainable Computing and Data Communication Systems (ICSCDS).
    - Epifelward Niño O. Amora; Michelle P. Ombid.
    - XAI + Apenas cita Federated Learning como trabalhos futuros + Random Forest e XGBoost + Não aplicam DP apenas citam.
    - Relação com o nosso: 2.
- **Scalable Data Mining Algorithm for Predictive Healthcare Analytics:**
    - IEEE Xplorer.
    - 2025 Tenth International Conference on Science Technology Engineering and Mathematics (ICONSTEM).
    - Sarika G. Shinde; K. G. Kharade; R.K. Kamat.
    - XAI (SHAP e LIME) + Federated Learning + XGBoost + Aplicam DP nos gradientes.
    - Relação com o nosso: 4.
- **Transforming Customer Experience in Fintech through Ethical, Scalable, and Secure AI Systems:**
    - IEEE Xplorer.
    - 2026 IEEE 5th International Conference on Computing and Machine Intelligence (ICMI).
    - Muthu Selvam.
    - XAI (SHAP e LIME) + Apenas cita Federated Learning como trabalho futuro + LSTM e Random Forest + Aplicam apenas na LSTM.
    - Relação com o nosso: 2.
- **A Secured Artificial Intelligence (AI) Assisted Personal Data Prediction and Leakage Prevention System Using Deep Learning Logic:**
    - IEEE Xplorer.
    - 2025 2nd International Conference on Artificial Intelligence and Knowledge Discovery in Concurrent Engineering (ICECONF).
    - Divyapriya S; P. Neelaveni; R. Sankar; V Mythily; C. Santhana Lakshmi; N. Vani.
    - XAI + Federated Learning (Apenas no Capsule Net) + Capsule Net e XGBoost + Aplica DP diretamente nos dados.
    - Relação com o nosso: 2.
- **Smart Grid Intrusion Detection for IEC 60870-5-104 With Feature Optimization, Privacy Protection, and Honeypot-Firewall Integration:**
    - IEEE Xplorer.
    -  IEEE Access ( Volume: 13) - 17 July 2025.
    - Pedamallu Sai Mrudula; Rayappa David Amar Raj; Archana Pallakonda; Yanamala Rama Muni Reddy; K. Krishna Prakasha; V. Anandkumar.
    - XAI + Federated Learning + XGBoost, LightGBM e Decision Tree + Aplicam DP nos dados.
    - Relação com o nosso: 4.

```
("Federated Learning") AND ("Regression Tree" OR "Decision Tree" OR "Random Forest" OR "Regression Forest" OR "GBDT" OR "XGBoost") AND ("Differential Privacy") AND ("Membership Inference" OR "Label Reconstruction" OR "Inversion Attack" OR "Inference Attack")
```

- **FedXHDP: A Federated XGBoost Framework With Hierarchical Differential Privacy for Horizontally Partitioned Data:**
    - IEEE Xplorer.
    - IEEE Access ( Volume: 13) - 08 August 2025.
    - B. Sasirekha; C. Gunavathi.
    - Federated Learning + XGBoost + Aplica DP nos gradientes e nos splits + Testa a resiliência contra MIA, Reconstrução de Dados e Inversão de modelo.
    - Relação com o nosso: 4.
- **Privacy-Preserving Loan Prediction using Federated Learning, Hash-VFL, Differential Privacy, and Secure Multi-Party Computation:**
    - IEEE Xplorer.
    - 2026 4th International Conference on Inventive Computing and Informatics (ICICI).
    - Navyanth Varma; Harshith Bezawada; V.Anusha.
    - Federated Learning (Acho que também faz com Random Forest) + MLP, CNN e Random Forest + Não aplica DP em Tree Based + Testa contra MIA e de inversão de gradiente.
    - Relação com o nosso: 3.
- **Quantum-Secured Federated and Lottery Federated Learning for Privacy-Preserving AI:**
    - IEEE Xplorer.
    - 2026 Second International Conference on Intelligent Systems for Communication, IoT and Security (ICISCoIS).
    - Abirami B; Karthika Renuka D; Anusuya R.
    - Federated Learning (Tree Based e Rede Neural) + Random Forest e Gradient Boost + Aplica DP nos dados + Testa contra MIA, Bizantinos, Evasão e Sybil.
    - Relação com o nosso: 4.
- **Differentially Private Deep Learning for Smartphone-Based Human Activity Recognition:**
    - IEEE Xplorer.
    - 2025 28th International Conference on Computer and Information Technology (ICCIT).
    - Indrojit Sarkar; Anjan Kumar Bagchi; Mohammad Sakib Shahriar.
    - Apenas cita Federated Learning + Só usa modelos Tree Based como baseline + DP-SGB na Rede Neural + Apenas menciona os ataques.
    - Relação com o nosso: 1.
- **Enhancing Data Privacy in Multi-Institutional Medical AI: A Secure Vertical Federated Learning Framework:**
    - IEEE Xplorer.
    - 2025 Third International Conference on Industry 4.0 Technology (I4Tech).
    - Samruddhi Prabhune; Balaso Jagdale.
    - Federated Learning + Gradient Boost + Aplica DP nos dados + Testa contra MIA.
    - Relação com o nosso: 4.
- **Demystifying Membership Inference Attacks in Machine Learning as a Service:**
    - IEEE Xplorer.
    -  IEEE Transactions on Services Computing ( Volume: 14, Issue: 6, 01 Nov.-Dec. 2021).
    - Stacey Truex; Ling Liu; Mehmet Emre Gursoy; Lei Yu; Wenqi Wei.
    - Federated Learning + Decision Tree + Não aplica DP, apenas cita + Testa contra o MIA.
    - Relação com o nosso: 3. 
- **Federated Learning for Privacy-Preserving Diabetes Detection: A Robust Framework for Heterogeneous Healthcare Data:**
    - IEEE Xplorer.
    - 2025 7th International Conference on Innovative Data Communication Technologies and Application (ICIDCA).
    - C Dhiya; Umesh Gjh; U Kavya; G Sivashankar; S Nagendra Prabhu.
    - Federated Learning (Apenas na Rede Neural) + Rede Neural (Apenas cita Tree Based) + Aplica aos gradientes + Testa contra ataques de inferência e Reconstrução.
    - Relação com o nosso: 0. 
- **A Deployment-Oriented Privacy-Preserving CTI Framework: Integrating PIR, Federated Learning, Differential Privacy, and Practical Hardenings:**
    - IEEE Xplorer.
    - IEEE Access ( Volume: 14) - 21 April 2026.
    - Emre Camalan; Baris Celiktas
    - Federated Learning + Random Forest + Aplicam DP apenas no modelo de Regresão Logistica + Testa contra o MIA.
    - Relação com o nosso: 3.
- **GradPriv: A Gradient based Decision-Aware Fine-Grained Framework for Privacy-Utility Trade-off Optimization for Machine Learning:**
    - IEEE Xplorer.
    - 2026 6th International Conference on Innovative Research in Applied Science, Engineering and Technology (IRASET).
    - Manal Gasmi; Nassima Ait Mansour; Karim Baïna; Hanae Sbai.
    - Apenas cita aprendizado federado + Random Forest, XGBoost e MLP + Aplicam DP aos dados + Testa contra o MIA.
    - Relação com o nosso: 3.
- **Fed-DPSDG-WGAN: Differentially Private Synthetic Data Generation for Loan Default Prediction via Federated Wasserstein GAN:**
    - IEEE Xplorer.
    - IEEE Access ( Volume: 13) - 18 March 2025.
    - Padmaja Ramachandra; Santhi Vaithiyanathan.
    - Federated Learning (Apenas no modelo de GAN) + Utiliza Tree Based apenas como teste + Aplicam DP à GAN + Testa contra ataque de Reconstrução.
    - Relação com o nosso: 0. 
- **Evaluating Privacy-Preserving Techniques for Secure Financial Behavior Classification:**
    - IEEE Xplorer.
    - 2026 4th International Conference on Self Sustainable Artificial Intelligence Systems (ICSSAS).
    - Nilesh Mali; Vidyullata Devmane.
    - Apenas cita Federated Learning + Usa Tree Based apenas para comparação + Aplicam DP apenas para a Regresão logistica + Testa contra o MIA.
    - Relação com o nosso: 0.
- **Unraveling Model Inversion Attacks: A Survey of Machine Learning Vulnerabilities:**
    - IEEE Xplorer.
    - 2024 2nd International Conference on Artificial Intelligence, Blockchain, and Internet of Things (AIBThings).
    - Tanzim Mostafa; Mohamed I. Ibrahem; Mostafa M. Fouda.
    - Não propõe nada, é um survey.
    - Relação com o nosso: 0.

## Definir Onde o Nosso se Enquadra

- Tipo de Aprendizado: Se é Centralizado ou Federado.
- Tipo de FL: Caso use Aprendizado federado, é Horizontal ou Vertical?
- Distribuição dos Dados: É IID ou Non IID?
- Modelo de Base: Se o modelo utilizado no artigo foi Tree-Based, Rede Neural ou etc...
- Tipo de Problema: É um problema de clasificação, regressão ou clusterização?
- Local da DP: A aplicação de Privacidade Diferencial foi nos Dados ou no Modelo? Onde foi?
- Mecanismo de DP: Foi nos gradientes, ou para o caso de Tree Based, foi nos nós de split? Nos nós folha? Com qual mecanismo? Exponencial, Laplace ou Gaussiano?
- Orçamentos Avaliados: Quais foram os níveis de privacidade avaliados?
- Mecanismo de Agregação: Usou FedAVG, ou algum adaptado para árvores?
- Avaliação de Explicabilidade: Avaliou a explicabilidade? Se sim, como?
- Ataques de Privacidade Avaliados: Testou a resiliência do modelo contra ataques? Se sim, quais.
- Métricas de Utilidade: Quais métricas utilizou para avaliar os modelos?
- Limitações Apontadas: Quais limitações o autor citou.

|     | Tipo de Aprendizado | Tipo de FL | Distribuição dos Dados | Modelo de Base | Tipo de Problema | Local da DP | Mecanismo de DP | Orçamentos Avaliados | Mecanismo de Agregação | Avaliação de Explicabilidade | Ataques de Privacidade Avaliados | Métricas de Utilidade | Limitações Apontadas | Relevância pro Nosso |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
Explainable AI and Federated Learning for Privacy-Preserving Loan Default Prediction in Cooperative Banking Networks  (Shilpa Shivshankar Mathpati; Abhijit Chirputkar, 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 5
Federated and Fairness-Aware Learning for Rural Healthcare Risk Prediction Under Data Scarcity (Dhruti A. et al., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Cross-Silo Prediction of Hospital Readmissions using a Federated Learning Framework (M. Ayyadurai; Mythreiy Anand; Eniyan P., 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Scalable Data Mining Algorithm for Predictive Healthcare Analytics (Sarika G. Shinde; K. G. Kharade; R.K. Kamat, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Smart Grid Intrusion Detection for IEC 60870-5-104 With Feature Optimization, Privacy Protection, and Honeypot-Firewall Integration (Pedamallu Sai Mrudula et al., 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
FedXHDP: A Federated XGBoost Framework With Hierarchical Differential Privacy for Horizontally Partitioned Data (B. Sasirekha; C. Gunavathi, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Quantum-Secured Federated and Lottery Federated Learning for Privacy-Preserving AI (Abirami B; Karthika Renuka D; Anusuya R., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Enhancing Data Privacy in Multi-Institutional Medical AI: A Secure Vertical Federated Learning Framework (Samruddhi Prabhune; Balaso Jagdale, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 4 
Healthcare Security using Federated Learning and Explainable AI with Secure Aggregation (Mukhila R. et al., 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
PrivCervBoost: Privacy-Enhanced Federated Gradient Boosting for Cervical Cancer Risk Prediction (N Meenakshisundaram; Sajiv G., 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
Federated Forest for Network Anomaly Detection (Flavien Donfack; Otily Toutsop; Tsion M. Yimer, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
Privacy-Preserving Loan Prediction using Federated Learning, Hash-VFL, Differential Privacy, and Secure Multi-Party Computation (Navyanth Varma et al., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
Demystifying Membership Inference Attacks in Machine Learning as a Service (Stacey Truex et al., 2021) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
A Deployment-Oriented Privacy-Preserving CTI Framework: Integrating PIR, Federated Learning, Differential Privacy, and Practical Hardenings (Emre Camalan; Baris Celiktas, 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
GradPriv: A Gradient based Decision-Aware Fine-Grained Framework for Privacy-Utility Trade-off Optimization for Machine Learning (Manal Gasmi et al., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 3 
Trustworthy Multimodal Fraud Detection with Federated Learning and Computer Vision (Dendy K Pramudito; Jufriadif Na'am; Ferda Ernawan, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
Leveraging XGBoost for Predictive Analytics in Healthcare: Enhancing Disease Diagnosis (Anurag Shrivastava et al., 2024) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
A Systematic Study of Machine Learning Frameworks Enabling Scalable Secure and Explainable Artificial Intelligence in Salesforce CRM Platforms (Achuta Krishna Kishore Varma Alluri, 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
Pre-Transaction Fraud Risk Prediction in DeFi Using Explainable AI (Rino Thomas; Savitha K.K., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
Towards Accountable and Resilient AI-Assisted Networks: Case Studies and Future Challenges (Shen Wang et al., 2024) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
Data Privacy in Machine Learning: A Pipeline for Privacy Risk Assessment (Epifelward Niño O. Amora; Michelle P. Ombid, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
Transforming Customer Experience in Fintech through Ethical, Scalable, and Secure AI Systems (Muthu Selvam, 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
A Secured Artificial Intelligence (AI) Assisted Personal Data Prediction and Leakage Prevention System Using Deep Learning Logic (Divyapriya S. et al., 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 2 
IoT-Driven Deep Learning Logic to Identify Cardiovascular Diseases using Electrocardiogram Images (J. Gokulapriya et al., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 1 
A Regulatory Aware AI Framework for Fraud Detection in Indian SMES (Siddhant Gupta et al., 2026) | A | B | C | D | E | F | G | H | I | J | K | L | M | 1
Differentially Private Deep Learning for Smartphone-Based Human Activity Recognition (Indrojit Sarkar; Anjan Kumar Bagchi; Mohammad Sakib Shahriar, 2025) | A | B | C | D | E | F | G | H | I | J | K | L | M | 1