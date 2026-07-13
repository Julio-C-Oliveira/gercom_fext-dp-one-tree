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
Explainable AI and Federated Learning for Privacy-Preserving Loan Default Prediction in Cooperative Banking Networks  (Shilpa Shivshankar Mathpati; Abhijit Chirputkar, 2026) | Federado | Horizontal | Non-IID | Random Forest | Classificação | Gradientes | Gaussiano | $\epsilon = 1.0$ | FedAvg (com logit fusion) | SHAP (TreeSHAP) e Contrafatuais | MIA | Acurácia, Precisão, Recall, F1 e Tempo | Dados sintéticos, fusão de logits simplificada e falta de validação em dados reais e MPC | 5
Federated and Fairness-Aware Learning for Rural Healthcare Risk Prediction Under Data Scarcity (Dhruti A. et al., 2026) | Federado | Horizontal | Non-IID (Dirichlet $\alpha=0.5$) | Ensemble (LightGBM + XGBoost) | Classificação | Nas predições enviadas ao servidor | Laplace | Baseado em perda empírica ($\Delta \text{AUC} = -0.014$) | Média ponderada pelo tamanho da amostra (nível de predição) | SHAP, LIME e PDP | Inferência de Pertencimento (MIA) | AUC, F1-Score, Acurácia, Precisão e Recall | Configuração simulada, explicabilidade global via proxy central e ausência de garantias formais de DP | 4 
Cross-Silo Prediction of Hospital Readmissions using a Federated Learning Framework (M. Ayyadurai; Mythreiy Anand; Eniyan P., 2025) | Federado | Horizontal | Non-IID | XGBoost Classifier | Classificação | Nós de Split e Nós Folha | - | Não quantificado nos testes | FedAvg (via Flower) | SHAP (TreeExplainer) + LLM Narrative (Gemini) | - | AUC, Acurácia, Precisão e F1-Score | Ambiente simulado (MIMIC-IV), heterogeneidade real atenuada e falta de métricas de overhead de comunicação | 4 
Scalable Data Mining Algorithm for Predictive Healthcare Analytics (Sarika G. Shinde; K. G. Kharade; R.K. Kamat, 2025) | Federado | Horizontal | Non-IID | XGBoost-Ray | Classificação (Predição de Sepse) | Nos gradientes e estatísticas locais (Hessians) | Ruído de gradiente via Opacus | $\epsilon = 1.5$, $\delta = 10^{-5}$ | Agregação segura via PySyft | SHAP (Seleção Global) + LIME (Explicação Local 3s) | MIA | AUROC, AUPRC, Sensibilidade, Especificidade | Degradação por data drift em 4 meses sem retreino e necessidade de otimização adaptativa fim-a-fim de rounds/privacy budget | 4 
Smart Grid Intrusion Detection for IEC 60870-5-104 With Feature Optimization, Privacy Protection, and Honeypot-Firewall Integration (Pedamallu Sai Mrudula et al., 2025) | Federado | Horizontal | - | Decision Tree / Isolation Forest | Classificação Multiclasse | Nos dados | Mecanismo de Laplace | $\epsilon \in \{0.1, 0.5, 1.0\}$ | Média de Pesos Global Segura | SHAP | MIA, evasão via FGSM/PGD e envenenamento de dados | Acurácia, Precisão, Recall, F1-Score, FPR e ROC-AUC | Alto overhead computacional da criptografia homomórfica (restrito a auditorias offline), alto custo de recursos para retreino adversarial e menor proteção nativa contra vazamento | 4 
FedXHDP: A Federated XGBoost Framework With Hierarchical Differential Privacy for Horizontally Partitioned Data (B. Sasirekha; C. Gunavathi, 2025) | Federado | Horizontal | IID (Otimizado) e Non-IID (Label skew via distribuição de Dirichlet com $\alpha = 0.2$ | XGBoost + DART Regularization | Classificação Binária e Multi-classe | Na seleção de splits de características, nos pesos dos nós folha e nas atualizações de gradientes | Ruído de Laplace (para splits e gradientes) e Ruído Gaussiano (para os pesos das folhas) | $\epsilon = 0.6$ por round ($\epsilon_{total} \approx 80.87$), $\delta = 10^{-5}$ | Um FedAVG modificado | Feature Importance | MIA | Acurácia | Degradação severa de desempenho sob label skew (Non-IID) e particionamento vertical, demonstrando a necessidade futura de orçamentos de privacidade adaptativos e podagem de árvores baseada em gradiente | 4 
Quantum-Secured Federated and Lottery Federated Learning for Privacy-Preserving AI (Abirami B; Karthika Renuka D; Anusuya R., 2026) | Federado | Horizontal | - | Ensemble Híbrido (Random Forest, Gradient Boosting e Stacked ANN) | Classificação | Nos dados (Pré-processamento) | Mecanismo de Laplace (aplicado em características sensíveis) | Não quantificado | FedAvg integrado com Lottery FL | - | MIA, Ataques Bizantinos, Sybil e Evasão | Acurácia de Teste | - | 4 
Enhancing Data Privacy in Multi-Institutional Medical AI: A Secure Vertical Federated Learning Framework (Samruddhi Prabhune; Balaso Jagdale, 2025) | Federado | Vertical | Non-IID (Particionamento vertical / features disjuntas por participante) | GBDT e SplitNN | Classificação | Nos dados / representações intermediárias (GBDT) e nos gradientes/pesos (SplitNN) | Ruído de Laplace (dados mesclados) e DP-SGD (SplitNN) | $\epsilon \in \{0.01, 0.1, 1.0, 2.5, 5.0, 10.0, 100.0\}$ | Concatenação horizontal segura (via SMPC) e fusão de ativações | - | MIA | Acurácia, Precisão, Recall e F1-Score | Degradação de utilidade pelo ruído de DP, overhead de comunicação/computação em criptografia (HE/TEE) e escalabilidade em dados reais heterogêneos | 4 
Healthcare Security using Federated Learning and Explainable AI with Secure Aggregation (Mukhila R. et al., 2025) | Federado | Horizontal | - | Random Forest | Classificação | Atualizações do modelo | - | $\epsilon < 3$ | FedAvg + Secure Aggregation | TreeSHAP | - | Acurácia, F1-Score e Latência | - | 3 
PrivCervBoost: Privacy-Enhanced Federated Gradient Boosting for Cervical Cancer Risk Prediction (N Meenakshisundaram; Sajiv G., 2025) | Federado | Horizontal | - | XGBoost | Classificação | - | - | - | Secure Aggregation (com máscaras) | SHAP | - | Acurácia, Precisão, Recall e F1-Score | P-value hipotético, ausência de DP (indicado como trabalho futuro), falta de integração com prontuários (EHR) reais e limitação a dados estruturados unimodais | 3 
Federated Forest for Network Anomaly Detection (Flavien Donfack; Otily Toutsop; Tsion M. Yimer, 2025) Federated Forest for Network Anomaly Detection (Flavien Donfack; Otily Toutsop; Tsion M. Yimer, 2025) | Federado | Horizontal | Non-IID | Random Forest | Classificação | - | - | - | Customizado (Média ponderada baseada no F1-score) | SHAP | - | Acurácia, Precisão, Recall e F1-Score | Performance ruim para o Cliente 1, sobreajuste potencial do SMOTE, erro na execução do SHAP e alto custo computacional de treinamento | 3 
Privacy-Preserving Loan Prediction using Federated Learning, Hash-VFL, Differential Privacy, and Secure Multi-Party Computation (Navyanth Varma et al., 2026) | Federado | Vertical | Non-IID | MLP, CNN e Random Forest | Classificação | Nos gradientes | Ruído Gaussiano | $\epsilon = 3.2$, $\delta = 10^{-5}$ | FedAvg | - | MIA, Inversão de Gradiente e Model Probing | Acurácia, F1-Score e AUC | Limitação de tamanho do dataset (614 registros) que simula apenas dados escassos locais e a perda de acurácia de até 9 pontos percentuais devido ao ruído de DP | 3 
Demystifying Membership Inference Attacks in Machine Learning as a Service (Stacey Truex et al., 2021) | Centralizado e Federado | Horizontal | IID e Non-IID | Regressão Logística, k-NN, Árvore de Decisão, Naïve Bayes e Redes Neurais | Classificação | - | - | - | Média de probabilidades (point-wise) e Votação por maioria | - | Inferência de Pertencimento (MIA Outsider e Insider) | Acurácia e Precisão | Severo trade-off utilidade-privacidade nas defesas (DP/regularização), mitigação que pode apenas reduzir o aprendizado real do modelo e forte dependência de dados/modelo | 3 
A Deployment-Oriented Privacy-Preserving CTI Framework: Integrating PIR, Federated Learning, Differential Privacy, and Practical Hardenings (Emre Camalan; Baris Celiktas, 2026) | Federado | Horizontal | - | Random Forest e Logistic Regression | Classificação | Nos gradientes/atualizações locais (LR) e nas predições/saídas (RF) | Ruído Gaussiano (para LR via DP-FedAvg) e perturbação de saída (para RF) | $\epsilon \in \{1.38, 1.66, 2.00, 2.16, 2.53, 3.44, 4.72, 5.33\}$ para $T=1$ (até $\epsilon \approx 30.12$ para $T=20$), com $\delta = 10^{-5}$ | FedAvg (com agregação segura simulada) | - | - | Acurácia, F1-Score, Precisão, Recall e ROC-AUC | Latência e overhead de banda do PIR, representação simplificada de agregação segura e suposições de não-colusão | 3 
GradPriv: A Gradient based Decision-Aware Fine-Grained Framework for Privacy-Utility Trade-off Optimization for Machine Learning (Manal Gasmi et al., 2026) | Centralizado | - | - | Random Forest, XGBoost e MLP | Classificação | Nos dados (L2, L3, L4, L5) e nos gradientes (L5 via DP-MLP/DP-SGD) | Ruído de Laplace (para L5 nos dados) e Ruído Gaussiano (para L3 nos dados e L5 via DP-SGD) | $\epsilon \in \{3.0, 6.0\}$ (L5 per-feature) e $\epsilon \in \{3.0, 6.0\}$ (DP-MLP) | - | - | MIA (baseado em confiança) e LiRA | Acurácia, F1-Score, Precisão e Recall | Composição simples no L5 eleva o budget total ($d \cdot \epsilon$), falta de testes contra ataques white-box/inversão de modelo e uso de proxy de sensibilidade simples | 3 
Trustworthy Multimodal Fraud Detection with Federated Learning and Computer Vision (Dendy K Pramudito; Jufriadif Na'am; Ferda Ernawan, 2025) | Federado | Horizontal | - | Ensemble Multimodal (XGBoost, GRU-Attention e CNN) | Classificação | Atualizações locais | Gaussiano | - | FedAvg | SHAP e pesos de atenção | - | ROC-AUC, Precisão, Recall, F1-Score e Average Precision | Partições sintéticas, generalização limitada ao dataset, stragglers e overhead de comunicação | 2 
Leveraging XGBoost for Predictive Analytics in Healthcare: Enhancing Disease Diagnosis (Anurag Shrivastava et al., 2024) | Centralizado | - | - | XGBoost | Classificação | - | - | - | - | SHAP | - | Acurácia, Precisão, Recall, F1-Score e AUC-ROC | Complexidade computacional, risco de viés por dados não representativos e falta de validação em cenários reais e clínicos | 2 
A Systematic Study of Machine Learning Frameworks Enabling Scalable Secure and Explainable Artificial Intelligence in Salesforce CRM Platforms (Achuta Krishna Kishore Varma Alluri, 2026) | Centralizado | - | - | Regressão Logística, Random Forest, SVM, XGBoost e LightGBM | Classificação | No modelo / Nos gradientes | Gaussiano | - | - | SHAP (LinearExplainer) | - | Acurácia, Precisão, Recall, F1-Score, ROC-AUC e CV ROC-AUC | Forte trade-off utilidade-privacidade na DP (queda de desempenho) e baixo recall em modelos ensemble/boosting para detecção de churn | 2 
Pre-Transaction Fraud Risk Prediction in DeFi Using Explainable AI (Rino Thomas; Savitha K.K., 2026) | Federado | Horizontal | - | GNN e Ensemble (XGBoost e FT-Transformer) | Classificação | Nas atualizações e gradientes do modelo | - | - | Agregação segura (SMPC / Byzantine-resistant) | SHAP e LIME | Ataques de envenenamento (poisoning) e evasão | - | Trabalho majoritariamente teórico/arquitetural, sem a validação empírica detalhada de métricas e orçamentos físicos de DP | 2 
Towards Accountable and Resilient AI-Assisted Networks: Case Studies and Future Challenges (Shen Wang et al., 2024) | Federado | Horizontal | Non-IID | Rede Neural (CNN/MLP para MNIST) | Classificação | Nas atualizações/gradientes do modelo | - | Não quantificado (avaliado em termos de variação de $1/\epsilon$) | FedAvg | - | - | Acurácia e User Diversity | Severa degradação da utilidade do modelo (queda de mais de 70% na acurácia) sob forte proteção de DP | 2 
Data Privacy in Machine Learning: A Pipeline for Privacy Risk Assessment (Epifelward Niño O. Amora; Michelle P. Ombid, 2025) Data Privacy in Machine Learning: A Pipeline for Privacy Risk Assessment (Epifelward Niño O. Amora; Michelle P. Ombid, 2025) | Centralizado | - | IID | Multi-modelo (XGBoost, Random Forest, SVM, Regressão Logística, k-NN) | Classificação | Nos dados (através do Privacy Risk Score calculado a partir de quase-identificadores) | - | - | - | - | - | Acurácia, Precisão, Recall, F1-Score e ROC-AUC | O uso de base totalmente rotulada/balanceada pode não refletir dados reais, risco de vazamento de privacidade devido ao uso do score de risco como feature e falta de DP/criptografia nativas | 2 
Transforming Customer Experience in Fintech through Ethical, Scalable, and Secure AI Systems (Muthu Selvam, 2026) | Centralizado | - | IID | Multi-modelo (Random Forest, LSTM e GPT-2 Medium) | Classificação | Nos gradientes (durante o treinamento via DP-SGD) | Gaussiano (via DP-SGD) | $\epsilon = 1.0$ | - | SHAP e LIME | Inversão de modelo, Injeção de prompt, Envenenamento de política adversarial e Vazamento de dados | Tempo de resposta (Latência), Vazão (Throughput), Índice de Viés, Índice de Justiça, Avaliação de Explicabilidade e Violações de Segurança | Restrito a ambiente de sandbox, aumento de 10-15% no custo computacional de pico, trade-off latência-justiça em larga escala, overhead de armazenamento do blockchain e desafios de integração com APIs legadas | 2 
A Secured Artificial Intelligence (AI) Assisted Personal Data Prediction and Leakage Prevention System Using Deep Learning Logic (Divyapriya S. et al., 2025) | Federado | - | - | CapsuleNet-XGBoost | Classificação | Nos dados (pré-processamento) | Gaussiano | $\epsilon = 1.2$ | - | Atenção, Shapley e LRP (Explainable AI) | MIA e Model Inversion | Acurácia, Precisão, Recall e F1-Score | - | 2 
IoT-Driven Deep Learning Logic to Identify Cardiovascular Diseases using Electrocardiogram Images (J. Gokulapriya et al., 2026) | Federado | Horizontal | - | InceptionNet + XGBoost | Classificação | Atualizações do modelo (pesos locais) | Homomorphic Encryption (criptografia) | - | FedAvg | SHAP | - | Acurácia, Sensibilidade, Especificidade e F1-Score | Sensibilidade a ruídos/artefatos, variabilidade do sinal por posicionamento do sensor e equipamentos com baixa amostragem | 1 
A Regulatory Aware AI Framework for Fraud Detection in Indian SMES (Siddhant Gupta et al., 2026) | Federado | - | - | GNN + VAE + IndicBERT | Classificação | - | - | - | - | SHAP | - | F1-Score e AUC | Dependência de dados sintéticos, fragilidade do parser em tags não padronizadas, menor precisão em Tamil e ausência de privacidade diferencial/agregação segura | 1
Differentially Private Deep Learning for Smartphone-Based Human Activity Recognition (Indrojit Sarkar; Anjan Kumar Bagchi; Mohammad Sakib Shahriar, 2025) | Centralizado | - | - | Rede Neural (AR-Net) | Classificação | Nos gradientes | DP-SGD com Ruído Gaussiano | $\epsilon \in {1.0, 35.0}$ | - | SHAP e Permutation Feature Importance | - | F1-Score e Acurácia | Maior queda de utilidade em modelos mais complexos e dificuldade em distinguir posturas estáticas similares | 1

# Comparação de Desempenho

## String de Busca
```
("Regression Tree" OR "Random Forest" OR "GBDT" OR "XGBoost" OR "Gradient Boosting" OR "GBRT" OR "Regression Forest") AND ("Regression) AND ("Differential Privacy" OR "DP" OR "differentially private")
```

## Retorno da Busca
