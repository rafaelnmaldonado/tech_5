import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import seaborn as sns


def app():
    st.title('Modelo Preditivo')

    st.write('Entre as inúmeras possibilidades de criação de modelos de previsão disponíveis, foi escolhida a tentativa de prever a quantidade de alunos não bolsistas para o ano seguinte aos dados fornecidos, ou seja, 2023. Para essa previsão, optou-se pelo uso do Random Forest, devido à sua capacidade de avaliar a importância de cada característica no processo de decisão, mas principalmente por sua eficiência no manuseio de dados desbalanceados quando combinado com técnicas como o SMOTE.')

    st.write('O modelo gerou a seguinte resposta:')

    file_path = './dados_passos.csv'
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path, delimiter=';')

    relevant_columns = [
        'IDA_2020', 'IEG_2020', 'IAN_2020', 'IPS_2020', 'IPP_2020', 'IPV_2020', 'IAA_2020',
        'IDA_2021', 'IEG_2021', 'IAN_2021', 'IPS_2021', 'IPP_2021', 'IPV_2021', 'IAA_2021',
        'INDICADO_BOLSA_2022'
    ]

    filtered_data = df[relevant_columns]
    filtered_data['INDICADO_BOLSA_2022'] = filtered_data['INDICADO_BOLSA_2022'].map({'Não': 0, 'Sim': 1})

    filtered_data.shape

    # Separar colunas numéricas e não numéricas
    numeric_cols = filtered_data.select_dtypes(include=['number']).columns
    non_numeric_cols = filtered_data.select_dtypes(exclude=['number']).columns

    # Criar o imputador para as colunas numéricas
    imputer = SimpleImputer(strategy='median')

    # Ajustar e transformar as colunas numéricas
    filtered_data[numeric_cols] = imputer.fit_transform(filtered_data[numeric_cols])

    # (Opcional) Imputação para colunas não numéricas
    imputer_non_numeric = SimpleImputer(strategy='most_frequent')
    filtered_data[non_numeric_cols] = imputer_non_numeric.fit_transform(filtered_data[non_numeric_cols])

    # Verificar se ainda há NaNs no dataset
    print(filtered_data.isna().sum())

    # Separar colunas numéricas e não numéricas
    numeric_cols = filtered_data.select_dtypes(include=[np.number])
    non_numeric_cols = filtered_data.select_dtypes(exclude=[np.number])

    # Remover NaNs das colunas numéricas e achatar a matriz
    numeric_data_matrix = numeric_cols.to_numpy()
    numeric_data_matrix = numeric_data_matrix[~np.isnan(numeric_data_matrix)]

    # Redimensionar a matriz numérica
    numeric_data_matrix = numeric_data_matrix.reshape(-1, numeric_cols.shape[1])

    # Converter de volta para DataFrame as colunas numéricas
    numeric_df = pd.DataFrame(numeric_data_matrix, columns=numeric_cols.columns)

    # Combinar de volta com as colunas não numéricas (com ou sem NaNs)
    final_df = pd.concat([numeric_df, non_numeric_cols.reset_index(drop=True)], axis=1)

    print(final_df.head())

    X = numeric_df.drop(columns=['INDICADO_BOLSA_2022'])
    y = numeric_df['INDICADO_BOLSA_2022']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Certifique-se de que as colunas no conjunto de treino e teste são as mesmas
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Inicialize o SMOTE
    smote = SMOTE(random_state=42)

    # Aplique o SMOTE para balancear os dados
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Verifique o balanceamento das classes após o SMOTE
    print(pd.Series(y_resampled).value_counts())

    # Separar as features e o target
    X = numeric_df.drop(columns=['INDICADO_BOLSA_2022'])
    y = numeric_df['INDICADO_BOLSA_2022']

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Certifique-se de que as colunas no conjunto de treino e teste são as mesmas
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


    # Aplicar o SMOTE para balancear as classes no conjunto de treino
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Treinar o modelo de Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_balanced, y_train_balanced)

    # Prever no conjunto de teste
    y_pred = rf_model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    st.image("./assets/resultado_modelo.jpg")

    # Prever no conjunto de teste
    y_pred = rf_model.predict(X_test)

    # Matriz de Confusão
    st.subheader('Matriz de Confusão:')
    st.write('Para aprimorar a interpretação do modelo criado, também foi utilizada uma ferramenta chamada Matriz de Confusão, que compara as previsões do modelo com as classes reais. Nesse contexto, o modelo obteve os seguintes resultados:')
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    st.write('**True Negatives (TN):** 239 - O modelo previu corretamente 239 alunos como não bolsistas.')
    st.write('**False Positives (FP):** 6 - O modelo previu erroneamente 6 alunos como não bolsistas.')
    st.write('**False Negatives (FN):** 24 - O modelo previu erroneamente 24 alunos como não bolsistas, quando eles deveriam ter sido classificados como bolsistas.')
    st.write('**True Positives (TP):** 1 - O modelo identificou corretamente apenas 1 aluno como bolsista.')

    # Prever probabilidades no conjunto de teste
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    # Precision-Recall Curve
    st.subheader('Relatório de Classificação:')
    st.write("O relatório de classificação é uma ferramenta fundamental na avaliação de modelos de machine learning, especialmente em tarefas de classificação. Ele oferece um conjunto de métricas que ajudam a entender a performance do modelo em prever corretamente as classes de um conjunto de dados. As principais métricas apresentadas no relatório de classificação incluem precisão (precision), recall, F1-score e suporte.")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker='.', color='blue')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')

    st.pyplot(fig)
    st.write('Analisando o gráfico acima, é possível inferir:')
    st.write('**Precisão para a classe 0 (não bolsistas):** 0.91 (91%) - Isso significa que, quando o modelo prevê que um aluno não é bolsista, ele está certo em 91% das vezes.')
    st.write('**Recall para a classe 0 (não bolsistas):** 0.98 (98%) - O modelo está capturando 98% dos alunos que realmente não são bolsistas.')
    st.write('**Precisão para a classe 1 (bolsistas):** 0.14 (14%) - O modelo é muito impreciso ao prever bolsistas.')
    st.write('**Recall para a classe 1 (bolsistas):** 0.04 (4%) - O modelo está capturando apenas 4% dos verdadeiros bolsistas.')
    

    st.subheader('Conclusão:')
    st.write('Levando em consideração os dados analisados até agora, conclui-se que o modelo é eficiente para prever alunos não bolsistas, mas apresenta limitações significativas na previsão de bolsistas. A porcentagem de alunos previstos como não-bolsistas é de 90.7%, o que indica uma forte tendência do modelo para essa classe. Isso sugere a necessidade de ajustes adicionais ou a utilização de técnicas mais avançadas para melhorar a precisão nas previsões de bolsistas.')
