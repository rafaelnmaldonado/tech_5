import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def app():

    file_path = './dados_passos.csv'
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path, delimiter=';')

    pd.DataFrame(df.loc[df['NOME'] == 'ALUNO-2']).transpose().T

    def filter_columns(df, filters: list): 
        selected_columns = [True] * len(df.columns)  
        for index, column in enumerate(df.columns):
            if any(filter in column for filter in filters): selected_columns[index] = False
        return df[df.columns[selected_columns]]

    def cleaning_dataset(df):
        _df = df.dropna(subset=df.columns.difference(['NOME']), how='all') 
        _df = _df[~_df.isna().all(axis=1)] 
        return _df

    df_2022 = filter_columns(df, ['2020', '2021'])
    df_2022 = cleaning_dataset(df_2022)
    # st.write(df_2022)
    
    scaler = MinMaxScaler()
    
    
    def criar_modelo(colunas_selecionadas, coluna_y):
        X = df_2022[colunas_selecionadas]
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y = df_2022[coluna_y]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        st.write(f'Acurácia do modelo: {acuracia*100:.3f}%')
        
        return modelo

    def fazer_previsao(dados_entrada, colunas_selecionadas, modelo):
        dados_df = pd.DataFrame([dados_entrada], columns=colunas_selecionadas)
        dados_df = pd.DataFrame(scaler.fit_transform(dados_df), columns=dados_df.columns)
        previsao = modelo.predict(dados_df)
        return previsao[0]
    
    with st.form(key='modelo'):
        colunas_modelo = {"Indice do Desenvolvimento Educacional":"INDE_2022", "Classificação Geral":"CG_2022", "Classificação na Fase":"CF_2022", "Classificação na Turma":"CT_2022", "Nota Português": "NOTA_PORT_2022", "Nota Matemática": "NOTA_MAT_2022", "Nota Inglês": "NOTA_ING_2022", "Indicador de Aprendizado": "IDA_2022", "Indicador de Auto Avaliação": "IAA_2022", "Indicador de Engajamento": "IEG_2022", "Indicador Psicossocial": "IPS_2022"}
        colunas_previsao = {"Pedra": "PEDRA_2022", "Ponto de Virada": "PONTO_VIRADA_2022", "Indicado a Bolsa": "INDICADO_BOLSA_2022"}
        
        colunas = st.multiselect(
            "Quais colunas você quer utilizar para previsão?", list(colunas_modelo.keys())
        )
        
        y = st.selectbox("Qual coluna você quer prever?", list(colunas_previsao.keys()))
        
        colunas_df = [colunas_modelo[string] for string in colunas]
        coluna_y = colunas_previsao[y]
    
        st.form_submit_button(label='Criar modelo')
    
    if colunas_df and coluna_y:
        modelo = criar_modelo(colunas_df, coluna_y)
        st.success("Modelo criado com sucesso!")
        
        st.write("Caso queira fazer uma previsão, preencha os dados:")
        entradas = {}
        for coluna in colunas:
            entradas[coluna] = st.text_input(coluna)
        botao_previsao = st.button(label='Fazer previsão')
        
        if botao_previsao:
            colunas_entrada = {colunas_modelo[chave]: valor for chave, valor in entradas.items()} 
            st.success("Previsão efetuada com sucesso!")       
            st.write(f'{y}: {fazer_previsao(colunas_entrada, colunas_df, modelo)}')


    
