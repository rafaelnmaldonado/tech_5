import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


def app():
    st.title("Gerador de Modelos Preditivos")
    
    st.write("Esta aplicação foi desenvolvida para analisar dados de avaliação da ONG Passos Mágicos, proporcionando uma ferramenta interativa e acessível. Ela permite que os analistas de dados construam e avaliem modelos preditivos, visualizem a importância das variáveis e façam previsões com base em novos dados. Isso facilita a compreensão dos padrões e das relações dentro dos dados, oferecendo insights valiosos para tomada de decisões e estudos adicionais.")
    
    st.subheader("Funcionalidades")
    st.write("As funcionalidades criadas para construção foram as seguintes:")
    
    st.write("- Normalização e Limpeza de Dados: Os dados são normalizados usando MinMaxScaler, garantindo consistência nas escalas das variáveis. Limpeza de dados para remover entradas inválidas ou incompletas.")
    
    st.write("- Treinamento de Modelos Preditivos: Criação de modelos preditivos utilizando Random Forest, com separação dos dados em treino e teste. Cálculo da acurácia do modelo para avaliar sua eficácia.")
    
    st.write("- Visualização de Importância das Variáveis: A importância das variáveis é visualizada em um gráfico de Pareto, permitindo a identificação dos fatores mais relevantes para as previsões.")
    
    st.write("- Interface de Previsão Interativa: Interface que permite aos usuários inserir novos dados para gerar previsões, oferecendo uma aplicação prática e em tempo real dos modelos preditivos. Esta aplicação não só aprimora a análise de dados para a ONG Passos Mágicos, mas também oferece uma base robusta para estudos futuros e otimização de processos internos.")

    st.subheader("Gerador")
    
    st.write("1 - Escolha a Variável Alvo: Selecione a variável que deseja prever.")
    st.write("2 - Selecione Variáveis Auxiliares: Escolha as variáveis que ajudarão a prever a variável alvo.")
    st.write('3 - Crie o Modelo: Clique no botão "Criar Modelo" para gerar o modelo preditivo.')
    st.write('4 - Verifique a Acurácia: O produto exibirá a acurácia do modelo criado.')
    st.write('5 - Veja o Gráfico de Pareto: Visualize o gráfico de Pareto para entender a importância de cada variável na previsão.')

    file_path = './dados_passos.csv'
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path, delimiter=';')

    pd.DataFrame(df.loc[df['NOME'] == 'ALUNO-2']).transpose().T

    def filter_columns(df, filters: list): 
        selected_columns = [True] * len(df.columns)  
        for index, column in enumerate(df.columns):
            if any(filter in column for filter in filters): selected_columns[index] = False
        return df[df.columns[selected_columns]]

    df_2022 = filter_columns(df, ['2020', '2021'])
    
    scaler = MinMaxScaler()
    
    
    def criar_modelo(colunas_selecionadas, coluna_y):
        todas_colunas = colunas_selecionadas + [coluna_y]
        df = df_2022[todas_colunas].dropna()
        X = df[colunas_selecionadas]
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        y = df[coluna_y]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(X_train, y_train)
        
        st.success("Modelo criado com sucesso!")

        y_pred = modelo.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        st.write(f"##### Acurácia do modelo: {acuracia*100:.2f}%")
        
        importancia = modelo.feature_importances_ * 100
        
        colunas = [chave for chave, valor in colunas_modelo.items() if valor in colunas_selecionadas]
        
        importancia = list(zip(importancia, colunas))
        
        df_imp = pd.DataFrame(importancia, columns=['imp', 'col']).sort_values(by='imp', ascending=False)
        
        df_imp['acum'] = df_imp['imp'].cumsum() / df_imp['imp'].sum() * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_imp['col'],
            y=df_imp['imp'],
            name='Importância'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_imp['col'],
            y=df_imp['acum'],
            name='Porcentagem Acumulada',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Gráfico de Pareto das importâncias',
            yaxis=dict(
                title='Importância (%)',
                range=[0, 100]
            ),
            yaxis2=dict(
                title='Porcentagem Acumulada',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            xaxis=dict(
                title='Colunas'
            )
        )
        
        st.plotly_chart(fig)
        
        return modelo

    def fazer_previsao(dados_entrada, colunas_selecionadas, modelo):
        dados_df = pd.DataFrame([dados_entrada], columns=colunas_selecionadas)
        dados_df = pd.DataFrame(scaler.fit_transform(dados_df), columns=dados_df.columns)
        previsao = modelo.predict(dados_df)
        return previsao[0]
    
    with st.form(key='modelo'):
        colunas_modelo = {"Indice do Desenvolvimento Educacional":"INDE_2022", "Classificação Geral":"CG_2022", "Classificação na Fase":"CF_2022", "Classificação na Turma":"CT_2022", "Nota Português": "NOTA_PORT_2022", "Nota Matemática": "NOTA_MAT_2022", "Nota Inglês": "NOTA_ING_2022", "Indicador de Aprendizado": "IDA_2022", "Indicador de Auto Avaliação": "IAA_2022", "Indicador de Engajamento": "IEG_2022", "Indicador Psicossocial": "IPS_2022"}
        colunas_previsao = {"Pedra": "PEDRA_2022", "Ponto de Virada": "PONTO_VIRADA_2022", "Indicado a Bolsa": "INDICADO_BOLSA_2022"}
        
        y = st.selectbox("Qual coluna você quer prever?", list(colunas_previsao.keys()))
        
        colunas = st.multiselect(
            "Quais colunas você quer utilizar para previsão?", list(colunas_modelo.keys())
        )
        
        colunas_df = [colunas_modelo[string] for string in colunas]
        coluna_y = colunas_previsao[y]
    
        st.form_submit_button(label='Criar modelo')
    
    if colunas_df and coluna_y:
        modelo = criar_modelo(colunas_df, coluna_y)
        
        st.subheader("Previsão")
        st.write("Preencha os Dados para Previsão: Insira os valores nas caixas de texto para prever novos resultados. O intervalo permitido para cada variável já está indicado nos campos correspondentes.")
        entradas = {}
        for coluna in colunas:
            import math
            col = colunas_modelo[coluna]
            min = math.floor(df_2022[col].min())
            max = math.ceil(df_2022[col].max())
            
            entradas[coluna] = st.text_input(coluna, placeholder=f"Entre {min} e {max}")
        botao_previsao = st.button(label='Fazer previsão')
        
        if botao_previsao:
            colunas_entrada = {colunas_modelo[chave]: valor for chave, valor in entradas.items()} 
            previsao = fazer_previsao(colunas_entrada, colunas_df, modelo)
            st.success("Previsão efetuada com sucesso!")       
            st.write(f'{y}: {previsao}')


    
