import streamlit as st

def app():
    st.title('Contexto')

    st.write('Como trabalho final da pós-graduação em Data Analytics, foi solicitado o desenvolvimento de um estudo que pudesse incluir um dashboard ou um modelo de previsão para ajudar a ONG a interpretar seus dados e tomar decisões sobre seu futuro.')
    st.write('Para fornecer uma visão geral mais completa do contexto da ONG, foi escolhido desenvolver um trabalho que abrangesse ambas as propostas de projeto.')
    st.write('Na aba de visualização, foi construído um dashboard em Power BI que pode ser baixado. Nele, é analisada a relação entre veteranos e ingressantes, além do processo de promoção de fases dos alunos da Passos Mágicos.')
    st.write('Na aba de modelo preditivo, foram explorados possíveis números de não-bolsistas, utilizando a ferramenta Random Forest.')
    st.write('Por fim, na aba de Gerador de Modelos, foi desenvolvida uma ferramenta destinada a ajudar analistas de dados a manipular as colunas do dataset fornecido, solucionando uma das dificuldades descritas pela equipe da Passos Mágicos.')