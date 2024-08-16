import streamlit as st


def app():
    st.title("Visualização dos dados")
    
    st.write('Com base nos dados apresentados nos dashboards de 2021 e 2022, é evidente que o trabalho realizado pela ONG Passos Mágicos tem gerado impactos significativos no progresso acadêmico de seus alunos, especialmente os veteranos. A análise comparativa entre os dois anos revela um padrão consistente: os alunos veteranos apresentam uma maior taxa de promoção de fase, confirmando a eficácia das práticas pedagógicas da ONG.')
    st.write('Em 2021, observamos que os veteranos, de maneira geral, superaram os ingressantes em termos de promoção de fase. Por exemplo, na equipe 1, 22,45% dos veteranos foram promovidos de fase, comparado a 14,14% dos ingressantes. Esse padrão se mantém em todas as equipes, refletindo a tendência dos veteranos a progredirem mais rapidamente em seus estudos.')
    st.write('O ano de 2022 reforça ainda mais essa observação. Nas avaliações bimestrais, os veteranos continuaram a mostrar um desempenho superior. No REC_AVA_2_2022, por exemplo, 16,18% dos veteranos foram promovidos de fase, enquanto apenas 9,33% dos ingressantes alcançaram essa mesma promoção. Além disso, há uma maior incidência de veteranos sendo promovidos com bolsas, o que demonstra não apenas a sua competência, mas também o reconhecimento do seu esforço e dedicação ao longo do tempo.')
    
    st.write('Esses resultados são um reflexo direto da metodologia aplicada pela ONG Passos Mágicos. A abordagem pedagógica estruturada, focada no desenvolvimento contínuo das habilidades acadêmicas e no estímulo à criatividade e à ética, tem capacitado os alunos de forma significativa. As atividades complementares, que vão além do conteúdo acadêmico tradicional, permitem que os alunos superem suas dificuldades e se desenvolvam integralmente, o que se traduz nas taxas mais altas de promoção observadas entre os veteranos.')

    st.write('O sucesso dos veteranos na ONG Passos Mágicos não é apenas uma estatística; é a prova viva de que um ambiente de aprendizado bem estruturado, combinado com apoio contínuo e motivação, pode transformar a trajetória educacional dos alunos. A dedicação dos profissionais da ONG em proporcionar um ensino de qualidade e em estimular o desenvolvimento ético e cultural tem sido fundamental para o crescimento acadêmico e pessoal desses jovens.')

    st.write('Assim, os dados apresentados corroboram a excelência do trabalho realizado pela ONG Passos Mágicos, mostrando que, com o tempo e o devido suporte, os alunos conseguem não apenas se manter em suas fases, mas também serem promovidos e recompensados por seu desempenho, confirmando a importância e a eficácia desse projeto na vida de muitas crianças e jovens.')
    
    with open("./assets/datathon.pbix", "rb") as file:
        st.download_button(
            label="Baixar arquivo",
            data=file,
            file_name="datathon.pbix",
            mime="application/pbix"
        )