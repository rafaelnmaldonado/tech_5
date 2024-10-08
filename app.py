
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os

import contexto, modelo, gerador_modelo, visualizacao, storytelling
st.set_page_config(
        page_title="Datathon",
)



class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):

        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # app = st.sidebar(
        with st.sidebar:        
            app = option_menu(
                menu_title='Menu ',
                options=['Contexto', 'Visualização', 'Modelo Preditivo', 'Gerador de Modelos', 'Storytelling'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={
                    "container": {"padding": "5!important","background-color":'black'},
                    "icon": {"color": "white", "font-size": "23px"}, 
                    "nav-link": {"color":"white","font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21"},}
                
                )

        
        if app == 'App':
            self.app()
        if app == 'Contexto':
            contexto.app()    
        if app == 'Modelo Preditivo':
            modelo.app()       
        if app == 'Gerador de Modelos':
            gerador_modelo.app()        
        if app == 'Visualização':
            visualizacao.app() 
        if app == 'Storytelling':
            storytelling.app()

multi_app = MultiApp()
multi_app.run()         
         








 