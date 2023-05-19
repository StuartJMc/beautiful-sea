# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:59:09 2023

This script is to run The Bayes Bunch Streamlit app for 
WDL Semi Finals

@author: mitra
"""
#Import libraries
import streamlit as st
import pandas as pd
from PIL import Image
import streamlit.components.v1 as components

##SIDEBAR

#Create image files for sidebar
image1 = Image.open('Team_Badge_Updated.png')
image2 = Image.open('wdl.jpg')
#Create sidebar to write about the team
st.sidebar.image(image1, width = 100)
st.sidebar.write('Created by The Bayes Bunch:')
st.sidebar.write('Mitra Ganeson, Samantha Hakes, Stuart McGibbon, Roisin Holmes, Zhen-Yen Chan')
st.sidebar.caption('for the World Data League Semi Finals, May 2023')
st.sidebar.image(image2, width = 100)

##DASHBOARD
#Set up tabs
tab1, tab2, tab3 = st.tabs(["Home", 'Feature Selection', "Forecast"])

##TAB 1 = HOME PAGE

with tab1:
    st.title('Welcome!')
    col1, col2 = st.columns([2,2])
    with col1:
        st.markdown('This page serves as a product of **:blue[Beautiful Sea]**, studying biodiversity in AMPA. This product is a part of our submission for the World Data League Semi-Finals.')
        st.markdown('**:blue[Beautiful Sea]** is our open source Data Science repository to aid in modelling of coastal ecosystems.')
        st.write('Repo link: https://github.com/StuartJMc/beautiful-sea')
        st.divider()
        st.write('Visit the **Feature Selection tab** to view our top variable selections to explain each key metric.')
        st.write('Visit the **Forecast tab** to view our model output as an interactive chart.')
    with col2:
        image3 = Image.open('fish_image.jpg')
        st.image(image3)

##TAB2 == FEATURE SELECTION
with tab2:
    st.title('Feature Selections for Biodiversity Metrics')
    st.write('This is an interactive chart, showcasing a list of features most relevant to each biodiversity metric.')
    st.markdown('The three metrics we have determined are a best fit in modelling biodiversity are **Endangered Species, Invasive Species, and the Shannon Index**.')
    with st.expander("What is the Shannon Diversity Index?"):
        st.write('The Shannon Diversity Index is a popular metric used in ecology, which helps you to estimate the diversity of species within a community.')
        st.write("It's based on Claude Shannon's formula for entropy and **estimates species diversity**."
                 ,"The index takes into account the number of species living in a habitat (richness) and their relative abundance (evenness).")
        st.write('To understand the index in depth, check out the website:',
                 "https://www.omnicalculator.com/ecology/shannon-index#how-to-use-the-shannon-diversity-index-calculator")
    components.html(
        '<div class="flourish-embed flourish-chart" data-src="visualisation/13811172"><script src="https://public.flourish.studio/resources/embed.js"></script></div>',
        height = 2000)
    
##TAB3 == FORECAST
with tab3:
    st.title('Forecasting the Future of Biodiversity in AMPA')
    st.caption ('Based on the Shannon–Wiener Diversity Index')
    st.write(
        'This is an interactive chart, showcasing a time-series forecast of biodiversity measured by the Shannon-Wiener Diversity Index based on altering sea water acidity (pH).',
        'As of today, the ocean pH is about 8.1, what happens when that changes?',
        "For context, in real-world ecological data, the Shannon Diversity index's range of values is usually 1.5 - 3.5.")
    #Creating space for Flourish chart
    components.html(
        '<div class="flourish-embed flourish-chart" data-src="visualisation/13802856"><script src="https://public.flourish.studio/resources/embed.js"></script></div>',
        height = 600)

    
    

##DATA IN STREAMLIT EXAMPLE

# #Define load data function
# DATA_SCENARIO = 'scenario_forecasts.csv'
# @st.cache_data
# def load_scenario_data(nrows):
#     data = pd.read_csv(DATA_SCENARIO, nrows = nrows)
#     del data[data.columns[0]]
#     return data
##making sliders = st slider
##making categorical sliders = st select slider
##if using float slider, can use np.close to get the closest value from data to selection
    ##problem is an error pops up saying the filter is ambigous (need to mask the bool)
