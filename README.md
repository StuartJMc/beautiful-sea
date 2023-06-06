![image](imgs/frances-gunn-9dMzyieG4OI-unsplash.jpg)

Photo by <a href="https://unsplash.com/@francesgunn?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Frances Gunn</a> on <a href="https://unsplash.com/photos/9dMzyieG4OI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
# Beautiful Sea
**An open source Data Science repository to aid in modelling of coastal ecosystems.**

---

<br>

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Package Reference](#package-reference)
- [Product Reference](#product-reference)
- [References](#references)

## Project Overview

Tools to aid in modelling of coastal ecosystems. Including:

- Functions for engineering relevant features from weather, climate, and oceanographic data.
- Functions for visualising and analysing data.
- A marine time series model for training and forecasting labels.
- A tool for forecasting scenarios.

## Quick Start

```python
biodiversity_model = MarineTimeSeriesAnalysis(
                                    df = species_df,
                                    label =  'biodiversity,
                                    features = ['temperature', 'ph', ... 'chlorophyll'],
                                    split_date ='2020-01-01',
                                    freq="M",
                                    lag_values=[1, 2, 3],
                                )

#analyse seanal trends in your data
biodiversity_model.seasonal_trends()

#forecast biodiversity for your test period
biodiversity_model.forecast()

#analyse model residuals to understand potential impacting variables
biodiversity_model.residual_analysis()
```

## Package Contents
### model
- MarineTimeSeriesAnalysis() -> for modelling timeseries models and performing feature analysis

### scenarios
- model_scenarios() -> for forecasting a traget variable over a range of projected out comes (changes to external variables)

### feature_eng
- get_weather_date() -> fetches daily wether data from meteo (can be used for modelling)
- get_uea_data() -> loads and transforms monthly climate data from UEA (can be used for modelling)
- get_copernicus_data() -> loads and transforms daily ocean chemical readings from Copernicus (can be used for modelling)
- get_AQI() -> loads and transforms daily air quality index data (can be used for modelling)
- get_ohi_data() -> loads and transforms yearly oncean health index data (can be used for modelling)

### label_eng
- calculate_biodiversity_metric() -> calculates biodiversity metric given occurences of mobile and sessile species in the community (can be used as a target variable)

## Product Reference
A MVP interactive Streamlit page of Beautiful Sea's capabilities is available to view. The page showcases time series model results, feature importances and simulations of changing environmental variables. The dashboard provides valuable insights to understand the factors that most impact biodiversity and species counts. The simulation plots also allow users to assess the impact of environmental changes and take proactive measures.

Link to page: https://mitrag-beautiful-sea-streamlit-d-beautiful-sea-streamlit-on6bwj.streamlit.app/

Link to page repository: https://github.com/MitraG/Beautiful-Sea-Streamlit-Dashboard/tree/main

## References

- Meteo open weather API https://open-meteo.com/
- University of East Anglia Climate Data (including references to papers) https://crudata.uea.ac.uk/cru/data/temperature/?_ga=2.35621282.2009552220.1683452211-1486011184.1682889046#datdow
- Copernicus Global Ocean Physics Reanalysis https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description
- Copernicus Atlantic-Iberian Biscay Irish- Ocean BioGeoChemistry NON ASSIMILATIVE Hindcast  https://data.marine.copernicus.eu/product/IBI_MULTIYEAR_BGC_005_003/description
- Worlds Air Pollution (AQI) - https://waqi.info/#/c/8.407/9.026/2.2z
- Ocean Health Index - https://oceanhealthindex.org/global-scores/data-download/
- Shannon Diversity Index Calculation - https://www.statology.org/shannon-diversity-index/#:~:text=The%20Shannon%20Diversity%20Index%20(sometimes,i%20*%20ln(pi)
- Deepananda and Macusci , Human Disturbance in a Tropical Rocky Shore Reduces Species Diversity - https://www.researchgate.net/publication/262458621_Human_Disturbance_in_a_Tropical_Rocky_Shore_Reduces_Species_Diversity
- pmdarima - https://github.com/alkaline-ml/pmdarima
- statsmodels - https://github.com/statsmodels/statsmodels

