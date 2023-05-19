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

Tools to aid modelling of coastal ecosystems. Including:

- Functions for engineering relevant features from weather, climate, and oceanographic data.
- Functions for visualising and analysing data.
- A marine time series model for training and forecasting labels
- A tool for forecasting scenarios 

## Quick Start

```python
biodiversity_model = MarineTimeSeriesAnalysis(
                                    df = 'species_sample_coastal.csv,
                                    label =  'biodiversity,
                                    features = ['temperature', 'ph', ... 'chlorophyll'],
                                    split_date ='2020-01-01',
                                    freq="M",
                                    lag_values=[1, 2, 3],
                                )

#analyze seanal trends in your data
biodiversity_model.seasonal_trends()

#forecast biodiversity for your test period
biodiversity_model.forecast()

#analyze model residuals to understand potential impacting variables
biodiversity_model.residual_analysis()
```

## Package Reference

## Product Reference
A MVP interactive Streamlit page of Beautiful Sea's capabilities is available to view. The page showcases time series model results, feature importances and simulations of changing environmental variables. The dashboard provides valuable insights to understand the factors that most impact biodiversity and species counts. The simulation plots also allow users to assess the impact of environmental changes and take proactive measures.

Link to page: https://mitrag-beautiful-sea-streamlit-d-beautiful-sea-streamlit-on6bwj.streamlit.app/

Link to page repository: https://github.com/MitraG/Beautiful-Sea-Streamlit-Dashboard/tree/main

<img width="960" alt="Product Welcome Page" src="https://github.com/StuartJMc/beautiful-sea/assets/82417027/309852b0-7d05-495f-8b0f-8a477af23b42">

## References
