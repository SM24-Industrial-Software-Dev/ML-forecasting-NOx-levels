# Forecasting NOx Atmospheric Concentrations for Pollution and Climate Change

This repository contains code that monitors and forecasts atmospheric NOx concentrations.

## Background

The problem we are addressing is the monitoring and forecasting of atmospheric NOx concentrations, which are primarily caused by anthropogenic activity. Large emitters of NOx include the combustion of fossil fuels for transportation, power generation, and industrial activity. These emissions can be mitigated by reducing activity, switching to alternative clean energy sources, and adapting source emissions capture systems. Not only are NOx concentrations important for pollution studies (e.g. smog), but they also serve as a proxy for the more hard-to-measure CO2 emissions, the primary greenhouse gas responsible for long-term global warming.
We aim to provide reliable forecasts of NOx concentrations, ultimately contributing to improved air quality management and reduced negative impacts on public health and the environment.

## Approach

This project implements a modeling and forecasting system for the spatially-localized levels of atmospheric NOx. To achieve this, we use the Sentinel 5P satellite data available in Google Earth Engine to obtain historical time series data of NOx concentrations over regions of interest.

Using this data, we employ state-space structural time series models to extract trends, seasonal effects and other important covariates using the historical time series data. Our models analyze daily data to identify short-term trends and effects and make predictions.

## Contents

This repo contains:

- [API](Data-APIs/msa.py) for selecting Metropolitan Statistical Areas (MSAs) and obtaining their boundaries and population data
- Data Collection, Cleaning and Analysis from Sentinel 5P satellite data in Google Earth Engine
  - [API](Data-APIs/receive_conc_api.py) for receiving NOx concentrations for a given geographical boundary
- Development and implementation of structural models for different time scales in which the individual components represent various facets of the time series, such as level, trend, seasonality, and individual covariate effects
  - [API](Data-APIs/forecaster.py) for forecasting NOx concentrations using [sts-jax](https://github.com/probml/sts-jax)
- [UI and workflow](Dash-App/Dash_Implementation.ipynb) for selecting MSAs, obtaining satellite data, modeling, and forecasting
- [Demos](Demos/) for utilizing our various APIs

## More Details

See the [presentation](Presentation/NOx_Forecasting_Presentation.pdf) or [video](https://www.youtube.com/watch?v=pV-9N6BvINk) for more details on the background, motivation, and technical approach for this project. 

## Authors

This project was developed by:

- [Aaron Poulad](https://www.linkedin.com/in/aaron-poulad-603a892a4/)
- [Ori Bach](https://www.linkedin.com/in/ori-bach-7a9b902ab/)
- [Elchonon Stein](https://www.linkedin.com/in/elchonon-stein/)

Under the mentorship of [Ramesh Natarajan](https://www.linkedin.com/in/ramesh-natarajan-07a05989/).
