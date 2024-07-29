# Machine Learning for Modeling and Forecasting Time Series of Atmospheric Concentrations of NOx Using Satellite Data

This repository contains code that monitors and forecasts atmospheric NOx concentrations.

## Background

The problem we are interested in is monitoring and forecasting atmospheric NOx concentrations, which are primarily caused by anthropogenic activity. Large producers of such emissions include fossil fuel combustion due to  transportation, power generation, and industrial activity. Not only are NOx concentrations important for pollution studies (e.g. smog), but they also serve as a proxy for the more hard-to-measure CO2 emissions, the primary GHG responsible for long-term global warming. 

## Approach

This project implements a modeling and forecasting system for the spatially-localized levels of atmospheric NOx. To achieve this, we use data from the Sentinel 5P satellite available in Google Earth Engine to obtain historical time series data of NOx concentrations over regions of interest.

Using this data, we built state space forecasting models to extract trends, seasonal effects and other important effects using the historical time series data. Our models cover both monthly-averaged data to identify long-term trends as well as daily-average data to identify short-term trends and effects.

## Contents

This repo contains:

- UI and workflow for selecting ROI, obtaining satellite data, modeling and forecasting
- Data Collection, Cleaning and Analysis from Sentinel 5P satellite data in Google Earth Engine
- Development and implementation of structural models for different time scales in which the individual components represent various facets of the time series, such as level, trend, seasonality, and individual covariate effects

## More Details

See the [presentation]() for more details on the background, motivation, and technical approach for this project. 

## Authors

This project was developed by:

- [Aaron Poulad](https://www.linkedin.com/in/aaron-poulad-603a892a4/)
- [Ori Bach](https://www.linkedin.com/in/ori-bach-7a9b902ab/)
- [Elchonon Stein](https://www.linkedin.com/in/elchonon-s-35b1b028b/)

Under the mentorship of [Ramesh Natarajan](https://www.linkedin.com/in/ramesh-natarajan-07a05989/).
