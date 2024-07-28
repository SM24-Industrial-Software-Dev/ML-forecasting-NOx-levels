import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
import sts_jax.structural_time_series as sts
import ee

class NOxForecaster():

  """
  A class for forecasting NOx concentrations.
  """

  def __init__(self, incomplete_df: pd.DataFrame):
    """
    Initializes the NOxForecaster class.

    Args:
        incomplete_df (pd.DataFrame): The input DataFrame containing NOx concentration data.

    Raises:
        AssertionError: If the input DataFrame is missing required columns.
    """
    required_columns = {'date', 'nox-concentration'#, 'isholiday'
    }
    assert required_columns.issubset(incomplete_df.columns), "Input DataFrame is missing required columns"

    self.df = _fill_df(incomplete_df)
    self.time_series = jnp.array(self.df['nox-concentration'].to_numpy(), dtype=np.float32)[:, None]
    self.dates = np.array(self.df['date'].to_numpy())
    # self.holidays = np.array(df['isholiday'].to_numpy())[:, None]

  def fit_dummy_seasonal_model(self, num_forecast_steps: int):
    """
    Fits a model with a dummy seasonal component.

    Args:
        num_forecast_steps (int): Number of forecast steps.

    Returns:
        model: The fitted model.
        opt_param: The optimal parameters.
    """
    model_components = [sts.LocalLinearTrend(), sts.SeasonalDummy(num_seasons=7)]
    return self._fit_model(model_components, num_forecast_steps)

  def fit_trig_seasonal_model(self, num_forecast_steps: int):
    """
    Fits a model with trigonometric seasonal component.

    Args:
        num_forecast_steps (int): Number of forecast steps.

    Returns:
        model: The fitted model.
        opt_param: The optimal parameters.
    """
    model_components = [sts.LocalLinearTrend(), sts.SeasonalTrig(num_seasons=7)]
    return self._fit_model(model_components, num_forecast_steps)

  def get_decomposition_data(model, param_samples, num_forecast_steps: int):
    """
    Gets the structural decomposition data of the time series.

    Args:
        model: The forecasting model.
        param_samples: Optimal parameter values obtained from maximum likelihood.
        num_forecast_steps (int): Number of forecast steps.

    Returns:
        decomposition_data (dict): A dictionary containing the decomposition data for each component.
    """
    # Decompose the fitted model into STS components
    component_posterior_dict = model.decompose_by_component(param_samples, self.time_series)

    # Extract the decomposition data for each component
    decomposition_data = {}
    for component_name, component_posterior in component_posterior_dict.items():
        component_mean = jnp.squeeze(component_posterior['pos_mean'])
        component_std = jnp.squeeze(jnp.sqrt(component_posterior['pos_cov']))
        decomposition_data[component_name] = {
          'mean': component_mean,
          'std': component_std
        }

    return decomposition_data

  def get_forecast(model, param_samples, num_forecast_steps: int):
    """
    Generates forecasts and forecast errors for a specified number of steps.

    Args:
        model: The forecasting model.
        param_samples: Optimal parameter values obtained from maximum likelihood.
        num_forecast_steps (int): Number of forecast steps.

    Returns:
        forecast_means (array): Mean forecast values.
        forecast_scales (array): Standard deviation of forecast values.
    """
    # Generate forecasts
    forecast_means, forecast_scales = model.forecast(param_samples, self.time_series, num_forecast_steps)

    # Concatenate and squeeze forecast arrays
    forecast_means = jnp.concatenate(forecast_means, axis=0).squeeze()
    forecast_scales = jnp.concatenate(forecast_scales, axis=0).squeeze()

    # Calculate mean forecast values and standard deviation of forecast scales
    forecast_means = forecast_means.mean(axis=0)
    forecast_scales = jnp.std(forecast_scales, axis=0)

    return forecast_means, forecast_scales    

  def _fit_model(self, model_components, num_forecast_steps: int):
    """
    Internal method to fit a structural time series model.

    Args:
        model_components (list): A list of model components (trend, seasonal, etc.).
        num_forecast_steps (int): Number of forecast steps.

    Returns:
        model: The fitted model.
        opt_param: The optimal parameters.
    """
    model = sts.StructuralTimeSeries(model_components, obs_distribution='Gaussian', obs_time_series=self.time_series)
    key = jr.PRNGKey(42)
    opt_param, _losses = model.fit_mle(obs_time_series=self.time_series, key=key)
    return model, opt_param

  def _fill_df(incomplet_df: pd.DataFrame):
    """
    Fills in the missing nox-concentration data with a rolling median.
    """
    # Create a complete date range
    dates = pd.date_range(incomplet_df['date'].min(), incomplet_df['date'].max())
    df = pd.merge(pd.DataFrame({'date': dates}), incomplet_df, on='date', how='left')

    # Fill missing values
    while df['nox-concentration'].isna().any():
        df['nox-concentration'] = df['nox-concentration'].fillna(df['nox-concentration'].rolling(7, center=True, min_periods=1).median())
    return df