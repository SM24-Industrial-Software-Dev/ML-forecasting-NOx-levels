import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax.random as jr
import sts_jax.structural_time_series as sts

class NOxForecaster:

    """
    A class for forecasting NOx concentrations.
    """

    def __init__(self, incomplet_df: pd.DataFrame):
        """
        Initializes the NOxForecaster class.

        Args:
            incomplet_df (pd.DataFrame): The input DataFrame containing NOx concentration data.

        Raises:
            AssertionError: If the input DataFrame is missing required columns.
        """
        required_columns = {'date', 'nox-concentration'}
        assert required_columns.issubset(incomplet_df.columns), "Input DataFrame is missing required columns"

        self.df = NOxForecaster.fill_df(incomplet_df)
        self.time_series = jnp.array(self.df['nox-concentration'].to_numpy(), dtype=np.float32)[:, None]
        self.dates = np.array(self.df['date'].to_numpy())
        if 'isholiday' in self.df.columns:
            self.holidays = np.array(self.df['isholiday'].to_numpy())[:, None]
        else:
            # Handle the case where the column does not exist
            self.holidays = None
        self.modelcomps = [
            sts.LocalLinearTrend(),
            sts.LinearRegression(
                dim_covariates=1,
                add_bias=False,
                name="holiday_effect",
            ),
            sts.Autoregressive(order=1,name="autoregress_effect")
        ]

    def fit_dummy_seasonal_model(self, with_covariates: bool = True) -> tuple:
        """
        Fits a structural time series model with a dummy seasonal component.

        Args:
            with_covariates (bool, optional): Whether to include covariates in the model. Defaults to True.

        Returns:
            tuple: A tuple containing:
                model (sts.StructuralTimeSeries): The fitted StructuralTimeSeries model.
                opt_param: The optimized parameters for the model.

        Note:
            If there is no holidays data, this method will always fit the model without covariates.
        """
        if with_covariates and self.holidays:
            return self._fit_model_with_covariates(sts.SeasonalDummy(num_seasons=7))
        return self._fit_model(sts.SeasonalDummy(num_seasons=7))
        
    def fit_trig_seasonal_model(self, with_covariates: bool = True) -> tuple:
        """
        Fits a structural time series model with a trigonometric seasonal component.

        Args:
            with_covariates (bool, optional): Whether to include covariates in the model. Defaults to True.

        Returns:
            tuple: A tuple containing:
                model (sts.StructuralTimeSeries): The fitted StructuralTimeSeries model.
                opt_param: The optimized parameters for the model.

        Note:
            If there is no holidays data, this method will always fit the model without covariates.
        """
        if with_covariates and self.holidays:
            return self._fit_model_with_covariates(sts.SeasonalTrig(num_seasons=7))
        return self._fit_model(sts.SeasonalTrig(num_seasons=7))

    def get_decomposition_data(self, model, param_samples) -> dict:
        """
        Gets the structural decomposition data of the time series.

        Args:
            model: The forecasting model.
            param_samples: The optimal parameters obtained from maximum likelihood.

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

    def get_forecast(self, model, param_samples, num_forecast_steps: int | None = None, holidays: pd.Series | None = None) -> tuple:
        """
        Generates forecasts and forecast errors for a specified number of steps,
        optionally using covariates.

        Args:
            model: The forecasting model.
            param_samples: Optimal parameter values obtained from maximum likelihood.
            num_forecast_steps (int, optional): Number of forecast steps. Defaults to None.
            holidays (pd.Series, optional): Holidays data for covariates. Defaults to None.

        Returns:
            tuple: A tuple containing:
                forecast_means (np.ndarray): Mean forecast values.
                forecast_scales (np.ndarray): Standard deviation of forecast values.
        """
        if holidays is not None and model.with_covariates:
            num_forecast_steps = len(holidays)
            forecast_covariates = [holidays, None]
            past_covariates = self.holidays
        elif num_forecast_steps is not None:
            forecast_covariates = None
            past_covariates = None
        else:
            return None

        # Generate forecast
        forecast_means, forecast_scales = model.forecast(
            param_samples,
            self.time_series,
            num_forecast_steps,
            past_covariates=past_covariates,
            forecast_covariates=forecast_covariates,
        )

        # Concatenate and squeeze forecast arrays
        forecast_means = jnp.concatenate(forecast_means, axis=0).squeeze()
        forecast_scales = jnp.concatenate(forecast_scales, axis=0).squeeze()

        # Calculate mean forecast values and standard deviation of forecast scales
        forecast_means = forecast_means.mean(axis=0)
        forecast_scales = jnp.std(forecast_scales, axis=0)

        return forecast_means, forecast_scales
    
    def _fit_model_with_covariates(self, seasonal_component) -> tuple:
        """
        Internal method to fit a structural time series model.

        Args:
            seasonal_component: The seasonal component to include in the model.

        Returns:
            tuple: A tuple containing:
                model (sts.StructuralTimeSeries): The fitted StructuralTimeSeries model.
                opt_param: The optimized parameters for the model.
        """
        model_components = self.modelcomps[:]
        model_components.insert(1, seasonal_component)
        model = sts.StructuralTimeSeries(
            model_components=model_components,
            obs_distribution="Gaussian",
            obs_time_series=self.time_series,
            covariates=self.holidays,
        )
        key = jr.PRNGKey(42)
        opt_param, _losses = model.fit_mle(
            obs_time_series=self.time_series,
            covariates=self.holidays,
            num_steps=2000,
            key=key
        )
        model.with_covariates = True
        return model, opt_param

    def _fit_model(self, seasonal_component) -> tuple:
        """
        Internal method to fit a structural time series model.

        Args:
            seasonal_component: The seasonal component to include in the model.

        Returns:
            tuple: A tuple containing:
                model (sts.StructuralTimeSeries): The fitted StructuralTimeSeries model.
                opt_param: The optimized parameters for the model.
        """
        model_components = [self.modelcomps[0], seasonal_component]
        model = sts.StructuralTimeSeries(model_components, obs_distribution='Gaussian', obs_time_series=self.time_series)
        key = jr.PRNGKey(42)
        opt_param, _losses = model.fit_mle(obs_time_series=self.time_series, key=key)
        model.with_covariates = False
        return model, opt_param

    @staticmethod
    def fill_df(incomplete_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills in the missing nox-concentration data with a rolling median.
        """
        # Create a complete date range
        dates = pd.date_range(incomplete_df['date'].min(), incomplete_df['date'].max())
        df = pd.merge(pd.DataFrame({'date': dates}), incomplete_df, on='date', how='left')

        # Fill missing values
        while df['nox-concentration'].isna().any():
            df['nox-concentration'] = df['nox-concentration'].fillna(
                df['nox-concentration'].rolling(7, center=True, min_periods=1).median()
            )

        return df