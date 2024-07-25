import ee
import os
import requests
import zipfile
import io
import geemap
import pandas as pd
import numpy as np

# The class definition
class MSA:
    def __init__(self):
      """
      Initializes an object representing a collection of Metropolitan Statistical Areas (MSAs)
      """
      self._msa_low_res = self._retrieve_msas()
      self._names = self._msa_low_res.aggregate_array('NAME').getInfo()
      self._pop_totals = self._get_pop_totals()
      self._pop_image = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Count') \
          .filter(ee.Filter.calendarRange(2020, 2020, 'year')) \
          .mean()

    @property
    def names(self) -> list[str]:
      """
      Returns a list of all the MSA names.
      """
      return self._names

    @property
    def all_msas(self) -> ee.FeatureCollection:
      """
      Returns a FeatureCollection of all Metropolitan Statistical Areas (MSAs)
      """
      return self._msa_low_res

    @property
    def all_msas_pop(self) -> dict[str, int]:
      """
      Returns a dictionary of all the MSAs' names and populations.
      """
      return {name: int(self._pop_totals.loc[name].iloc[0]) for name in self._names}

    @property
    def all_msas_pop_image(self) -> ee.Image:
      """
      Returns an Image of all Metropolitan Statistical Areas (MSAs) with population data.
      """
      return self._pop_image.clipToCollection(self._msa_low_res)

    def get_msas(self, names: str | list[str]) -> ee.FeatureCollection:
      """
      Filters a FeatureCollection of MSAs by the selected name(s).

      Args:
        names (str or list[str]): The name(s) to filter by.

      Returns:
        ee.FeatureCollection: A FeatureCollection of MSAs that match the selected names.
      """
      if not names:
        return self.all_msas
      if isinstance(names, str):
        names = [names]
      return self._msa_low_res.filter(ee.Filter.inList('NAME', names))

    def get_msas_pop(self, names: str | list[str]) -> dict[str, int]:
      """
      Returns a dictionary of the selected MSA's names and populations.

      Args:
        names (str or list[str]): The name(s) to get the population(s) for.

      Returns:
        dict[str, int]: A dictionary of MSA names and populations.
      """
      if not names:
        return {}
      if isinstance(names, str):
        names = [names]

      # Get the population for each MSA
      return {name: int(self._pop_totals.loc[name]) for name in names}

    def get_msas_pop_image(self, names: str | list[str]) -> ee.Image:
      """
      Returns an Image of the selected MSAs and populations.

      Args:
        names (str or list[str]): The name(s) to get the population(s) for.

      Returns:
        ee.Image: An Image of MSAs with population data.
      """
      if not names:
        return None
      if isinstance(names, str):
        names = [names]

      msas = self.get_msas(names)

      # Get the population image for each MSA
      return self._pop_image.clipToCollection(msas)

    # Only used so far to obtain the low resolution shapefile, at a resolution of 1:20,000,000
    def _retrieve_msas(self, resolution='20m') -> ee.FeatureCollection:
      """
      Retrieves a shapefile from the Census Bureau containing all Metropolitan Statistical Areas (MSAs) in the US and Puerto Rico.

      Args:
        resolution (str): The resolution of the shapefile (20m by default, can be 500k, 5m, or 20m)

      Returns:
        FeatureCollection: A FeatureCollection of Metropolitan Statistical Areas (MSAs)
      """
      year = 2023
      filename = f'cb_{year}_us_cbsa_{resolution}'
      if not os.path.exists(os.path.join('.',f'{filename}.shp')):
        # Download the shapefile
        response = requests.get(f'https://www2.census.gov/geo/tiger/GENZ{year}/shp/{filename}.zip')
        # Check if the request was successful
        if response.status_code != 200:
            raise Exception(f"Failed to download shapefile. Status code: {response.status_code}")
        # Extract the shapefile
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall()

      # Upload the shapefile to GEE by reading it with Latin-1 encoding, which is commonly used for shapefiles
      cbsas = geemap.shp_to_ee(f'{filename}.shp', encoding='latin1')

      # Then filter for Metropolitan Statistical Areas (MSAs), which are CBSAs with a population > 50k
      return cbsas.filter(ee.Filter.eq('LSAD', 'M1'))

    def _get_pop_totals(self) -> pd.DataFrame:
      """
      Retrieves the total population for each MSA
      """
      areas_pop = pd.read_excel('https://www2.census.gov/programs-surveys/popest/tables/2020-2023/metro/totals/cbsa-met-est2023-pop.xlsx').iloc[:, [0, 5]]
      areas_pop.columns = ['Area', 'Est. Pop July 1 2023']
      areas_pop = areas_pop.set_index(areas_pop.columns[0])
      citation = areas_pop.index[-3:]
      areas_pop = areas_pop.iloc[3:-6]
      areas_pop = areas_pop.dropna()
      areas_pop.index = areas_pop.index.str.lstrip('.')
      areas_pop.index = areas_pop.index.str.replace('In Metropolitan Statistical Area', '^ In All MSAs')
      areas_pop.index = areas_pop.index.str.replace(' Metro Area', '')
      areas_pop = areas_pop[~areas_pop.index.str.endswith('Division')]
      for line in citation:
          areas_pop.loc[line] = np.nan
      return areas_pop