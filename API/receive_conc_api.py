import ee
import pandas as pd

def get_Data(locations: ee.FeatureCollection,
             start_date: str,
             end_date: str,
             cloudmasking: float) -> pd.DataFrame:
  """
  Retrieves NOx concentration data for given locations and time period.

  Args:
      locations: Earth Engine FeatureCollection representing the locations.
      start_date: Start date in YYYY-MM-DD format.
      end_date: End date in YYYY-MM-DD format.
      cloudmasking: Cloud masking fraction.

  Returns:
      A Pandas DataFrame containing NOx concentration data.
  """

  adminSelect = locations
  no2Raw = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')

  CLOUD_MASK_FRACTION = cloudmasking
  def maskClouds(image):
    cf = image.select('cloud_fraction')
    mask = cf.lte(CLOUD_MASK_FRACTION)
    return image.updateMask(mask).copyProperties(image)

  try:
    startDate, endDate = ee.Date(start_date), ee.Date(end_date)
  except:
    print("Invalid dates")
    return None

  ndays = endDate.difference(startDate, 'days')

  def create_date_list_fn(startDate):
    def date_list_fn(days):
      return startDate.advance(days, 'days')
    return date_list_fn

  # generate the list of dates

  date_list_fn = create_date_list_fn(startDate)
  list_of_dates = ee.List.sequence(0, ndays, 1).map(date_list_fn)

  def image_mediancomposite_by_date(date):
    return ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')\
    .filterDate(ee.Date(date), ee.Date(date).advance(1, 'day'))\
    .map(maskClouds) \
    .select('tropospheric_NO2_column_number_density')\
    .median()\
    .set('system:time_start', ee.Date(date).millis())

  no2 = ee.ImageCollection(
        ee.List.sequence(0, ndays, 1)\
        .map(date_list_fn)\
        .map(image_mediancomposite_by_date)
      )

  def createConc(img):

        def getConc(img):
            no2Mean=img.reduceRegion(
                  reducer = ee.Reducer.mean(),
                  geometry = adminSelect.geometry(),
                  scale =  7000
                ).get('tropospheric_NO2_column_number_density')

            region = img.get('region')

            doy=img.date().getRelative('day', 'year')
            dow=img.date().format('E')
            dt=img.date().format("YYYY-MM-dd")

            # Handle potential missing values
            feature_dict = {
                'DOY': doy,
                'DOW': dow,
                'DATE': dt
            }
            if no2Mean:
                feature_dict['conc'] = no2Mean
                return ee.Feature(None, feature_dict)
            else:
                return None

        return getConc(img)

  no2AggChange=no2.filterDate(startDate, endDate) \
                              .map(lambda img:createConc(img)) \
                              .filter(ee.Filter.notNull(['conc']))

  def fc_to_dict(fc):
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()),
        selectors=prop_names).get('list')

    return ee.Dictionary.fromLists(prop_names, prop_lists)

  NO2_stat_dict = fc_to_dict(no2AggChange).getInfo()
  NO2_stat_df = pd.DataFrame(NO2_stat_dict)

  return NO2_stat_df
