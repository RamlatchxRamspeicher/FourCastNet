import cdsapi
import os

baseDir="/pfs/work7/workspace/scratch/hx9916-MA/data/ERA5"
years = [i for i in range(1990, 2020)]
years.reverse()
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
import concurrent.futures

def download_data(year, month):
    if os.path.exists(baseDir+f'/{year}_{month}_pl.nc') and os.path.exists(baseDir+f'/{year}_{month}_sfc.nc'):
        return
    c = cdsapi.Client()
    if not os.path.exists(baseDir+f'/{year}_{month}_pl.nc'):
        c.retrieve(
            'reanalysis-era5-pressure-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'geopotential', 'relative_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind',
                ],
                'pressure_level': [
                    '50', '500', '850',
                    '1000',
                ],
                'year': str(year),
                'month': month,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
            },
            baseDir+f'/{year}_{month}_pl.nc')
        
    if not os.path.exists(baseDir+f'/{year}_{month}_sfc.nc'):
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                    'mean_sea_level_pressure', 'surface_pressure', 'total_column_water_vapour',
                ],
                'year': str(year),
                'month': month,
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '06:00', '12:00',
                    '18:00',
                ],
            },
            baseDir+f'/{year}_{month}_sfc.nc')

# Create a ThreadPoolExecutor with maximum 5 threads
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Submit the download_data function for each year and month combination
    futures = [executor.submit(download_data, year, month) for year in years for month in months]

    # Wait for all the tasks to complete
    concurrent.futures.wait(futures)


#    '/project/projectdirs/dasrepo/ERA5/oct_2021_19_31_pl.nc')
