import xarray
import os

baseDir="/pfs/work7/workspace/scratch/hx9916-MA/data/ERA5"
years = [i for i in range(1979, 2022)]
for year in years:
    if os.path.exists(baseDir+f'/{year}_pl.nc') and os.path.exists(baseDir+f'/{year}_sfc.nc'):
        continue
    if not os.path.exists(baseDir+f'/{year}_pl.nc'):
        ds = xarray.open_mfdataset(baseDir+f'/{year}_*_pl.nc', combine='by_coords',concat_dim='time')
        ds.to_netcdf(baseDir+f'/{year}_pl.nc')
        del ds
    if not os.path.exists(baseDir+f'/{year}_sfc.nc'):
        ds = xarray.open_mfdataset(baseDir+f'/{year}_*_sfc.nc', combine='by_coords',concat_dim='time')
        ds.to_netcdf(baseDir+f'/{year}_sfc.nc')
        del ds