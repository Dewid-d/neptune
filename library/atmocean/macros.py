from atmocean.stats.oras import MEAN_DATA_VARS_ORAS5, STD_DATA_VARS_ORAS5
from atmocean.stats.era import MEAN_DATA_VARS_ERA5, STD_DATA_VARS_ERA5

from enum import Enum


class RuntimeMacros:
    """
    Class to access to Macros at runtime.
    """
    def __init__(self):
        self.levels = [300, 500, 600, 700, 850, 925, 1000]
        self.depths = [0.50576, 1.5558553, 2.6676817, 8.092519, 26.5583, 53.850636, 108.03028, 199.7900, 300.8875, 508.6399, 1045.854, 1945.296, 2955.570, 5089.479]
        self.atmos_3d = ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'specific_humidity', 'vertical_velocity']
        self.atmos_2d = ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'sea_ice_cover', 'sea_surface_temperature', 'surface_pressure']
        self.ocean_3d = ['thetao_oras', 'so_oras', 'uo_oras', 'vo_oras']
        self.ocean_2d = ['zos_oras', 'mlotst_oras', 'siconc_oras', 'sithick_oras']

    @property
    def atmos_vars(self):
        v = [f'{f}_{l}' for f in self.atmos_3d for l in self.levels]
        return v + self.atmos_2d

    @property
    def ocean_vars(self):
        v = [f'{f}_{int(d)}' for f in self.ocean_3d for d in self.depths]
        return v + self.ocean_2d

    @property
    def all_vars(self):
        return self.atmos_vars + self.ocean_vars

    @property
    def stats(self):
        return {'mean': {**MEAN_DATA_VARS_ERA5, **MEAN_DATA_VARS_ORAS5},  'std': {**STD_DATA_VARS_ERA5, **STD_DATA_VARS_ORAS5}}

    @property
    def atmos_only_vars(self):
        excluded = ['mean_sea_level_pressure', 'sea_ice_cover', 'sea_surface_temperature', 'surface_pressure']
        v = [f'{f}_{l}' for f in self.atmos_3d for l in self.levels]
        v += [f for f in self.atmos_2d if f not in excluded]
        return v

    @property
    def ocean_only_vars(self):
        excluded = [
            'mlotst_oras', 'siconc_oras', 'sithick_oras',
            'mlotst_glor', 'siconc_glor', 'sithick_glor',
            'mlotst_cglo', 'siconc_cglo', 'sithick_cglo',
        ]
        v = [f'{f}_{int(d)}' for f in self.ocean_3d for d in self.depths]
        v += [f for f in self.ocean_2d if f not in excluded]
        return v

    def _get_system_ids(self, sys_vars):
        """Internal method to map variable names with indices"""
        all_v = self.all_vars
        sys_ids = {v: [] for v in sys_vars}
        for var in sys_vars:
            for i, v in enumerate(all_v):
                # Se è una 3D (var_level) o una 2D (var)
                if '_'.join(v.split('_')[:-1]) == var or v == var:
                    sys_ids[var].append(i)
        return sys_ids

    @property
    def atmos_3d_ids(self): return self._get_system_ids(self.atmos_3d)

    @property
    def atmos_2d_ids(self): return self._get_system_ids(self.atmos_2d)

    @property
    def ocean_3d_ids(self): return self._get_system_ids(self.ocean_3d)

    @property
    def ocean_2d_ids(self): return self._get_system_ids(self.ocean_2d)

    @property
    def drivers_configuration(self):
        return {
            'atmos_ocean_full': self.all_vars, 
            'atmos': self.atmos_vars, 
            'ocean': self.ocean_vars, 
            'atmos_only': self.atmos_only_vars, 
            'ocean_only': self.ocean_only_vars, 
        }

    @property
    def all_drivers(self):
        io_dict = {
            'atmos': {'fields_3d': self.atmos_3d, 'fields_2d': self.atmos_2d}, 
            'ocean': {'fields_3d': self.ocean_3d, 'fields_2d': self.ocean_2d}
        }
        return {
            'levels': self.levels, 
            'depths': self.depths, 
            'input': io_dict, 
            'output': io_dict
        }

# Initialization
# if we call this at the beginning of training script,
# it will read the correct enviroment variable
macros = RuntimeMacros()
MEAN_DATA_VARS = {**MEAN_DATA_VARS_ERA5, **MEAN_DATA_VARS_ORAS5}
STD_DATA_VARS = {**STD_DATA_VARS_ERA5, **STD_DATA_VARS_ORAS5}
ATMOS_3D_VARS = macros.atmos_3d
ATMOS_2D_VARS = macros.atmos_2d
OCEAN_3D_VARS = macros.ocean_3d
OCEAN_2D_VARS = macros.ocean_2d
ALL_DATA_VARS = macros.all_vars
ATMOS_3D_IDS = macros.atmos_3d_ids
ATMOS_2D_IDS = macros.atmos_2d_ids
OCEAN_3D_IDS = macros.ocean_3d_ids
OCEAN_2D_IDS = macros.ocean_2d_ids
ALL_DRIVERS = macros.all_drivers

# fixed constants
YEARS = list(range(1993, 2023))
TRAIN_YEARS = list(range(1993, 2017))
VALID_YEARS = [2017, 2018, 2019]
TEST_YEARS = [2020, 2021, 2022]

SHAPE_025_DEGREE = (721, 1440)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# MACROS FOR COMPARISON METRICS (2025-12-05)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class CompModels(Enum):
    REA = 'rea'
    CLM = 'clm'
    ORAS5 = 'oras5'
    OM4 = 'om4'
    ATLAS = 'atlas'
    ORCADL = 'orca_dl'
    SAMUDRA_TH_DYN = 'samudra_thermo_dynamic'
    SAMUDRA_TH = 'samudra_thermo'

MODEL_NAMES = {
    'rea': 'Reanalysis', 
    'clm': 'Climatology', 
    'oras5': 'ORAS5 (ORCA-DL)', 
    'om4': 'OM4', 
    'atlas': 'Atlas', 
    'orca_dl': 'ORCA-DL', 
    'samudra_thermo_dynamic': 'Samudra Thermo+Dynamic', 
    'samudra_thermo': 'Samudra Thermo', 
    'pers': 'Persistence', 
}

MODEL_COLORS = {
    'rea': 'dodgerblue', 
    'clm': 'purple', 
    'oras5': 'magenta', 
    'om4': 'lime', 
    'atlas': 'tab:red', 
    'orca_dl': 'tab:orange', 
    'samudra_thermo_dynamic': 'mediumorchid', 
    'samudra_thermo': 'forestgreen', 
    'pers': 'tab:blue', 
}

def __getattr__(name):
    """
    This leverages a trick of Python 3.7+ that allows to
    intercept the access to the variables within a module.
    """
    # Maps the old names to the macros properties
    mapping = {
        'MEAN_DATA_VARS': lambda: macros.stats['mean'],
        'STD_DATA_VARS': lambda: macros.stats['std'],
        'ATMOS_3D_VARS': 'atmos_3d',
        'ATMOS_2D_VARS': 'atmos_2d',
        'OCEAN_3D_VARS': 'ocean_3d',
        'OCEAN_2D_VARS': 'ocean_2d',
        'ALL_DATA_VARS': 'all_vars',
        'ATMOS_3D_IDS': 'atmos_3d_ids',
        'ATMOS_2D_IDS': 'atmos_2d_ids',
        'OCEAN_3D_IDS': 'ocean_3d_ids',
        'OCEAN_2D_IDS': 'ocean_2d_ids',
        'ALL_DRIVERS': 'all_drivers',
    }

    if name in mapping:
        attr = mapping[name]
        if callable(attr):
            return attr()
        return getattr(macros, attr)
    
    raise AttributeError(f"module {__name__} has no attribute {name}")