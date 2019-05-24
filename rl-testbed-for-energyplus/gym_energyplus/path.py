"""
Common path string
"""
import os
import platform

if platform.system() == 'Darwin':
    energyplus_instdir = '/Applications'
elif platform.system() == 'Linux':
    energyplus_instdir = '/usr/local'
else:
    raise ValueError('Unsupported system {}'.format(platform.system()))

ENERGYPLUS_VERSION = "8-8-0"
ENERGYPLUS_DIR = os.path.join(energyplus_instdir, 'EnergyPlus-{}'.format(ENERGYPLUS_VERSION))

WEATHER_DIR = os.path.join(ENERGYPLUS_DIR, 'WeatherData')

energyplus_bin_path = os.path.join(ENERGYPLUS_DIR, 'energyplus')

ENERGYPLUS_WEATHER_dict = {
    'sf': 'USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw',
    'golden': 'USA_CO_Golden-NREL.724666_TMY3.epw',
    'chicago': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
    'sterling': 'USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw',
}

ENERGYPLUS_MODEL_dict = {
    'temp': '2ZoneDataCenterHVAC_wEconomizer_Temp.idf',
    'temp_fan': '2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf'
}


def get_weather_filepath(location):
    assert location in ENERGYPLUS_WEATHER_dict.keys(), 'Unknown location {}'.format(location)
    return os.path.join(WEATHER_DIR, ENERGYPLUS_WEATHER_dict[location])


MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../EnergyPlus/Model')


def get_model_filepath(control_mode):
    assert control_mode in ENERGYPLUS_MODEL_dict.keys(), 'Unknown control mode {}'.format(control_mode)
    return os.path.join(MODEL_DIR, ENERGYPLUS_MODEL_dict[control_mode])
