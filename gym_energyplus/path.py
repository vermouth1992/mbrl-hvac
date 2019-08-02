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
    'SF': 'USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw',
    'Golden': 'USA_CO_Golden-NREL.724666_TMY3.epw',
    'Chicago': 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
    'Sterling': 'USA_VA_Sterling-Washington.Dulles.Intl.AP.724030_TMY3.epw',
    'Tampa': 'USA_FL_Tampa.Intl.AP.722110_TMY3.epw',
    'Nanjing': 'CHN_Jiangsu.Nanjing.582380_CSWD.epw',
}

ENERGYPLUS_MODEL_dict = {
    'temp': '2ZoneDataCenterHVAC_wEconomizer_Temp.idf',
    'temp_fan': '2ZoneDataCenterHVAC_wEconomizer_Temp_Fan.idf'
}


def get_weather_filepath(locations):
    if isinstance(locations, str):
        locations = [locations]

    output = []
    for location in locations:
        assert location in ENERGYPLUS_WEATHER_dict.keys(), 'Unknown location {}'.format(location)
        output.append(os.path.join(WEATHER_DIR, ENERGYPLUS_WEATHER_dict[location]))
    return ','.join(output)


MODEL_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../EnergyPlus/Model')


def get_model_filepath(control_mode):
    assert control_mode in ENERGYPLUS_MODEL_dict.keys(), 'Unknown control mode {}'.format(control_mode)
    return os.path.join(MODEL_DIR, ENERGYPLUS_MODEL_dict[control_mode])
