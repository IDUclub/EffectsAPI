import os
from iduconfig import Config
from blocksnet.enums import LandUse


config = Config()

API_TITLE = 'Effects API'
API_DESCRIPTION = 'API for assessing territory transformation effects'
EVALUATION_RESPONSE_MESSAGE = 'Evaluation started'
DEFAULT_CRS = 4326
NORMATIVES_YEAR = 2024

if config.get("DATA_PATH"):
  DATA_PATH =  os.path.abspath('data')
else:
  # DATA_PATH = 'app/data'
  raise Exception('No DATA_PATH in env file')
if config.get("URBAN_API"):
  URBAN_API = config.get("URBAN_API")
else:
  # URBAN_API = 'http://10.32.1.107:5300'
  raise Exception('No URBAN_API in env file')


LU_SHARE = 0.5
SQ_M_IN_HA = 10_000


mapping = {
  'residential': LandUse.RESIDENTIAL,
  'recreation': LandUse.RECREATION,
  'special': LandUse.SPECIAL,
  'industrial': LandUse.INDUSTRIAL,
  'agriculture': LandUse.AGRICULTURE,
  'transport': LandUse.TRANSPORT,
  'business': LandUse.BUSINESS,
  'residential_individual': LandUse.RESIDENTIAL,
  'residential_lowrise': LandUse.RESIDENTIAL,
  'residential_midrise': LandUse.RESIDENTIAL,
  'residential_multistorey': LandUse.RESIDENTIAL,
 }


residential_mapping = {
  'residential': (250,350),
  'residential_individual': (30,35),
  'residential_lowrise': (50,150),
  'residential_midrise': (250,350),
  'residential_multistorey': (350,450),
}

DUMMY_BUILDING_PARAMS = {
  'id' : -1,
  'build_floor_area' : 0,
  'living_area' : 0,
  'non_living_area' : 0,
  'footprint_area' : 0,
  'number_of_floors' : 1,
}
