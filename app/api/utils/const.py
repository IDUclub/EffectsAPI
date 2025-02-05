import os
from iduconfig import Config

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