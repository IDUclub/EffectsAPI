from blocksnet.enums import LandUse


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
