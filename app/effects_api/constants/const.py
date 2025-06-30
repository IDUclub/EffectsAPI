from blocksnet.enums import LandUse

# UrbanDB to Blocksnet land use types mapping
LAND_USE_RULES = {
    "residential": LandUse.RESIDENTIAL,
    "recreation": LandUse.RECREATION,
    "special": LandUse.SPECIAL,
    "industrial": LandUse.INDUSTRIAL,
    "agriculture": LandUse.AGRICULTURE,
    "transport": LandUse.TRANSPORT,
    "business": LandUse.BUSINESS,
    "residential_individual": LandUse.RESIDENTIAL,
    "residential_lowrise": LandUse.RESIDENTIAL,
    "residential_midrise": LandUse.RESIDENTIAL,
    "residential_multistorey": LandUse.RESIDENTIAL,
}


# TODO add map autogeneration
SERVICE_TYPES_MAPPING = {
    # basic
    1: "park",
    21: "kindergarten",
    22: "school",
    28: "polyclinic",
    34: "pharmacy",
    61: "cafe",
    66: "pitch",
    68: None,  # спортивный зал
    74: "playground",
    78: "police",
    # additional
    30: None,  # стоматология
    35: "hospital",
    50: "museum",
    56: "cinema",
    57: "mall",
    59: "stadium",
    62: "restaurant",
    63: "bar",
    77: None,  # скейт парк
    79: None,  # пожарная станция
    80: "train_station",
    89: "supermarket",
    99: None,  # пункт выдачи
    100: "bank",
    107: "veterinary",
    143: "sanatorium",
    # comfort
    5: "beach",
    27: "university",
    36: None,  # роддом
    48: "library",
    51: "theatre",
    91: "market",
    93: None,  # одежда и обувь
    94: None,  # бытовая техника
    95: None,  # книжный магазин
    96: None,  # детские товары
    97: None,  # спортивный магазин
    108: None,  # зоомагазин
    110: "hotel",
    114: "religion",  # религиозный объект
    # others
    26: None,  # ССУЗ
    32: None,  # женская консультация
    39: None,  # скорая помощь
    40: None,  # травматология
    45: "recruitment",
    47: "multifunctional_center",
    55: "zoo",
    65: "bakery",
    67: "swimming_pool",
    75: None,  # парк аттракционов
    81: "train_building",
    82: "aeroway_terminal",  # аэропорт??
    86: "bus_station",
    88: "subway_entrance",
    102: "lawyer",
    103: "notary",
    109: "dog_park",
    111: "hostel",
    112: None,  # база отдыха
    113: None,  # памятник
}

# Rules for agregating building properties from UrbanDB API
BUILDINGS_RULES = {
    "number_of_floors": [
        ["floors"],
        ["properties", "storeys_count"],
        ["properties", "osm_data", "building:levels"],
    ],
    "footprint_area": [
        ["building_area_official"],
        ["building_area_modeled"],
        ["properties", "area_total"],
    ],
    "build_floor_area": [
        ["properties", "area_total"],
    ],
    "living_area": [
        ["properties", "living_area_official"],
        ["properties", "living_area"],
        ["properties", "living_area_modeled"],
    ],
    "non_living_area": [
        ["properties", "area_non_residential"],
    ],
    "population": [["properties", "population_balanced"]],
}

LIVING_BUILDINGS_ID = 4
ROADS_ID = 26
WATER_ID = 4