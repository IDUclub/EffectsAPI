from blocksnet.analysis.indicators.socio_economic import (
    DemographicIndicator,
    EcologicalIndicator,
    EconomicIndicator,
    EngineeringIndicator,
    GeneralIndicator,
    SettlementIndicator,
    SocialCountIndicator,
    SocialIndicator,
    SocialProvisionIndicator,
    TransportIndicator,
)
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
    1: "park",
    5: "beach",
    21: "kindergarten",
    22: "school",
    23: None,  # доп образование
    24: None,  # доп образование
    26: "college",
    27: "university",
    28: "polyclinic",
    29: None,  # детская поликлиника
    30: None,  # стоматология
    31: None,  # фельдшерско-акушерский пункт
    32: None,  # женская консультация
    34: "pharmacy",
    35: "hospital",
    36: None,  # роддом
    37: None,  # детская больница
    38: None,  # хоспис
    39: None,  # скорая помощь
    40: None,  # травматология
    41: None,  # морг
    42: None,  # диспансер
    43: None,  # центры соц обслуживания
    44: "social_facility",  # дом престарелых
    45: "recruitment",
    46: None,  # детский дом
    47: "multifunctional_center",
    48: "library",
    49: None,  # дворцы культуры
    50: "museum",
    51: "theatre",
    53: None,  # концертный зал
    55: "zoo",
    56: "cinema",
    57: "mall",
    59: "stadium",
    60: None,  # ледовая арена
    61: "cafe",
    62: "restaurant",
    63: "bar",
    64: "cafe",
    65: "bakery",
    66: "pitch",
    67: "swimming_pool",
    68: None,  # спортивный зал
    69: None,  # каток
    70: None,  # футбольное поле
    72: None,  # эко тропа
    74: "playground",
    75: None,  # парк аттракционов
    77: None,  # скейт парк
    78: "police",
    79: None,  # пожарная станция
    80: "train_station",
    81: "train_building",
    82: "aeroway_terminal",
    84: "fuel",
    86: "bus_station",
    88: "subway_entrance",
    89: "supermarket",
    91: "market",
    93: None,  # одежда и обувь
    94: None,  # бытовая техника
    95: None,  # книжный магазин
    96: None,  # детские товары
    97: None,  # спортивный магазин
    98: "post",
    99: None,  # пункт выдачи
    100: "bank",
    102: "lawyer",
    103: "notary",
    107: "veterinary",
    108: None,  # зоомагазин
    109: "dog_park",
    110: "hotel",
    111: "hostel",
    112: None,  # база отдыха
    113: None,  # памятник
    114: "religion",  # религиозный объект
    # электростанции -- start
    118: "substation",  # Атомная электростанция
    119: "substation",  # Гидро-электростанция
    120: "substation",  # Тепловая электростанция
    # электростанции -- end
    124: "water_works",
    # водоочистные сооружения -- start
    126: "wastewater_plant",  # Сооружения для очистки воды
    128: "wastewater_plant",  # Водоочистные сооружения
    # водоочистные сооружения -- end
    143: "sanatorium",
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

# For each Infrastructure_type we will also add a weighting factor to give preference to the basic service, and another switch for capabilities.
INFRASTRUCTURES_WEIGHTS = {"basic": 0.5714, "additional": 0.2857, "comfort": 0.1429}

# Mapping for translation of english provision properties
COL_RU = {
    "demand": "Спрос",
    "capacity": "Емкость сервисов",
    "demand_left": "Неудовлетворенный спрос",
    "demand_within": "Спрос в пределах нормативной доступности",
    "demand_without": "Спрос за пределами нормативной доступности",
    "capacity_left": "Оставшаяся емкость сервисов",
    "capacity_within": "Емкость сервисов в пределах нормативной доступности",
    "capacity_without": "Емкость сервисов за пределами нормативной доступности",
    "provision_strong": "Обеспеченность сервисами",
}
# ID of living building physical_object_type_id
LIVING_BUILDINGS_ID = 4

# ID of road physical_object_function_id
ROADS_ID = 26

# ID of water objects physical_object_function_id
WATER_ID = 4

# Maximum number of function evaluations
MAX_EVALS = 1000

# Maximum number of runs for optimization
MAX_RUNS = 1000

PRED_VALUE_RU = {
    "urban": "Жилой или смешанный (бизнес)",
    "industrial": "Промышленный",
    "non_urban": "Рекреация",
}

PROB_COLS_EN_TO_RU = {
    "prob_urban": "Вероятность жилого или бизнес видов использования, %",
    "prob_non_urban": "Вероятность рекреационного вида использования, %",
    "prob_industrial": "Вероятность промышленного вида использования, %",
}

SOCIAL_INDICATORS_MAPPING = {
    SocialIndicator.EXTRACURRICULAR: [23, 24],
    SocialIndicator.AMBULANCE: [39, 40],
    SocialIndicator.SPECIAL_MEDICAL: [41],
    SocialIndicator.PREVENTIVE_MEDICAL: [42],
    SocialIndicator.GYM: [68],
    SocialIndicator.ORPHANAGE: [46],
    SocialIndicator.SOCIAL_SERVICE_CENTER: [43],
    SocialIndicator.CULTURAL_CENTER: [49],
    SocialIndicator.CONCERT_HALL: [53],
    SocialIndicator.ICE_ARENA: [60],
    SocialIndicator.ECO_TRAIL: [72],
    SocialIndicator.FIRE_STATION: [79],
    SocialIndicator.TOURIST_BASE: [112],
}

INDICATORS_MAPPING = {
    # общие
    GeneralIndicator.AREA: 4,
    GeneralIndicator.URBANIZATION: 16,
    # демография
    DemographicIndicator.POPULATION: 1,
    DemographicIndicator.DENSITY: 37,
    # транспорт
    TransportIndicator.ROAD_NETWORK_DENSITY: 60,
    TransportIndicator.SETTLEMENTS_CONNECTIVITY: 59,
    TransportIndicator.ROAD_NETWORK_LENGTH: 65,
    TransportIndicator.FUEL_STATIONS_COUNT: 71,
    TransportIndicator.AVERAGE_FUEL_STATION_ACCESSIBILITY: 72,
    TransportIndicator.RAILWAY_STOPS_COUNT: 75,
    TransportIndicator.AVERAGE_RAILWAY_STOP_ACCESSIBILITY: 76,
    TransportIndicator.AIRPORTS_COUNT: 78,
    TransportIndicator.AVERAGE_AIRPORT_ACCESSIBILITY: None,  # TODO Средняя доступность аэропортов (без разделения на международные и местные)
    # инженерная инфраструктура
    EngineeringIndicator.INFRASTRUCTURE_OBJECT: 88,
    EngineeringIndicator.SUBSTATION: 89,
    EngineeringIndicator.WATER_WORKS: 90,
    EngineeringIndicator.WASTEWATER_PLANT: 91,
    EngineeringIndicator.RESERVOIR: 92,
    EngineeringIndicator.GAS_DISTRIBUTION: 93,
    # социальная инфраструктура
    # образование
    SocialCountIndicator.KINDERGARTEN: 309,
    SocialProvisionIndicator.KINDERGARTEN: 207,  # Обеспеченность детскими садами
    SocialCountIndicator.SCHOOL: 338,
    SocialProvisionIndicator.SCHOOL: 208,  # Обеспеченность школами
    SocialCountIndicator.COLLEGE: 310,
    SocialProvisionIndicator.COLLEGE: None,  # FIXME Обеспеченность образовательными учреждениями СПО (нет их)
    SocialCountIndicator.UNIVERSITY: 311,
    SocialProvisionIndicator.UNIVERSITY: 350,
    SocialCountIndicator.EXTRACURRICULAR: None,  # FIXME Организации дополнительного образования детей (нет их)
    SocialProvisionIndicator.EXTRACURRICULAR: None,  # FIXME Обеспеченность организациями дополнительного образования детей (нет их)
    # здравоохранение
    SocialCountIndicator.HOSPITAL: 341,
    SocialProvisionIndicator.HOSPITAL: 361,  #  Обеспеченность больницами
    SocialCountIndicator.POLYCLINIC: 342,
    SocialProvisionIndicator.POLYCLINIC: 362,  # Обеспеченность поликлиниками
    SocialCountIndicator.AMBULANCE: 343,
    SocialProvisionIndicator.AMBULANCE: None,  # FIXME Обеспеченность объектами скорой медицинской помощи
    SocialCountIndicator.SANATORIUM: 312,
    SocialProvisionIndicator.SANATORIUM: None,  # FIXME Обеспеченность объектами санаторного назначения
    SocialCountIndicator.SPECIAL_MEDICAL: None,  # FIXME Медицинские учреждения особого типа
    SocialProvisionIndicator.SPECIAL_MEDICAL: None,  # FIXME Обеспеченность медицинскими учреждениями особого типа
    SocialCountIndicator.PREVENTIVE_MEDICAL: 346,
    SocialProvisionIndicator.PREVENTIVE_MEDICAL: None,  # FIXME Обеспеченность лечебно-профилактическими медицинскими учреждениями
    SocialCountIndicator.PHARMACY: 345,
    SocialProvisionIndicator.PHARMACY: 213,  # Обеспеченность аптеками
    # спорт
    SocialCountIndicator.GYM: 313,
    SocialProvisionIndicator.GYM: 243,  # Обеспеченность спортзалами ОП / фитнес-центрами
    SocialCountIndicator.SWIMMING_POOL: 314,
    SocialProvisionIndicator.SWIMMING_POOL: 245,  # Обеспеченность ФОК / бассейнами
    SocialCountIndicator.PITCH: 340,
    SocialProvisionIndicator.PITCH: 357,
    SocialCountIndicator.STADIUM: 315,
    SocialProvisionIndicator.STADIUM: 356,
    # социальная помощь
    SocialCountIndicator.ORPHANAGE: 316,
    SocialProvisionIndicator.ORPHANAGE: None,  # FIXME Обеспеченность детскими домами-интернатами
    SocialCountIndicator.SOCIAL_FACILITY: 317,
    SocialProvisionIndicator.SOCIAL_FACILITY: None,  # FIXME Обеспеченность домами престарелых
    SocialCountIndicator.SOCIAL_SERVICE_CENTER: 318,
    SocialProvisionIndicator.SOCIAL_SERVICE_CENTER: None,  # FIXME Обеспеченность центрами социального обслуживания
    # услуги
    SocialCountIndicator.POST: 319,
    SocialProvisionIndicator.POST: 247,  # Обеспеченность пунктами доставки / почтовыми отделениями
    SocialCountIndicator.BANK: 320,
    SocialProvisionIndicator.BANK: 250,  # Обеспеченность отделениями банков
    SocialCountIndicator.MULTIFUNCTIONAL_CENTER: 321,
    SocialProvisionIndicator.MULTIFUNCTIONAL_CENTER: 351,
    # культура и отдых
    SocialCountIndicator.LIBRARY: 322,
    SocialProvisionIndicator.LIBRARY: 232,  # Обеспеченность медиатеками / библиотеками
    SocialCountIndicator.MUSEUM: 323,
    SocialProvisionIndicator.MUSEUM: 352,
    SocialCountIndicator.THEATRE: 324,
    SocialProvisionIndicator.THEATRE: 353,
    SocialCountIndicator.CULTURAL_CENTER: 325,
    SocialProvisionIndicator.CULTURAL_CENTER: 231,  # Обеспеченность комьюнити-центрами / домами культуры
    SocialCountIndicator.CINEMA: 326,
    SocialProvisionIndicator.CINEMA: 354,
    SocialCountIndicator.CONCERT_HALL: 327,
    SocialProvisionIndicator.CONCERT_HALL: None,  # FIXME Обеспеченность концертными залами
    # SocialCountIndicator.STADIUM : 315, ПОВТОР
    # SocialProvisionIndicator.STADIUM : 356, ПОВТОР
    SocialCountIndicator.ICE_ARENA: 328,
    SocialProvisionIndicator.ICE_ARENA: None,  # FIXME Обеспеченность ледовыми аренами
    SocialCountIndicator.MALL: 329,
    SocialProvisionIndicator.MALL: 355,
    SocialCountIndicator.PARK: 330,
    SocialProvisionIndicator.PARK: 238,  # Обеспеченность парками
    SocialCountIndicator.BEACH: 331,
    SocialProvisionIndicator.BEACH: None,  # FIXME Обеспеченность пляжами
    SocialCountIndicator.ECO_TRAIL: 332,
    SocialProvisionIndicator.ECO_TRAIL: None,  # FIXME Обеспеченность экологическими тропами
    # безопасность
    SocialCountIndicator.FIRE_STATION: 333,
    SocialProvisionIndicator.FIRE_STATION: 260,  # Обеспеченность пожарными депо
    SocialCountIndicator.POLICE: 334,
    SocialProvisionIndicator.POLICE: 258,  # Обеспеченность пунктами полиции
    # туризм
    SocialCountIndicator.HOTEL: 335,
    SocialProvisionIndicator.HOTEL: 358,
    SocialCountIndicator.HOSTEL: 336,
    SocialProvisionIndicator.HOSTEL: 359,
    SocialCountIndicator.TOURIST_BASE: 337,
    SocialProvisionIndicator.TOURIST_BASE: 360,
    SocialCountIndicator.CATERING: 344,
    SocialProvisionIndicator.CATERING: 226,  # Обеспеченность кафе / кофейнями
}

SPEED = 5 * 1_000 / 60
