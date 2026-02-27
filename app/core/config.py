import os
from dotenv import load_dotenv

# .env 파일 로드 (상위 폴더를 거슬러 올라가며 찾음)
load_dotenv()

TMAP_API_KEY = os.getenv("TMAP_API_KEY")

# 1. 거리별 표준 운임표 (단위: 원)
FREIGHT_RATES = {
    20:  {"1t": 40000, "1.4t": 50000, "2.5t": 70000, "3.5t": 80000, "5t": 90000},
    30:  {"1t": 50000, "1.4t": 60000, "2.5t": 80000, "3.5t": 90000, "5t": 100000},
    50:  {"1t": 60000, "1.4t": 70000, "2.5t": 90000, "3.5t": 100000, "5t": 120000},
    70:  {"1t": 70000, "1.4t": 80000, "2.5t": 100000, "3.5t": 110000, "5t": 130000},
    100: {"1t": 90000, "1.4t": 100000, "2.5t": 120000, "3.5t": 130000, "5t": 150000},
    200: {"1t": 130000, "1.4t": 140000, "2.5t": 180000, "3.5t": 190000, "5t": 210000},
    300: {"1t": 170000, "1.4t": 180000, "2.5t": 250000, "3.5t": 260000, "5t": 280000},
    400: {"1t": 200000, "1.4t": 210000, "2.5t": 280000, "3.5t": 300000, "5t": 320000},
    500: {"1t": 250000, "1.4t": 260000, "2.5t": 320000, "3.5t": 340000, "5t": 360000},
    600: {"1t": 300000, "1.4t": 310000, "2.5t": 380000, "3.5t": 400000, "5t": 420000},
}

# 2. 차종별 TMAP API 요청용 스펙
VEHICLE_SPECS = {
    "1t": {
        "truckType": 1, 
        "truckWidth": 160,
        "truckHeight": 220,
        "truckLength": 280,
        "truckWeight": 1000,
        "truckTotalWeight": 2600
    },
    "1.4t": {
        "truckType": 1, 
        "truckWidth": 160,
        "truckHeight": 230,
        "truckLength": 310,
        "truckWeight": 1400,
        "truckTotalWeight": 3000
    },
    "2.5t": {
        "truckType": 1, 
        "truckWidth": 180,
        "truckHeight": 260,
        "truckLength": 420,
        "truckWeight": 4000,
        "truckTotalWeight": 6500
    },
    "3.5t": {
        "truckType": 1, 
        "truckWidth": 200,
        "truckHeight": 270,
        "truckLength": 440,
        "truckWeight": 6000,
        "truckTotalWeight": 9500
    },
    "5t": {
        "truckType": 1, 
        "truckWidth": 230,
        "truckHeight": 320,
        "truckLength": 620,
        "truckWeight": 6500,
        "truckTotalWeight": 11500
    }
}
