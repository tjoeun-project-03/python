import httpx
from app.core.config import VEHICLE_SPECS 

class TMapClient:
    def __init__(self, api_key: str):
        self.api_key = api_key 
        self.base_url = "https://apis.openapi.sk.com/tmap/truck/routes?version=1"

    async def get_route_data(self, start_lat, start_lng, end_lat, end_lng, car_type):
        """
        TMAP 화물차 경로 API 호출
        """
        # 1. 차종 스펙 가져오기
        specs = VEHICLE_SPECS.get(car_type, VEHICLE_SPECS["1t"])

        payload = {
            "startX": start_lng,
            "startY": start_lat,
            "endX": end_lng,
            "endY": end_lat,
            "reqCoordType": "WGS84GEO",
            "resCoordType": "WGS84GEO",
            "truckType": specs["truckType"],
            "truckWidth": specs["truckWidth"],
            "truckHeight": specs["truckHeight"],
            "truckLength": specs["truckLength"],
            "truckWeight": specs["truckWeight"],
            "truckTotalWeight": specs["truckTotalWeight"],
            "trafficInfo": "Y"
        }

        headers = {
            "appKey": self.api_key,
            "Content-Type": "application/json"
        }

        # [수정됨] 장거리 경로 계산을 위해 타임아웃을 30초로 늘림 (기본 5초)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            
            # 에러 발생 시 터미널에 상세 내용 출력
            if response.status_code != 200:
                print(f"❌ TMAP API Error: {response.text}")
                raise Exception(f"TMAP API Error Code: {response.status_code}")

            data = response.json()
            properties = data["features"][0]["properties"]
            
            return {
                "total_distance_m": properties["totalDistance"],
                "total_time_sec": properties["totalTime"]
            }