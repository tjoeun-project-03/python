# 견적 로직

import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.modules.tmap_client import TMapClient
from app.modules.cost_calculator import CostCalculator
from app.core.config import TMAP_API_KEY
# import oracledb  # 💡 실제 오라클 연동 시 주석을 해제하세요

router = APIRouter()
tmap_client = TMapClient(TMAP_API_KEY)

class EstimateRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    car_type: str

def fetch_latest_surcharge_from_db():
    """
    [TODO: 실시간 Oracle DB 조회 함수]
    - 견적 API가 호출될 때마다 매번 실행되어 최신 할증률을 가져옵니다.
    """
    # try:
    #     connection = oracledb.connect(user="계정", password="비번", dsn="호스트:포트/서비스명")
    #     cursor = connection.cursor()
    #     # 최신 설정 1개만 가져오는 Oracle 쿼리
    #     query = "SELECT HOLIDAY_RULE, NIGHT_RULE FROM PRICING_RULES ORDER BY ID DESC FETCH FIRST 1 ROWS ONLY"
    #     cursor.execute(query)
    #     row = cursor.fetchone()
    #     connection.close()
    #     
    #     if row:
    #         return int(row[0]), int(row[1]) # (holiday_rule, night_rule)
    # except Exception as e:
    #     print(f"DB Fetch Error: {e}")
    #     return 0, 0 # 에러 발생 시 기본값 0으로 fallback 방어 로직
    
    # [임시] DB 연동 전까지 프론트엔드 테스트를 위해 하드코딩된 값을 리턴합니다.
    return 0, 0

@router.post("/estimate")
async def calculate_estimate(req: EstimateRequest):
    try:
        route_data = await tmap_client.get_route_data(
            req.start_lat, req.start_lng, 
            req.end_lat, req.end_lng, 
            req.car_type
        )
        distance_km = route_data["total_distance_m"] / 1000
        
        # 1. 기본 요금 산출
        base_cost = CostCalculator.get_base_cost(distance_km, req.car_type)
        if base_cost == 0:
            raise HTTPException(status_code=400, detail="요금표 오류")

        # 2. 매번 실시간으로 DB 찔러서 최신 할증률 가져오기
        holiday_rule, night_rule = fetch_latest_surcharge_from_db()

        # 3. DB에서 빼온 값을 계산기로 던져서 합연산
        cost_info = CostCalculator.apply_dynamic_surcharges(base_cost, holiday_rule, night_rule)

        return {
            "success": True,
            "data": {
                "distance_km": round(distance_km, 1),
                "duration_min": round(route_data["total_time_sec"] / 60),
                "base_cost": cost_info["base_cost"],
                "total_surcharge_amount": cost_info["total_surcharge_amount"],
                "total_cost": cost_info["final_cost"]
            }
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))