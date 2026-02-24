# 견적 로직

import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.modules.tmap_client import TMapClient
from app.modules.cost_calculator import CostCalculator
from app.core.config import TMAP_API_KEY

router = APIRouter()

tmap_client = TMapClient(TMAP_API_KEY)

class EstimateRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    car_type: str

@router.post("/estimate")
async def calculate_estimate(req: EstimateRequest):
    try:
        route_data = await tmap_client.get_route_data(
            req.start_lat, req.start_lng, 
            req.end_lat, req.end_lng, 
            req.car_type
        )
        distance_km = route_data["total_distance_m"] / 1000
        
        base_cost = CostCalculator.get_base_cost(distance_km, req.car_type)
        if base_cost == 0:
            raise HTTPException(status_code=400, detail="요금표 오류")

        surcharge_info = CostCalculator.apply_night_surcharge(base_cost)

        return {
            "success": True,
            "data": {
                "distance_km": round(distance_km, 1),
                "duration_min": round(route_data["total_time_sec"] / 60),
                "total_cost": surcharge_info["final_cost"],
                "is_night": surcharge_info["is_night"]
            }
        }
    except Exception as e:
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}