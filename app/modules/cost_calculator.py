from datetime import datetime
from app.core.config import FREIGHT_RATES, NIGHT_SURCHARGE_RATE, NIGHT_START_HOUR, NIGHT_END_HOUR

class CostCalculator:
    
    @staticmethod
    def get_base_cost(distance_km: float, car_type: str) -> int:
        """
        거리표 기반 기본 요금 조회
        """
        sorted_distances = sorted(FREIGHT_RATES.keys())
        
        target_distance = 0
        for dist in sorted_distances:
            if distance_km <= dist:
                target_distance = dist
                break
        
        # 600km 초과 시 로직
        if target_distance == 0:
            max_dist = sorted_distances[-1]
            base = FREIGHT_RATES[max_dist].get(car_type, 0)
            extra_km = distance_km - max_dist
            # 600km 초과 시 10km당 1만원 추가 (가정)
            extra_cost = int(extra_km / 10) * 10000 
            return base + extra_cost

        return FREIGHT_RATES[target_distance].get(car_type, 0)

    @staticmethod
    def apply_night_surcharge(base_cost: int) -> dict:
        """심야 할증 적용"""
        now = datetime.now()
        hour = now.hour
        
        is_night = (hour >= NIGHT_START_HOUR) or (hour < NIGHT_END_HOUR)
        surcharge = 0
        
        if is_night:
            surcharge = int(base_cost * (NIGHT_SURCHARGE_RATE - 1))
            
        return {
            "is_night": is_night,
            "surcharge_amount": surcharge,
            "final_cost": base_cost + surcharge
        }