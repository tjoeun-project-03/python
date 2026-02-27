from datetime import datetime
from app.core.config import FREIGHT_RATES

class CostCalculator:
    
    @staticmethod
    def get_base_cost(distance_km: float, car_type: str) -> int:
        """거리표 기반 기본 요금 조회"""
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
            extra_cost = int(extra_km / 10) * 10000 
            return base + extra_cost

        return FREIGHT_RATES[target_distance].get(car_type, 0)

    @staticmethod
    def apply_dynamic_surcharges(base_cost: int, holiday_val: int, night_val: int) -> dict:
        """
        [합연산 다중 할증 로직 (공휴일 + 야간)]
        DB에서 가져온 할증률을 인자로 전달받아 계산만 수행합니다.
        """
        total_surcharge_rate = holiday_val + night_val
        surcharge_amount = int(base_cost * (total_surcharge_rate / 100))
        final_cost = base_cost + surcharge_amount
        
        return {
            "base_cost": base_cost,
            "holiday_rule_pct": holiday_val,
            "night_rule_pct": night_val,
            "total_surcharge_amount": surcharge_amount,
            "final_cost": final_cost
        }