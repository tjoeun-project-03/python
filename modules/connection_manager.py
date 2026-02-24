# 파일명: app/modules/connection_manager.py
# 역할: WebSocket 연결 상태를 관리하고 메시지를 중계(Broadcast)함

from typing import List, Dict
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        # { "주문ID": [접속자1, 접속자2, ...] } 형태로 저장
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        """사용자(화주/차주)가 특정 주문방(room_id)에 입장"""
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        """연결 끊김 처리"""
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)
            # 방에 아무도 없으면 방 삭제 (메모리 관리)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, message: dict, room_id: str):
        """특정 방에 있는 모든 사람에게 위치 데이터 전송"""
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                # 연결이 살아있는지 확인 후 전송
                try:
                    await connection.send_json(message)
                except Exception:
                    # 에러나면 연결 끊기 처리 등 추가 로직 가능
                    pass

# 전역에서 쓸 수 있도록 인스턴스 생성
manager = ConnectionManager()