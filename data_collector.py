import os
import sys
import traci
import pandas as pd

# ==========================================
# 설정: 본인의 파일 이름에 맞게 수정하세요
# 방금 test_env.py가 성공했으므로 'osm.sumocfg'가 맞습니다.
# ==========================================
SUMO_CONFIG_FILE = "osm.sumocfg" 

def check_sumo_env():
    """SUMO 환경변수 확인"""
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("에러: 'SUMO_HOME' 환경변수가 없습니다.")

def run_collection():
    check_sumo_env()
    
    # GUI 모드로 실행 (눈으로 확인 가능)
    # 시뮬레이션 창이 뜨면 상단 'Play' 버튼을 눌러야 진행됩니다.
    # 만약 창 없이 빠르게 데이터만 뽑고 싶다면 "sumo"로 바꾸세요.
    sumoCmd = ["sumo-gui", "-c", SUMO_CONFIG_FILE] 
    
    print(">>> 시뮬레이션을 시작합니다 (창이 뜨면 'Play' 버튼을 누르세요)...")
    traci.start(sumoCmd)
    
    # 데이터 저장소
    collected_data = []
    
    step = 0
    max_steps = 3600  # 3600초(1시간) 동안 데이터 수집
    
    while step < max_steps:
        # 1. 시뮬레이션 1초 진행
        traci.simulationStep()
        
        # 2. 현재 도로 위 모든 차량 ID 가져오기
        vehicle_ids = traci.vehicle.getIDList()
        
        # (옵션) 차량이 한 대도 없으면 대기하거나 종료
        # if step > 100 and len(vehicle_ids) == 0:
        #     print("차량이 더 이상 없어 종료합니다.")
        #     break
            
        for veh_id in vehicle_ids:
            # 좌표 (x, y) 수집
            x, y = traci.vehicle.getPosition(veh_id)
            # 속도 (m/s) 수집
            speed = traci.vehicle.getSpeed(veh_id)
            
            # 리스트에 추가 [시간, 차량ID, x, y, 속도]
            collected_data.append([step, veh_id, x, y, speed])
            
        step += 1
    
    traci.close()
    print(f">>> 수집 완료! 총 {len(collected_data)}개의 데이터 포인트")
    return collected_data

if __name__ == "__main__":
    # 데이터 수집 실행
    try:
        data = run_collection()
        
        # CSV 저장
        if len(data) > 0:
            df = pd.DataFrame(data, columns=['Step', 'VehicleID', 'x', 'y', 'Speed'])
            df.to_csv("mobility_dataset.csv", index=False)
            print(">>> 성공! 'mobility_dataset.csv' 파일로 저장되었습니다.")
            print(df.head()) # 데이터 미리보기
        else:
            print(">>> 경고: 수집된 데이터가 없습니다. 시뮬레이션 상에 차량이 생성되었는지 확인하세요.")
            
    except Exception as e:
        print(f"에러 발생: {e}")