import os
import sys
import traci

# 1. SUMO_HOME 환경변수 체크
if 'SUMO_HOME' in os.environ:
    print(f"OK: SUMO_HOME found at {os.environ['SUMO_HOME']}")
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("FAIL: Please declare environment variable 'SUMO_HOME'")

# 2. SUMO 실행 명령어
# 본인의 .sumocfg 파일명이 맞는지 확인하세요!
sumoCmd = ["sumo", "-c", "osm.sumocfg"] 

try:
    print("Starting SUMO via TraCI...")
    traci.start(sumoCmd)
    print("SUCCESS: SUMO has started successfully!")

    # 5 스텝만 돌려봄
    for step in range(5):
        traci.simulationStep()
        print(f"Step {step} done.")

    traci.close()
    print("Test Complete.")

except Exception as e:
    print(f"Error: {e}")