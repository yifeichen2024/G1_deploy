#!/usr/bin/env bash
# chmod +x run_motion_loop.sh
# ./run_motion_loop.sh
# 目标：掉线或 CTRL-C 后 2 s 自动重启，日志按日期保存

SCRIPT="/home/unitree/g1_deploy/g1_highlevel_motion_controller.py"
TRAJ_DIR="/home/unitree/g1_deploy/traj"          # 所有 npz 放这里
IP="127.0.0.1"                                   # 本机 DDS
SPEED="1.0"
LOG_DIR="/home/unitree/g1_logs"
mkdir -p "$LOG_DIR"

while true; do
    ts=$(date +"%Y%m%d_%H%M%S")
    python3 "$SCRIPT" "$TRAJ_DIR"/*.npz --ip "$IP" --speed "$SPEED" \
        2>&1 | tee "$LOG_DIR/motion_$ts.log"
    echo "=== script exited, restart in 2 s ==="
    sleep 2
done
