#!/bin/bash

# 获取当前目录
current_dir=$(pwd)
echo "当前目录为：$current_dir"

# 获取当前时间戳
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo "当前时间戳为：$timestamp"

# 创建包含时间戳的log文件
log_file="byte_$timestamp.log"
touch "$log_file"
echo "已创建log文件：$log_file"

# 定义需要执行的目录列表
dirs=(
    "toolkits/computation-BF16"
    "toolkits/computation-FP16"
    "toolkits/computation-FP32"
    "toolkits/computation-FP64"
    "toolkits/computation-TF32"
    "toolkits/computation-FP8"
    "toolkits/computation-INT8"
    # "toolkits/interconnect-MPI_intraserver"
    # "toolkits/interconnect-P2P_intraserver"
    "toolkits/main_memory-bandwidth"
    "toolkits/main_memory-capacity"
)

# 循环设置环境变量并执行脚本
for device in {0..7}; do
    echo "set enviroment MUSA_VISIBLE_DEVICES=$device"
    export MUSA_VISIBLE_DEVICES=$device
    for dir in "${dirs[@]}"; do
        echo "run $current_dir/$dir"
        cd "$current_dir/$dir" || exit 1  # 如果目录不存在则退出脚本
        # echo "执行脚本：$dir/main.sh"
        bash main.sh >> "$log_file"
    done
done

cd  "$current_dir"/toolkits/interconnect-MPI_intraserver
bash main.sh >> "$log_file"

cd  "$current_dir"/toolkits/interconnect-P2P_intraserver
bash main.sh >> "$log_file"

echo "日志已写入 $log_file 文件中"