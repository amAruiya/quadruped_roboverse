#!/bin/bash

# ================= 路径配置 =================
# 1. Isaac Lab 的根目录
ISAACLAB_ROOT="/home/ubuntu/RoboVerse/third_party/IsaacLab"

# 2. Python 解释器路径
PYTHON_EXE="/home/ubuntu/conda/envs/metasim/bin/python"

# 3. URDF 输入文件路径
INPUT_URDF="/home/ubuntu/RoboVerse/roboverse_data/robots/Leap/urdf/Leap.urdf"

# 4. USD 输出文件路径
OUTPUT_USD="/home/ubuntu/RoboVerse/roboverse_data/robots/Leap/usd/Leap.usd"

# ================= 环境设置 =================
# 修复 CXXABI 报错的关键步骤
export LD_LIBRARY_PATH=/home/ubuntu/conda/envs/metasim/lib:$LD_LIBRARY_PATH

# ================= 执行逻辑 =================

# 1. 检查目标工具是否存在
TOOL_SCRIPT="$ISAACLAB_ROOT/scripts/tools/convert_urdf.py"
if [ ! -f "$TOOL_SCRIPT" ]; then
    echo "错误: 找不到官方转换工具: $TOOL_SCRIPT"
    echo "请检查 ISAACLAB_ROOT 路径是否正确。"
    exit 1
fi

# 2. 自动切换到工具目录
# (某些 Isaac Lab 版本要求必须在当前目录下运行，以正确加载相对路径的库)
cd "$ISAACLAB_ROOT/scripts/tools" || exit
echo "已切换工作目录至: $(pwd)"

# 3. 调用官方脚本
echo "正在运行 Isaac Lab 官方转换工具..."

# 注意参数配置：
# - 这种调用方式不需要 'input=' 或 'output='，直接跟路径即可
# - 四足机器人 (Leap) 通常不需要 --fix-base (浮动基座)
# - --make-instanceable 推荐开启，优化性能

"$PYTHON_EXE" convert_urdf.py \
    "$INPUT_URDF" \
    "$OUTPUT_USD" \
    --joint-stiffness 0.0 \
    --joint-damping 0.0 \

# ================= 结果检查 =================
if [ $? -eq 0 ]; then
    echo "---------------------------------------"
    echo "SUCCESS: 转换成功！"
    echo "输出文件: $OUTPUT_USD"
    echo "---------------------------------------"
else
    echo "---------------------------------------"
    echo "ERROR: 转换失败，请检查上方报错信息。"
    echo "---------------------------------------"
fi