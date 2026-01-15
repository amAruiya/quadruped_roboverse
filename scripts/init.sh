#!/bin/bash

# =================配置区域=================
# 定义需要配置的 Conda 环境列表
ENVS=("metasim" "metasim_isaacgym" "metasim_genesis")
# 定义 HuggingFace 镜像地址 (推荐使用 hf-mirror.com)
HF_MIRROR_URL="https://hf-mirror.com"
# =========================================

echo "🚀 开始配置 RoboVerse 环境 (GitPython + Tensordict + Fixes)..."

# 1. 配置 HuggingFace 镜像 (写入 .bashrc 并立即生效)
echo "---------------------------------------"
echo "🌐 正在配置 HuggingFace 镜像..."

if grep -q "HF_ENDPOINT" ~/.bashrc; then
    echo "✅ .bashrc 中已存在 HF_ENDPOINT 配置，跳过写入。"
else
    echo "export HF_ENDPOINT=$HF_MIRROR_URL" >> ~/.bashrc
    echo "✅ 已将 HF_ENDPOINT 写入 ~/.bashrc"
fi

# 立即导出变量
export HF_ENDPOINT=$HF_MIRROR_URL


# 2. 初始化 Conda
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "$CONDA_BASE" ]; then
    if [ -f "$HOME/conda/etc/profile.d/conda.sh" ]; then
        source "$HOME/conda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "❌ 无法自动找到 conda，请确保 conda 已在 PATH 中。"
        exit 1
    fi
else
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi


# 3. 循环处理每个环境
echo "---------------------------------------"
echo "📦 开始为 Conda 环境安装依赖..."

for env_name in "${ENVS[@]}"; do
    if conda info --envs | grep -q "$env_name"; then
        echo ">> 正在处理环境: [$env_name] ..."
        
        # 激活环境 
        conda activate "$env_name"
        
        # 安装 tensordict 和 GitPython (解决 No module named 'git')
        echo "   正在安装 tensordict 和 GitPython..."
        pip install tensordict GitPython tensorboard FreeSimpleGUI
        
        # 特殊处理：修复 metasim 的 packaging 冲突
        if [ "$env_name" == "metasim" ]; then
            echo "   🔧 [metasim] 正在修复 packaging 版本冲突 (降级至 23.0)..."
            pip install packaging==23.0
        fi
        
        echo "✅ 环境 [$env_name] 处理完毕。"
    else
        echo "⚠️  警告: 未找到环境 [$env_name]，跳过。"
    fi
done

sudo apt update
sudo apt install tree xsel cloc -y

echo "---------------------------------------"
echo "🎉 所有环境配置完成！" 
echo "👉 请运行: source ./setup_roboverse.sh (如果尚未运行)"