#!/bin/bash
# 代码格式化脚本
# 自动格式化所有 Python 代码

set -e

echo "=== 代码格式化 ==="

# 检查 black 是否安装
if ! command -v black &> /dev/null; then
    echo "错误: black 未安装，请运行: uv add --dev black"
    exit 1
fi

# 检查 isort 是否安装
if ! command -v isort &> /dev/null; then
    echo "错误: isort 未安装，请运行: uv add --dev isort"
    exit 1
fi

echo ""
echo "--- 1. 排序 imports (isort) ---"
isort backend/ main.py

echo ""
echo "--- 2. 格式化代码 (black) ---"
black backend/ main.py

echo ""
echo "=== 格式化完成 ==="
