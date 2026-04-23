#!/bin/bash
# 代码质量检查脚本
# 运行格式化和代码检查

set -e

echo "=== 代码质量检查 ==="

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
echo "--- 1. 检查 import 排序 (isort) ---"
isort --check-only --diff backend/ main.py

echo ""
echo "--- 2. 检查代码格式 (black) ---"
black --check --diff backend/ main.py

echo ""
echo "=== 所有检查通过 ==="
