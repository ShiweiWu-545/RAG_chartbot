#!/bin/bash
# 设置 git pre-commit 钩子
# 在每次提交前自动运行代码格式检查

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_DIR/.git/hooks"

echo "=== 设置 pre-commit 钩子 ==="

# 创建 hooks 目录（如果不存在）
mkdir -p "$HOOKS_DIR"

# 创建 pre-commit 钩子
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit 钩子 - 在提交前运行格式检查

set -e

echo "Running pre-commit hooks..."

# 检查 black
if command -v black &> /dev/null; then
    echo "Checking code format with black..."
    black --check backend/ main.py || {
        echo "Code format issues found. Run 'bash scripts/format.sh' to fix."
        exit 1
    }
fi

# 检查 isort
if command -v isort &> /dev/null; then
    echo "Checking import order with isort..."
    isort --check-only backend/ main.py || {
        echo "Import order issues found. Run 'bash scripts/format.sh' to fix."
        exit 1
    }
fi

echo "Pre-commit checks passed!"
EOF

chmod +x "$HOOKS_DIR/pre-commit"

echo ""
echo "Pre-commit 钩子已设置完成！"
echo "每次 git commit 前会自动运行格式检查。"
