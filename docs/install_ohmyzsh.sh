#!/bin/bash
# MiniMind Oh-My-Zsh 安装脚本
# 适用于多用户系统，不影响其他用户

set -e

echo "=========================================="
echo "Oh-My-Zsh 用户级安装脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查zsh是否已安装
check_zsh() {
    # 优先检查conda安装的zsh
    if [ -f "$HOME/miniconda3/bin/zsh" ]; then
        ZSH_PATH="$HOME/miniconda3/bin/zsh"
        echo -e "${GREEN}✓ 找到conda zsh: $ZSH_PATH${NC}"
        return 0
    elif [ -f "$HOME/anaconda3/bin/zsh" ]; then
        ZSH_PATH="$HOME/anaconda3/bin/zsh"
        echo -e "${GREEN}✓ 找到conda zsh: $ZSH_PATH${NC}"
        return 0
    elif command -v zsh &> /dev/null; then
        ZSH_PATH=$(command -v zsh)
        echo -e "${GREEN}✓ zsh 已安装: $ZSH_PATH${NC}"
        return 0
    else
        echo -e "${RED}✗ zsh 未安装${NC}"
        return 1
    fi
}

# 检查zsh版本
check_zsh_version() {
    if check_zsh; then
        # 使用找到的zsh路径检查版本
        if [ -n "$ZSH_PATH" ] && [ -f "$ZSH_PATH" ]; then
            ZSH_VERSION=$("$ZSH_PATH" --version 2>/dev/null | awk '{print $2}' || echo "unknown")
        else
            ZSH_VERSION=$(zsh --version 2>/dev/null | awk '{print $2}' || echo "unknown")
        fi
        
        if [ "$ZSH_VERSION" != "unknown" ]; then
            echo -e "${GREEN}  zsh 版本: $ZSH_VERSION${NC}"
            # zsh 5.0+ 推荐
            MAJOR_VERSION=$(echo $ZSH_VERSION | cut -d. -f1)
            if [ "$MAJOR_VERSION" -ge 5 ]; then
                echo -e "${GREEN}  ✓ 版本符合要求${NC}"
            else
                echo -e "${YELLOW}  ⚠ 建议使用 zsh 5.0 或更高版本${NC}"
            fi
        fi
        return 0
    else
        return 1
    fi
}

# 安装oh-my-zsh
install_ohmyzsh() {
    OHMYZSH_DIR="$HOME/.oh-my-zsh"
    
    if [ -d "$OHMYZSH_DIR" ]; then
        echo -e "${YELLOW}⚠ Oh-My-Zsh 已存在于 $OHMYZSH_DIR${NC}"
        read -p "是否重新安装? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "跳过安装"
            return 0
        fi
        echo "备份现有安装..."
        mv "$OHMYZSH_DIR" "${OHMYZSH_DIR}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    echo -e "${GREEN}正在安装 Oh-My-Zsh...${NC}"
    
    # 使用官方安装脚本，但安装到用户目录
    export ZSH="$OHMYZSH_DIR"
    export RUNZSH=no  # 不自动运行zsh
    export CHSH=no    # 不自动切换shell
    
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
    
    if [ -d "$OHMYZSH_DIR" ]; then
        echo -e "${GREEN}✓ Oh-My-Zsh 安装成功！${NC}"
        return 0
    else
        echo -e "${RED}✗ Oh-My-Zsh 安装失败${NC}"
        return 1
    fi
}

# 创建激活脚本
create_activation_script() {
    ACTIVATE_SCRIPT="$HOME/bin/activate-zsh"
    mkdir -p "$HOME/bin"
    
    # 确定zsh路径
    ZSH_CMD="zsh"
    if [ -f "$HOME/miniconda3/bin/zsh" ]; then
        ZSH_CMD="$HOME/miniconda3/bin/zsh"
    elif [ -f "$HOME/anaconda3/bin/zsh" ]; then
        ZSH_CMD="$HOME/anaconda3/bin/zsh"
    elif command -v zsh &> /dev/null; then
        ZSH_CMD=$(command -v zsh)
    fi
    
    cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# 激活 zsh 和 oh-my-zsh
# 使用方法: source ~/bin/activate-zsh 或 . ~/bin/activate-zsh

# 检查zsh是否可用
ZSH_CMD="$ZSH_CMD"
if [ ! -f "\$ZSH_CMD" ] && ! command -v zsh &> /dev/null; then
    echo "错误: zsh 未安装"
    echo "请先安装 zsh:"
    echo "  - 使用conda: conda install -c conda-forge zsh"
    echo "  - 如果有sudo权限: sudo apt install zsh"
    return 1
fi

# 如果指定路径不存在，尝试系统zsh
if [ ! -f "\$ZSH_CMD" ]; then
    ZSH_CMD=\$(command -v zsh)
fi

# 检查oh-my-zsh是否已安装
if [ ! -d "\$HOME/.oh-my-zsh" ]; then
    echo "错误: Oh-My-Zsh 未安装"
    echo "请先运行安装脚本: bash ~/minimind/docs/install_ohmyzsh.sh"
    return 1
fi

# 启动zsh
exec "\$ZSH_CMD"
EOF
    
    chmod +x "$ACTIVATE_SCRIPT"
    echo -e "${GREEN}✓ 激活脚本已创建: $ACTIVATE_SCRIPT${NC}"
    echo ""
    echo "使用方法:"
    echo "  source ~/bin/activate-zsh"
    echo "  或"
    echo "  . ~/bin/activate-zsh"
}

# 配置.zshrc
configure_zshrc() {
    ZSHRC="$HOME/.zshrc"
    
    if [ -f "$ZSHRC" ]; then
        echo -e "${YELLOW}⚠ .zshrc 已存在，备份为 .zshrc.backup${NC}"
        cp "$ZSHRC" "${ZSHRC}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # 创建基础配置
    cat > "$ZSHRC" << 'EOF'
# Oh-My-Zsh 配置
export ZSH="$HOME/.oh-my-zsh"

# 主题设置（可选，默认robbyrussell）
ZSH_THEME="robbyrussell"

# 插件设置
plugins=(
    git
    python
    pip
    conda
    docker
    history
    colored-man-pages
    command-not-found
)

# 加载Oh-My-Zsh
source $ZSH/oh-my-zsh.sh

# 自定义配置
# 历史记录设置
HISTFILE=~/.zsh_history
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS
setopt HIST_SAVE_NO_DUPS

# 其他有用的设置
setopt AUTO_CD              # 输入目录名自动cd
setopt CORRECT              # 命令拼写纠正
setopt COMPLETE_IN_WORD     # 在单词中间也能补全
setopt ALWAYS_TO_END        # 补全后光标移到末尾

# 别名（可选）
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'

# Conda环境提示（如果使用conda）
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    export PS1="($CONDA_DEFAULT_ENV) $PS1"
fi

# MiniMind项目快捷方式（可选）
alias minimind='cd ~/minimind'
alias mm='cd ~/minimind'
EOF
    
    echo -e "${GREEN}✓ .zshrc 配置已创建${NC}"
}

# 使用conda安装zsh（无需sudo）
install_zsh_with_conda() {
    echo ""
    echo "尝试使用conda安装zsh（无需sudo权限）..."
    
    # 检测conda路径
    if [ -f "$HOME/miniconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/miniconda3"
    elif [ -f "$HOME/anaconda3/bin/conda" ]; then
        CONDA_PATH="$HOME/anaconda3"
    elif command -v conda &> /dev/null; then
        CONDA_PATH=$(dirname $(dirname $(command -v conda)))
    else
        echo -e "${RED}✗ 未找到conda${NC}"
        return 1
    fi
    
    echo -e "${GREEN}找到conda: $CONDA_PATH${NC}"
    
    # 初始化conda（如果还没初始化）
    if [ ! -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        echo "初始化conda环境..."
        "$CONDA_PATH/bin/conda" init bash 2>/dev/null || true
    fi
    
    # 使用conda-forge安装zsh
    echo "正在从conda-forge安装zsh..."
    "$CONDA_PATH/bin/conda" install -y -c conda-forge zsh 2>&1 | grep -v "^$" || {
        echo -e "${YELLOW}尝试使用mamba（如果可用）...${NC}"
        if command -v mamba &> /dev/null; then
            mamba install -y -c conda-forge zsh || return 1
        else
            return 1
        fi
    }
    
    # 验证安装
    if [ -f "$CONDA_PATH/bin/zsh" ]; then
        ZSH_PATH="$CONDA_PATH/bin/zsh"
        echo -e "${GREEN}✓ zsh 安装成功: $ZSH_PATH${NC}"
        return 0
    else
        echo -e "${RED}✗ zsh 安装失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    echo ""
    echo "步骤 1: 检查 zsh 安装状态"
    echo "----------------------------------------"
    if ! check_zsh_version; then
        echo ""
        echo -e "${YELLOW}zsh 未安装，尝试自动安装...${NC}"
        echo ""
        
        # 尝试使用conda安装
        if command -v conda &> /dev/null || [ -f "$HOME/miniconda3/bin/conda" ] || [ -f "$HOME/anaconda3/bin/conda" ]; then
            if install_zsh_with_conda; then
                echo -e "${GREEN}✓ zsh 安装成功！${NC}"
            else
                echo ""
                echo -e "${RED}自动安装失败，请手动安装zsh:${NC}"
                echo ""
                echo "方法1: 使用conda（推荐，无需sudo）"
                echo "  ${YELLOW}conda install -c conda-forge zsh${NC}"
                echo ""
                echo "方法2: 如果有sudo权限"
                echo "  ${YELLOW}sudo apt install zsh${NC}"
                echo ""
                echo "方法3: 从源码编译（较复杂，见文档）"
                echo ""
                read -p "安装zsh后，按Enter继续，或Ctrl+C退出..."
                
                if ! check_zsh; then
                    echo -e "${RED}zsh 仍未安装，退出安装${NC}"
                    exit 1
                fi
            fi
        else
            echo -e "${RED}错误: zsh 未安装且未找到conda${NC}"
            echo ""
            echo "请先安装 zsh:"
            echo "  1. 如果有sudo权限:"
            echo "     ${YELLOW}sudo apt update && sudo apt install zsh${NC}"
            echo ""
            echo "  2. 安装conda后使用:"
            echo "     ${YELLOW}conda install -c conda-forge zsh${NC}"
            echo ""
            echo "  3. 或联系系统管理员安装zsh"
            echo ""
            read -p "安装zsh后，按Enter继续..."
            
            if ! check_zsh; then
                echo -e "${RED}zsh 仍未安装，退出安装${NC}"
                exit 1
            fi
        fi
    fi
    
    echo ""
    echo "步骤 2: 安装 Oh-My-Zsh"
    echo "----------------------------------------"
    if ! install_ohmyzsh; then
        echo -e "${RED}安装失败，退出${NC}"
        exit 1
    fi
    
    echo ""
    echo "步骤 3: 配置 .zshrc"
    echo "----------------------------------------"
    configure_zshrc
    
    echo ""
    echo "步骤 4: 创建激活脚本"
    echo "----------------------------------------"
    create_activation_script
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}安装完成！${NC}"
    echo "=========================================="
    echo ""
    echo "重要提示:"
    echo "  1. 此安装不会修改系统默认shell"
    echo "  2. 每次需要使用zsh时，运行:"
    echo "     ${YELLOW}source ~/bin/activate-zsh${NC}"
    echo "  3. 或者添加到 ~/.bashrc 中（可选）:"
    echo "     ${YELLOW}echo 'alias zsh-activate=\"source ~/bin/activate-zsh\"' >> ~/.bashrc${NC}"
    echo ""
    echo "现在可以运行以下命令激活zsh:"
    echo "  ${GREEN}source ~/bin/activate-zsh${NC}"
    echo ""
}

# 运行主函数
main

