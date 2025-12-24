# Oh-My-Zsh 快速使用指南

## ✅ 安装完成

您的 Oh-My-Zsh 已成功安装到用户目录，**无需sudo权限**，**不影响其他用户**。

## 📍 安装位置

- **zsh**: `~/miniconda3/bin/zsh` (通过conda安装)
- **Oh-My-Zsh**: `~/.oh-my-zsh`
- **配置文件**: `~/.zshrc`
- **激活脚本**: `~/bin/activate-zsh`

## 🚀 使用方法

### 方法1: 使用激活脚本（推荐）

```bash
source ~/bin/activate-zsh
```

或者简写：

```bash
. ~/bin/activate-zsh
```

### 方法2: 直接运行zsh

```bash
~/miniconda3/bin/zsh
```

### 方法3: 添加到bashrc（可选）

如果您希望每次登录时自动提示使用zsh，可以添加到 `~/.bashrc`：

```bash
echo 'alias zsh-activate="source ~/bin/activate-zsh"' >> ~/.bashrc
source ~/.bashrc
```

然后就可以使用：

```bash
zsh-activate
```

## ⚙️ 配置说明

### 当前配置

- **主题**: `robbyrussell` (默认)
- **插件**: git, python, pip, conda, docker, history, colored-man-pages, command-not-found

### 修改主题

编辑 `~/.zshrc`，修改 `ZSH_THEME` 变量：

```bash
# 查看可用主题
ls ~/.oh-my-zsh/themes

# 编辑配置文件
nano ~/.zshrc
# 或
vim ~/.zshrc

# 修改这一行
ZSH_THEME="robbyrussell"  # 改为你喜欢的主题名
```

### 添加插件

编辑 `~/.zshrc`，在 `plugins` 数组中添加：

```bash
plugins=(
    git
    python
    pip
    conda
    docker
    history
    colored-man-pages
    command-not-found
    zsh-autosuggestions  # 自动建议插件
    zsh-syntax-highlighting  # 语法高亮插件
)
```

### 安装额外插件

```bash
# 进入oh-my-zsh插件目录
cd ~/.oh-my-zsh/custom/plugins

# 克隆插件（例如：zsh-autosuggestions）
git clone https://github.com/zsh-users/zsh-autosuggestions

# 然后在 ~/.zshrc 中添加插件名
```

## 📝 常用命令

### 更新Oh-My-Zsh

```bash
cd ~/.oh-my-zsh && git pull
```

### 重新加载配置

在zsh中运行：

```zsh
source ~/.zshrc
```

或使用别名：

```zsh
omz reload
```

### 退出zsh

```bash
exit
# 或按 Ctrl+D
```

## 🔧 故障排除

### zsh命令找不到

确保conda环境已激活：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
```

### Oh-My-Zsh未加载

检查 `~/.zshrc` 文件是否存在且包含：

```bash
export ZSH="$HOME/.oh-my-zsh"
source $ZSH/oh-my-zsh.sh
```

### 主题不生效

1. 确认主题名称正确
2. 重新加载配置：`source ~/.zshrc`
3. 检查主题文件是否存在：`ls ~/.oh-my-zsh/themes/主题名.zsh-theme`

## 💡 提示

1. **不影响系统**: 所有文件都在您的用户目录下，不会影响其他用户
2. **手动激活**: 每次需要使用zsh时手动激活，不会自动切换shell
3. **bash兼容**: 退出zsh后回到bash，所有bash配置保持不变
4. **备份配置**: 原始 `.zshrc` 已备份为 `.zshrc.backup.*`

## 📚 更多资源

- [Oh-My-Zsh 官方文档](https://github.com/ohmyzsh/ohmyzsh)
- [Zsh 用户指南](http://zsh.sourceforge.net/Guide/)
- [Oh-My-Zsh 主题列表](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)
- [Oh-My-Zsh 插件列表](https://github.com/ohmyzsh/ohmyzsh/wiki/Plugins)

---

**安装日期**: $(date +%Y-%m-%d)  
**zsh版本**: $(~/miniconda3/bin/zsh --version 2>/dev/null | awk '{print $2}')

