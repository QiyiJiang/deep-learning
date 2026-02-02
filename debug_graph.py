"""
编写 debug_graph.py 脚本，实例化模型 MiniMindForCausalLM，构造一个小批量 Dummy 输入（如 input_ids = torch.randint(0, 100, (1, 10))），执行一次前向并使用 torchviz.make_dot 生成计算图 PDF。
"""

from model.model_minimind import MiniMindForCausalLM
import torch
import torchviz
from model.model_minimind import MiniMindConfig
import warnings
warnings.filterwarnings('ignore')

# 注意：完整 512/16 层模型的计算图节点非常多，任何可视化都会“看起来糊/字很小”。
# 画图建议先用小模型做结构调试；需要时再把参数调大。
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=16, num_hidden_layers=2, max_seq_len=256, use_moe=False))
input_ids = torch.randint(0, 100, (1, 8))
output = model(input_ids)
print(output)

dot = torchviz.make_dot(output.logits.sum(), params=dict(model.named_parameters()))
# 说明：
# - PDF 本身是矢量（理论上不糊），但很多 IDE 内置预览会用低分辨率栅格化显示，所以看起来“糊”。
# - 因此这里同时输出 SVG(放大不糊) + PNG(高 dpi，IDE 预览也清晰) + PDF(用于分享/打印)。
dot.graph_attr.update(
    dpi="600",          # 对 PNG 有效；PDF/SVG 主要影响不大
    fontsize="18",      # 整体字体变大，IDE 预览更容易读
    ranksep="0.45",
    nodesep="0.35",
)
dot.node_attr.update(fontsize="14")

dot.render("model_graph", format="pdf")