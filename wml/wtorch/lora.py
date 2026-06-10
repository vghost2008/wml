import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from .utils import simple_model_device


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, rank=4, lora_alpha=16, dropout=0.0):
        super(LoRALinear, self).__init__()
        
        # 1. 冻结的原始线性层
        self.linear = nn.Linear(in_features, out_features,bias=bias)
        # 冻结原始权重，不参与梯度更新
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # 2. LoRA 适配器部分
        self.rank = rank
        self.lora_alpha = lora_alpha
        # 缩放系数，通常设为 rank 的倍数或固定值，用于稳定训练
        self.scaling = self.lora_alpha / self.rank
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        
        # 低秩矩阵 A (in_features -> rank)
        # 初始化：高斯分布
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        # 低秩矩阵 B (rank -> out_features)
        # 初始化：零初始化，确保训练初期 Delta W 为 0，不破坏预训练知识
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # 原始路径
        original_output = self.linear(x)
        
        # LoRA 路径: x -> Dropout -> A -> B -> Scaling
        # 注意矩阵乘法顺序: (x @ A.T) @ B.T 或者按照定义 B @ (A @ x)
        # 这里假设输入 x 形状为 [batch, seq_len, in_features]
        # lora_A shape: [rank, in_features]
        # lora_B shape: [out_features, rank]
        
        # 计算低秩更新量
        # x: [B, L, D_in]
        # A: [D_in, R] (转置后) -> x @ A.T = [B, L, R]
        # B: [R, D_out] (转置后) -> result @ B.T = [B, L, D_out]
        
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output

class LoRAConv(nn.Module):
    '''
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    '''

    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,rank=4, lora_alpha=16, dropout=0.0):
        super().__init__()
        
        # 1. 冻结的原始线性层
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        # 冻结原始权重，不参与梯度更新
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # 2. LoRA 适配器部分
        self.rank = rank
        self.lora_alpha = lora_alpha
        # 缩放系数，通常设为 rank 的倍数或固定值，用于稳定训练
        self.scaling = self.lora_alpha / self.rank
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout) if dropout>0 else nn.Identity()
        
        # 低秩矩阵 A (in_features -> rank)
        # 初始化：高斯分布
        self.lora_A = nn.Conv2d(in_channels=in_channels,out_channels=rank,kernel_size=1,bias=False)
        # 低秩矩阵 B (rank -> out_features)
        # 初始化：零初始化，确保训练初期 Delta W 为 0，不破坏预训练知识
        self.lora_B = nn.Conv2d(in_channels=rank,out_channels=out_channels,kernel_size=1,bias=False)
        self.lora_B.weight.data.zero_()

        
    def forward(self, x):
        # 原始路径
        original_output = self.conv(x)
        
        # LoRA 路径: x -> Dropout -> A -> B -> Scaling
        # 注意矩阵乘法顺序: (x @ A.T) @ B.T 或者按照定义 B @ (A @ x)
        # 这里假设输入 x 形状为 [batch, seq_len, in_features]
        # lora_A shape: [rank, in_features]
        # lora_B shape: [out_features, rank]
        
        # 计算低秩更新量
        # x: [B, L, D_in]
        # A: [D_in, R] (转置后) -> x @ A.T = [B, L, R]
        # B: [R, D_out] (转置后) -> result @ B.T = [B, L, D_out]
        
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output

def simple_get_model(
    model: nn.Module, 
    target_modules=[nn.Linear],
    rank=4, lora_alpha=16, dropout=0.0,
):
    """
    简化版的 get_peft_model 实现，仅以 LoRA 为例展示核心逻辑。
    """

    # 3. 遍历模型，查找并替换目标模块 (Target Modules)
    # 例如：将 nn.Linear 替换为支持 LoRA 的 Linear
    if model is None:
        return model
    
    # 递归遍历模型的所有子模块
    for name, module in model.named_modules():
        # 检查当前模块是否在目标列表中，且是线性层
        # 注意：实际库中的匹配逻辑更复杂，支持正则表达式和模块类型匹配
        if _is_target_module(name, module, target_modules):
            # 获取原始线性层的参数
            old_linear = module
            if old_linear.in_features<rank or old_linear.out_features<rank:
                continue
            
            # 创建新的 LoRA 线性层，继承原始层的权重和偏置
            # LoraLinear 内部会初始化 lora_A 和 lora_B，并保持 original_weight 冻结
            new_lora_layer = LoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                bias=old_linear.bias is not None,
                rank=rank,lora_alpha=lora_alpha,dropout=dropout,
            )
            new_lora_layer.to(simple_model_device(old_linear))

            
            # 复制原始权重到新层 (作为冻结的基础权重)
            new_lora_layer.linear.weight.data = old_linear.weight.data.clone()
            if old_linear.bias is not None:
                new_lora_layer.linear.bias.data = old_linear.bias.data.clone()
                
            # 在父模块中替换旧模块为新模块
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
            else:
                parent_module = model
                
            setattr(parent_module, child_name, new_lora_layer)

    return model

def simple_get_conv_model(
    model: nn.Module, 
    target_modules=[nn.Conv2d],
    rank=4, lora_alpha=16, dropout=0.0,

):
    """
    简化版的 get_peft_model 实现，仅以 LoRA 为例展示核心逻辑。
    """
    # 递归遍历模型的所有子模块
    if model is None:
        return model
    for name, module in model.named_modules():
        # 检查当前模块是否在目标列表中，且是线性层
        # 注意：实际库中的匹配逻辑更复杂，支持正则表达式和模块类型匹配
        if _is_target_module(name, module, target_modules):
            # 获取原始线性层的参数
            old_conv = module
            if max(old_conv.stride)>1 or old_conv.in_channels<rank or old_conv.out_channels<rank:
                continue
            
            # 创建新的 LoRA 线性层，继承原始层的权重和偏置
            # LoraLinear 内部会初始化 lora_A 和 lora_B，并保持 original_weight 冻结
            new_lora_layer = LoRAConv(
                in_channels=old_conv.in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=old_conv.groups,
                bias=old_conv.bias is not None,
                rank=rank,lora_alpha=lora_alpha,dropout=dropout,
            )
            new_lora_layer.to(simple_model_device(old_conv))
            
            # 复制原始权重到新层 (作为冻结的基础权重)
            new_lora_layer.conv.weight.data = old_conv.weight.data.clone()
            if old_conv.bias is not None:
                new_lora_layer.conv.bias.data = old_conv.bias.data.clone()
                
            # 在父模块中替换旧模块为新模块
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            
            if parent_name:
                parent_module = dict(model.named_modules())[parent_name]
            else:
                parent_module = model
                
            setattr(parent_module, child_name, new_lora_layer)

    return model

def _is_target_module(name: str, module: nn.Module, target_modules: Union[list, str]) -> bool:
    """
    简单的辅助函数，判断模块是否为目标模块。
    实际 PEFT 库中使用更复杂的匹配策略 (如 fnmatch 或 isinstance 检查)。
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]
        
    # 检查模块名称是否匹配
    for target in target_modules:
        #if name.endswith(target):
            #return True
        if isinstance(module,target):
            return True
        # 也可以检查模块类型，例如 isinstance(module, nn.Linear)
    return False
