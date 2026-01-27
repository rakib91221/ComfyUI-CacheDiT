# ComfyUI-CacheDiT ⚡

<div align="center">

**Production-ready DiT Model Acceleration for ComfyUI**

[English](#english) | [中文](#中文)

[![cache-dit](https://img.shields.io/badge/cache--dit-v1.2.0+-blue)](https://github.com/vipshop/cache-dit)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-green)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

<a name="english"></a>
## 🚀 Overview

ComfyUI-CacheDiT integrates the [cache-dit](https://github.com/vipshop/cache-dit) library into ComfyUI, providing **1.5x-3x speedup** for DiT (Diffusion Transformer) model inference through inter-step residual caching.

### Supported Models (2026)

| Model | Pattern | Recommended Config | Notes |
|-------|---------|-------------------|-------|
| **Qwen-Image** | Pattern_1 | F1B0, threshold=0.12 | Separate CFG |
| **Qwen-Image-Layered** | Pattern_1 | F8B4, threshold=0.10 | Alpha layer protection |
| **LTX-2 (T2V/I2V)** | Pattern_1 | F4B4, skip_interval=3 | Temporal consistency |
| **Z-Image** | Pattern_1 | F8B0, noise_scale=0.0015 | Small noise injection |
| **Z-Image-Turbo** | Pattern_1 | F4B0, threshold=0.15 | 4-9 steps distilled |
| **Flux** | Pattern_0 | F10B0, threshold=0.10 | Standard MMDiT |
| **HunyuanVideo** | Pattern_3 | F6B2, skip_interval=2 | Fused CFG |
| **Wan 2.1** | Pattern_3 | F6B2, skip_interval=2 | Separate CFG |

## 📦 Installation

### Prerequisites

```bash
# Install cache-dit library (v1.2.0+)
pip install cache-dit>=1.2.0
```

### Install Node

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-org/ComfyUI-CacheDiT.git
```

Or download and extract to `ComfyUI/custom_nodes/ComfyUI-CacheDiT/`

## 🎯 Quick Start

1. Load your model using any model loader
2. Connect to **⚡ CacheDiT Model Optimizer** node
3. Select your model type from presets
4. Connect to KSampler

```
[Load Checkpoint] → [⚡ CacheDiT Model Optimizer] → [KSampler]
```

## ⚙️ Parameters

### Basic Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | Combo | Custom | Model preset (auto-configures settings) |
| `forward_pattern` | Combo | Pattern_1 | Transformer block forward pattern |
| `strategy` | Combo | adaptive | Caching strategy: adaptive/static/dynamic |
| `threshold` | Float | 0.12 | Residual diff threshold (0.01-0.5) |
| `fn_blocks` | Int | 8 | Front blocks for diff calculation (Fn) |
| `bn_blocks` | Int | 0 | Back blocks for feature fusion (Bn) |

### Advanced Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup_steps` | Int | 8 | Steps before caching starts |
| `skip_interval` | Int | 0 | Force compute every N steps (for video) |
| `noise_scale` | Float | 0.0 | Noise injection (0.001-0.003 typical) |
| `taylor_order` | Int | 1 | TaylorSeer order (0=disabled) |
| `scm_policy` | Combo | none | Steps Computation Mask policy |
| `separate_cfg` | Combo | auto | CFG separation mode |
| `verbose` | Bool | False | Verbose logging |
| `print_summary` | Bool | True | Print performance dashboard |

## 📊 Caching Strategies

### Adaptive (Recommended)
- Auto-balances quality and speed
- Best for most use cases

### Static
- Aggressive caching
- Maximum speedup
- May reduce quality for complex scenes

### Dynamic
- Conservative caching
- Limits continuous cached steps
- Better quality preservation

## 🔬 How It Works

### DBCache Algorithm

Cache-dit implements Dual Block Cache (DBCache):

1. **Warmup Phase**: First N steps compute normally to establish baseline
2. **Caching Phase**: Compare residuals between steps
   - If $\|r_t - r_{t-1}\|_1 < \text{threshold}$: Use cached output
   - Otherwise: Compute full forward pass
3. **Fn/Bn Blocks**: 
   - Fn (front blocks): Always compute for stable diff estimation
   - Bn (back blocks): Fuse features for accuracy

### TaylorSeer

Uses Taylor series expansion to predict future residuals:
- Order 0: Pure caching (disabled)
- Order 1: First-order prediction (recommended)
- Order 2: Second-order prediction (more accurate, slower)

### Skip Interval (Video)

For video models (LTX-2, HunyuanVideo), temporal consistency requires:
```
skip_interval=3  →  [Compute, Cache, Cache, Compute, Cache, Cache, ...]
```

## 📈 Performance Dashboard

After sampling, a rich ASCII dashboard is printed:

```
╔════════════════════════════════════════════════════════════════╗
║          CacheDiT Performance Dashboard                  ║
╠════════════════════════════════════════════════════════════════╣
║  Model: Qwen-Image                                             ║
║  Pattern: Pattern_1                                            ║
║  Strategy: adaptive                                            ║
╠────────────────────────────────────────────────────────────────╣
║  📊 Performance Metrics                                        ║
║────────────────────────────────────────────────────────────────║
║  Total Steps:              28                                  ║
║  Computed Steps:           12                                  ║
║  Cached Steps:             16                                  ║
║  Cache Hit Rate:           57.1%                               ║
║  Estimated Speedup:        2.33x                               ║
╠────────────────────────────────────────────────────────────────╣
║  🎯 Quality Metrics                                            ║
║  Threshold:                0.1200                              ║
║  Avg Residual Diff:        0.089234                            ║
║  Fn/Bn Blocks:             F8B0                                ║
╚════════════════════════════════════════════════════════════════╝
```

## 🎛️ Tuning Guide

### Finding the "Sweet Spot"

1. **Start Conservative**: threshold=0.08, warmup=10
2. **Check Quality**: Run a test generation
3. **Adjust Threshold**: 
   - Quality issues? Lower threshold
   - Too slow? Raise threshold
4. **Monitor Dashboard**: Aim for 1.5x-2.5x speedup

### Model-Specific Tips

**Qwen-Image-Layered**:
- Use F8B4 to protect Alpha layer
- Lower threshold (0.10) for transparency accuracy

**LTX-2 Video**:
- Always set skip_interval=3 for temporal consistency
- Add small noise_scale=0.001 to prevent static frames

**Z-Image**:
- noise_scale=0.0015 prevents "dead" regions
- F8B0 works well for most cases

---

<a name="中文"></a>
## 🚀 概述

ComfyUI-CacheDiT 将 [cache-dit](https://github.com/vipshop/cache-dit) 库集成到 ComfyUI，通过步间残差缓存为 DiT（Diffusion Transformer）模型推理提供 **1.5x-3x 加速**。

### 支持的模型（2026）

| 模型 | 模式 | 推荐配置 | 备注 |
|------|------|----------|------|
| **Qwen-Image** | Pattern_1 | F1B0, threshold=0.12 | 分离 CFG |
| **Qwen-Image-Layered** | Pattern_1 | F8B4, threshold=0.10 | Alpha 层保护 |
| **LTX-2 (T2V/I2V)** | Pattern_1 | F4B4, skip_interval=3 | 时序一致性 |
| **Z-Image** | Pattern_1 | F8B0, noise_scale=0.0015 | 小噪声注入 |
| **Z-Image-Turbo** | Pattern_1 | F4B0, threshold=0.15 | 4-9步蒸馏版 |
| **Flux** | Pattern_0 | F10B0, threshold=0.10 | 标准 MMDiT |
| **混元视频** | Pattern_3 | F6B2, skip_interval=2 | 融合 CFG |
| **万相 2.1** | Pattern_3 | F6B2, skip_interval=2 | 分离 CFG |

## 📦 安装

### 前置要求

```bash
# 安装 cache-dit 库 (v1.2.0+)
pip install cache-dit>=1.2.0
```

### 安装节点

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-org/ComfyUI-CacheDiT.git
```

或下载并解压到 `ComfyUI/custom_nodes/ComfyUI-CacheDiT/`

## 🎯 快速开始

1. 使用任意模型加载器加载模型
2. 连接到 **⚡ CacheDiT Model Optimizer** 节点
3. 从预设中选择模型类型
4. 连接到 KSampler

```
[加载检查点] → [⚡ CacheDiT Model Optimizer] → [KSampler]
```

## ⚙️ 参数说明

### 基础设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | 下拉框 | Custom | 模型预设（自动配置参数）|
| `forward_pattern` | 下拉框 | Pattern_1 | Transformer 块前向传播模式 |
| `strategy` | 下拉框 | adaptive | 缓存策略：adaptive/static/dynamic |
| `threshold` | 浮点数 | 0.12 | 残差阈值（0.01-0.5）|
| `fn_blocks` | 整数 | 8 | 用于差分计算的前置块数（Fn）|
| `bn_blocks` | 整数 | 0 | 用于特征融合的后置块数（Bn）|

### 高级设置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `warmup_steps` | 整数 | 8 | 缓存开始前的预热步数 |
| `skip_interval` | 整数 | 0 | 每 N 步强制计算（用于视频）|
| `noise_scale` | 浮点数 | 0.0 | 噪声注入强度（通常 0.001-0.003）|
| `taylor_order` | 整数 | 1 | TaylorSeer 阶数（0=禁用）|
| `scm_policy` | 下拉框 | none | 步数计算掩码策略 |
| `separate_cfg` | 下拉框 | auto | CFG 分离模式 |
| `verbose` | 布尔值 | False | 详细日志 |
| `print_summary` | 布尔值 | True | 打印性能仪表盘 |

## 📊 缓存策略

### Adaptive（推荐）
- 自动平衡质量和速度
- 适用于大多数场景

### Static
- 激进缓存
- 最大加速
- 复杂场景可能降低质量

### Dynamic
- 保守缓存
- 限制连续缓存步数
- 更好的质量保持

## 🔬 工作原理

### DBCache 算法

cache-dit 实现了双块缓存（DBCache）：

1. **预热阶段**：前 N 步正常计算以建立基线
2. **缓存阶段**：比较步间残差
   - 如果 $\|r_t - r_{t-1}\|_1 < \text{threshold}$：使用缓存输出
   - 否则：计算完整前向传播
3. **Fn/Bn 块**：
   - Fn（前置块）：始终计算以稳定差分估计
   - Bn（后置块）：融合特征以提高精度

### TaylorSeer

使用泰勒级数展开预测未来残差：
- 阶数 0：纯缓存（禁用）
- 阶数 1：一阶预测（推荐）
- 阶数 2：二阶预测（更精确但更慢）

### Skip Interval（视频）

对于视频模型（LTX-2、混元视频），时序一致性需要：
```
skip_interval=3  →  [计算, 缓存, 缓存, 计算, 缓存, 缓存, ...]
```

## 🎛️ 调优指南

### 寻找"甜点位"

1. **从保守开始**：threshold=0.08, warmup=10
2. **检查质量**：运行测试生成
3. **调整阈值**：
   - 质量问题？降低阈值
   - 太慢？提高阈值
4. **监控仪表盘**：目标 1.5x-2.5x 加速

### 模型专属技巧

**Qwen-Image-Layered**：
- 使用 F8B4 保护 Alpha 层
- 降低阈值（0.10）以保证透明度精度

**LTX-2 视频**：
- 始终设置 skip_interval=3 以保证时序一致性
- 添加小噪声 noise_scale=0.001 防止静态帧

**Z-Image**：
- noise_scale=0.0015 防止"死区"
- F8B0 适用于大多数情况

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [cache-dit](https://github.com/vipshop/cache-dit) - The underlying acceleration library
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - The UI framework
- All contributors and the open-source community

---

<div align="center">

**Made with ⚡ for the ComfyUI community**

</div>
