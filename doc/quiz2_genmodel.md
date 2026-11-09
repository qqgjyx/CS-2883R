# Quiz 2 cheat sheet: Generative Modeling unit (28 Oct)

四篇: JiT, MeanFlow (10/14) + Drifting, Diffusion Forcing (10/21). 闭卷 10 分钟, 每篇约 2 题。

**Quiz 1 的教训**: 考的是"精确机制名词"。丢分的都是: 第二个 loss 项 (perceptual loss)、
"predicted" 和 "last observed" 的区别、模型到底输出什么 (index 而不是 reconstruction)、
停止规则 (误差 < 阈值)、训练图里的 stop-grad / EMA / loss 公式。
所以每篇都要能答: **网络输入什么、输出什么、loss 是什么、stop-grad 在哪、推理几步 (NFE)**。

## 进考场前最后 1 分钟看这张表

| | JiT | MeanFlow | Drifting | Diffusion Forcing |
| --- | --- | --- | --- | --- |
| 网络输出 | **干净图 x** (x-pred) | **平均速度 u**(z, r, t) | 样本 x = f(ε) | 噪声 ε_t (每个 token) |
| Loss | **v-loss**: ‖v_θ − v‖², v_θ = (x_θ − z)/(1−t) | ‖u_θ − sg(v − (t−r)·du/dt)‖² | ‖f(ε) − sg(f(ε) + V)‖² | Σ_t ‖ε_t − ε_θ‖² |
| stop-grad | 无 | target 上 (避免穿过 JVP 的 double backprop) | target x + V 上 | 无 |
| 关键技巧 | 高维 patch + 流形假设; bottleneck embed | JVP, tangent **(v, 0, 1)**; r = t 退化为 FM | V = 吸引 − 排斥, 反对称 ⇒ p = q 时 V = 0; 特征编码器 φ | 每个 token 独立噪声 k_t; scheduling matrix; MCG |
| 推理 | 50 步 Heun | **1 NFE** (z_0 = ε − u(ε, 0, 1)) | **1 NFE** | 按调度矩阵逐行去噪 (自回归, 可变长度) |
| 空间 | **像素**, patch 16/32 | VAE latent (ImageNet) | latent 或 pixel | 序列 (视频帧, 动作) |
| CFG | 标准 CFG + interval | 写进训练 target, 采样仍 1-NFE | 训练时加真实无条件负样本, α 作输入 | 不适用 (用 reward guidance) |
| 头条数字 | G/16 **1.82** @256, G/32 **1.78** @512 | XL/2 **3.43** (1-NFE), 2.20 (2-NFE) | L/2 **1.54** latent, L/16 **1.61** pixel | Maze 141.7 vs Diffuser 119.5; 机器人 80% |

**时间约定陷阱**: MeanFlow 是 t=1 噪声 (z = (1−t)x + tε, v = ε − x);
JiT 是 t=1 数据 (z = tx + (1−t)ε, v = x − ε)。

**故事线**: MeanFlow 和 Drifting 都在追求 1 步生成, 而且都是 Kaiming He 组的工作。
- MeanFlow: 把"积分"学进网络 (平均速度)。
- Drifting: 把"迭代"挪到训练时完成 (样本漂移)。
- JiT 也是 He 组的, 但主题不同: "预测什么"决定了能不能直接在高维像素上做。
- Diffusion Forcing 讲的是"序列 + 每个 token 不同噪声", 和后面的 world model 单元直接相关。

## 0. 通用背景 (四篇共用)

- **Flow / diffusion 插值**: z_t = (1−t)·x + t·ε, x 是数据, ε ~ N(0, I)。
  MeanFlow 和 JiT 都用这个线性插值, 但**端点约定相反**, 见各节。
- **三种预测目标**: ε-prediction (噪声), x-prediction (干净图), v-prediction (速度 v = dz/dt)。
  知道任意两个可以互推: 已知 z_t 和预测的 x, 就有 ε = (z_t − (1−t)x)/t, 进而得到 v。
- **NFE** (number of function evaluations): 推理时调用网络的次数。
  CFG 每步要 cond + uncond 两次, 记为 "×2"。
- **CFG** (classifier-free guidance): v_cfg = ω·v(z|c) + (1−ω)·v(z)。
  训练时以 10% 概率丢掉 class label, 学出 uncond 分支。
- **FID** 越低越好。ImageNet 256×256, 在 50K 张图上算 (FID-50K)。

## 1. MeanFlow (Geng, Deng, Bai, Kolter, Kaiming He; arXiv 2505.13447)

**一句话**: 不学瞬时速度 v, 而是学**平均速度 (average velocity) u**。
一步生成: x = ε − u_θ(ε, r=0, t=1)。从头训练, 不需要 pretraining、distillation 或 curriculum。

**约定**: z_t = (1−t)x + tε。**t=0 是数据, t=1 是噪声**。conditional velocity v_t = ε − x。

**定义 (Eq. 3)**: u(z_t, r, t) = 1/(t−r) · ∫_r^t v(z_τ, τ) dτ, 即位移 / 时间间隔。
- 边界条件: r→t 时 u→v。
- 自带"一致性": (t−r)u(z_t,r,t) = (s−r)u(z_s,r,s) + (t−s)u(z_t,s,t), 来自积分可加性,
  所以**不需要额外的 consistency 约束**。

**MeanFlow Identity (Eq. 6)**: 对 t 求导, 把 r 当作与 t 无关
(用乘积法则 product rule + 微积分基本定理), 得到

&nbsp;&nbsp;&nbsp;&nbsp;**u = v − (t−r)·du/dt**

**全导数 (Eq. 8)**: du/dt = v·∂_z u + ∂_t u, 因为 dz/dt = v, dr/dt = 0, dt/dt = 1。
- 用 **JVP** (Jacobian-vector product) 一次算出: Jacobian 是 (∂_z u, ∂_r u, ∂_t u),
  **tangent 是 (v, 0, 1)**。
- 实现: torch.func.jvp 或 jax.jvp, 额外开销约等于一次 backward (< 20% 训练时间)。

**Loss (Eq. 9 到 11, Alg. 1)**:
- L = E ‖u_θ(z_t, r, t) − **sg**(u_tgt)‖²
- u_tgt = v_t − (t−r)·(v_t·∂_z u_θ + ∂_t u_θ)
- **stop-grad 加在 target 上**, 作用是避免 double backprop 穿过 JVP (不需要高阶梯度)。
- 唯一的 ground-truth 信号是 v_t = ε − x, 不需要算积分。
- **r = t 时第二项消失, 方法就退化为 Flow Matching**。

**采样 (Eq. 12, Alg. 2)**: z_r = z_t − (t−r)·u(z_t, r, t)。
1 步: z_0 = z_1 − u(z_1, 0, 1)。2 步也很直接。

**为什么 FM 一步不行**: 真实的 marginal velocity 轨迹是弯的, 粗步长会有大误差。
在 t=1 处 v(z,1) = z − E[x], 一步 Euler 就落到**数据均值** (这是我 Hacker toy 的直观)。

**CFG (§4.2, Eq. 13 到 19)**:
- 把 CFG 当作 ground-truth 场的性质: v_cfg = ω v(z|c) + (1−ω) v(z)。
- 训练 target 里用 ṽ_t = ω·v_t + (1−ω)·u_θ^cfg(z_t, t, t)。
- 采样时**不需要再做线性组合, 仍是 1-NFE**。ω=1 时退化为无 CFG。

**设计选择 (§4.3)**:
- 自适应 loss 权重 w = 1/(‖Δ‖² + c)^p, 用法是 sg(w)·‖Δ‖²。
- (r, t) 取两个 logit-normal 样本, 大的给 t, 小的给 r, 并按比例让一部分 r = t。
- 网络条件输入用 (t, t−r) 最好。

**关于架构的考点**: ImageNet 实验在 **VAE latent** (32×32×4) 上做, 不是 pixel;
ablation 用 ViT-B/4 (DiT 风格)。CIFAR-10 实验在 pixel 上用 U-Net。

**数字 (Tab. 1, B/4, 1-NFE FID)**:

| ablation | 结果 |
| --- | --- |
| 1a: r≠t 的比例 | **0% (= FM) 328.91**, **25% 61.06** (最好), 50% 63.14, 100% 67.32 |
| 1b: JVP tangent | **(v,0,1) 61.06 正确**; (v,0,0) 268.06; (v,1,0) 329.22; (v,1,1) 137.96 |
| 1c: 条件输入 | (t, t−r) 61.06 最好; (t, r) 61.75; 只给 t−r 63.13 也能用 |
| 1e: loss 权重 p | p=0 (纯 L2) 79.75, p=0.5 (≈ Pseudo-Huber) 63.98, **p=1 61.06** |
| 1f: CFG ω | 1.0 61.06, 2.0 20.15, **3.0 15.53**, 5.0 20.75 |

**数字 (Tab. 2, ImageNet 256)**:
- 1-NFE: XL/2 **3.43**。之前最好: Shortcut 10.60 (1-NFE), IMM 7.77 (1×2)。
- 2-NFE: XL/2 **2.93**, XL/2+ **2.20**。对比 250×2 NFE 的 DiT 2.27 和 SiT 2.06
  (正文写 2.15)。
- CIFAR-10 (uncond, 1-NFE): 2.92。
- 1-NFE FID 随模型变大而下降: B/2 6.17 → M/2 5.01 → L/2 3.84 → XL/2 3.43。

**和前人工作的区别**:
- Consistency Models 相当于固定 r ≡ 0, 只有单时间变量。
- Shortcut 和 IMM 有两个时间变量, 但要另加 self-consistency 约束。
- MeanFlow 只靠平均速度的定义推出训练目标, 不加额外假设。

## 2. JiT: Back to Basics, Let Denoising Generative Models Denoise (Tianhong Li, Kaiming He; arXiv 2511.13720)

**一句话**: 网络**直接预测干净图像 (x-prediction)**, 但 **loss 在 v 空间算 (v-loss)**。
这样一个普通 ViT 直接在**像素**上、用**大 patch** (16 或 32) 就能做 diffusion。
不要 tokenizer, 不要 pretraining, 不要额外 loss。JiT = "Just image Transformers"。

**约定 (和 MeanFlow 相反!)**: z_t = t·x + (1−t)·ε (Eq. 1)。**t=0 是噪声, t=1 是数据**。v = x − ε。

**动机: 流形假设 (manifold assumption, Fig. 1)**
- 干净图 x 在低维流形上。噪声 ε 和速度 v 都在 **off-manifold**, 占满整个高维空间。
- 预测 ε 或 v: 网络必须把满维的噪声信息原样保留, 需要很大容量。
- 预测 x: 网络只需输出低维结构, 把噪声滤掉。
- 高维 patch 是问题所在: 16×16×3 = **768 维**, 正好等于 JiT-B 的宽度 768;
  32×32×3 = 3072 维。在 latent 空间里每个 token 维度很小, 问题只是被掩盖了, 没有被解决。

**Toy 实验 (Fig. 2)**:
- 设置: 2 维数据 (螺旋), 用随机的列正交投影 P (PᵀP = I) 埋到 D 维, D ∈ {2, 8, 16, 512}。
- 模型: 5 层 ReLU MLP, 隐藏层 256 维, 三种预测都用 v-loss。
- 结果: D 变大时 **ε-pred 和 v-pred 崩溃** (D=512 时彻底失败), **只有 x-pred 一直正常**。

**训练 (Alg. 1, Eq. 6)**:
```
z = t*x + (1-t)*e
v = (x - z)/(1-t)
x_pred = net(z, t)
v_pred = (x_pred - z)/(1-t)
loss = ‖v − v_pred‖²
```
- 1/(1−t) 的分母**裁剪到 0.05**。v-loss 等价于加权的 x-loss, 权重是 1/(1−t)²。
- **没有 stop-grad, 没有 teacher**: target 只由数据和噪声决定。
- EMA 只用于评估 (存 3 份 decay, 选 FID 最好的那份)。
- t 用 logit-normal 采样 (μ = −0.8, σ = 0.8)。

**预测空间和 loss 空间可以分开**: 3 × 3 = 9 种组合 (Tab. 1)。
推理时总是先把输出换算成 v, 再解 ODE: **50 步 Heun**, 加 CFG interval。

**架构 (Fig. 3)**:
- 就是 DiT 直接作用在像素 patch 上: p×p×3 的 patch → 线性嵌入 → Transformer → 线性头还原成 p×p×3。
- 条件 t 和 class 用 adaLN-Zero 注入。
- 分辨率和 patch 一起放大, 序列长度保持 256: p=16 @256, p=32 @512, p=64 @1024。
- **Bottleneck patch embedding**: 把线性嵌入换成低秩的两层线性层 (patch → 128 → hidden)。
  这样能**提升** FID, 8.62 → 7.35 (d′=64) 或 7.48 (d′=128), 16 维也不会崩。
- 其他改动: SwiGLU, RMSNorm, RoPE, qk-norm, 32 个 in-context class tokens。

**核心数字 (Tab. 2a, JiT-B/16 @256, 768 维 patch, FID)**:

| | x-pred | ε-pred | v-pred |
| --- | --- | --- | --- |
| x-loss | 10.14 | 379.21 | 107.55 |
| ε-loss | 10.45 | 394.58 | 126.88 |
| v-loss | **8.62** | 372.38 | 96.53 |

- **只有 x-pred 能用**, 无论哪种 loss。
- 低维对照: B/4 @64 (patch 只有 48 维) 时 9 种组合**都能用** (3.46 到 6.20)。
  说明问题出在**高维**, 而不是 ε-pred 本身。
- 调噪声 shift (μ) 或改 loss 加权都救不了 ε-pred 和 v-pred (始终 ≥ 96)。

**Headline (600 epoch, 50 步 Heun)**:
- 256: B/16 3.66, L/16 2.36, H/16 1.86, **G/16 1.82**。
- 512: **G/32 1.78**。
- 1024: B/64 4.82。

**局限 / 讨论**:
- x-prediction 不是新东西 (DDPM 代码里就有), 本文强调的是它在**高维**下必不可少。
- 加分类 loss 还能更好, 但作者故意不用。
- 作者希望推广到 tokenizer 难设计的领域: 蛋白质、分子、天气。

## 3. Drifting: Generative Modeling via Drifting (Deng, H. Li, T. Li, Du, Kaiming He; arXiv 2602.04770)

**一句话**: 生成器 f 一次前向就把噪声变成样本: x = f(ε), 推前分布 (pushforward) 是 q = f#p_ε。
"分布逐步演化"发生在**训练时** (每一步 SGD 更新 f, 样本就"漂移"一点), 而 diffusion/flow 是在**推理时**多步积分。
所以推理天然是 **1-NFE**, 不需要 ODE/SDE, 也**不是对抗训练** (non-adversarial, 不同于 GAN)。

**漂移场 (drifting field) V_{p,q}(x)** (Eq. 8 到 12, 受 mean-shift 启发):
- **吸引 (attraction)** 指向真实数据: V⁺ = (1/Z_p)·E_{y⁺~p}[ k(x,y⁺)·(y⁺ − x) ]。**正样本 y⁺ = 真实数据**。
- **排斥 (repulsion)** 推离生成样本: V⁻ = (1/Z_q)·E_{y⁻~q}[ k(x,y⁻)·(y⁻ − x) ]。**负样本 y⁻ = 生成样本** (直接复用同一 batch: `y_neg = x`)。
- **V = V⁺ − V⁻**。
- 核函数 k(x,y) = exp(−‖x−y‖/τ), 用 ℓ2 距离 (不平方), τ 是温度。实现上做 softmax 归一化 (像 InfoNCE)。

**反对称 → 平衡 (Prop. 3.1)**:
- V_{p,q} = −V_{q,p}。所以 **q = p 时 V = 0**: 吸引和排斥互相抵消, 样本不再漂移。
- 注意: 逆命题 (V = 0 ⇒ q = p) **没有被证明**, 作者把它列为局限。

**Loss (Eq. 6, Alg. 1)**:

&nbsp;&nbsp;&nbsp;&nbsp;**L = E_ε ‖ f_θ(ε) − stopgrad( f_θ(ε) + V(f_θ(ε)) ) ‖²**

- **stop-grad 包住整个 target x + V**: V 依赖 q_θ, 不穿过它反传。做法像 SimSiam。
- loss 的值等于 ‖V‖², 梯度 = −2V, 也就是把样本沿 V 推。
```
x = f(e); y_neg = x
V = compute_V(x, y_pos, y_neg)
x_drifted = stopgrad(x + V)
loss = mse_loss(x - x_drifted)
```

**特征空间 (feature space, §3.4)**:
- V 不在原始空间算, 而是在**预训练自监督编码器 φ 的多尺度特征**里算: L = Σ_j ‖φ_j(x) − sg(φ_j(x) + V(φ_j(x)))‖²。
- 默认 φ 是 **latent-MAE** (ResNet 风格, 在 SD-VAE latent 上做 MAE 预训练), 再加分类微调。
- **φ 只在训练时用**。**没有 φ, ImageNet 上训不起来**。
- 和 perceptual loss 的区别: perceptual loss 需要把 x 和 x_target 配对; drifting 不需要配对, 而是匹配分布 φ#q ≈ φ#p。
- 多尺度特征 (每个位置 + 全局/patch 的 mean 和 std), 多温度 τ ∈ {0.02, 0.05, 0.2}, 每个特征单独算 loss 再求和。

**CFG (§3.5)**: **完全在训练时完成**。
- 把随机类别的真实图像 (近似无条件数据) 当作**额外负样本**。
- 结果: q(·|c) = α·p(·|c) − (α−1)·p(·|∅), α = 1/(1−γ)。
- α 作为条件输入网络 f(ε, c, α)。推理时 α 任意选, 仍是 1-NFE。

**生成器**:
- DiT 风格 (B/2, L/2), 在 **SD-VAE latent** (32×32×4) 上生成。
- 也有 **pixel** 版本 (L/16 在 256×256×3 上)。
- 正样本来自 MoCo 式的每类样本队列 (sample queue)。生成器有 EMA。

**数字**:
- **Tab. 1 反对称消融** (B/2):
  - 正常 V⁺ − V⁻: **8.46**
  - 1.5× 吸引 41.05, 1.5× 排斥 46.28
  - 2× 吸引 86.16, 2× 排斥 112.84
  - **只有吸引: 177.14**
  - 结论: 破坏反对称就失败。
- **Tab. 2**: 正、负样本数越多越好。N_pos 从 1 到 64: 20.43 → 8.46。像对比学习。
- **Tab. 3 编码器**: SimCLR 11.05, MoCo-v2 8.41, latent-MAE 8.46 → 更宽、训更久、加 cls ft 后 **3.36**。
- **Headline (ImageNet 256, 1-NFE)**:
  - **latent: L/2 1.54** (B/2 1.75)。
  - **pixel: L/16 1.61** (B/16 1.76)。
  - 对比: iMF 1.72, MeanFlow 3.43, StyleGAN-XL 2.30 (算力是 Drifting 的约 18 倍)。
- 最优 CFG: L/2 在 α = 1.0 (即不用 CFG), B/2 在 α = 1.1。
- **机器人 (Tab. 7)**: 用 Drifting Policy (**1 NFE**) 替换 Diffusion Policy (100 NFE)。
  - 漂移**直接在控制表示上算, 不用特征编码器**。
  - 大多数任务持平或更好, 少数略差。
- **Toy (Fig. 3)**: 双峰目标, 无论 q 怎么初始化 (包括塌缩在一个峰上), 都**没有 mode collapse**。

**局限**:
- 只证明了 q = p ⇒ V = 0, 反方向理论不清楚。
- 核函数、特征和架构都未必最优。
- **依赖预训练特征编码器**。

## 4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion (Boyuan Chen et al., NeurIPS 2024; arXiv 2407.01392)

**一句话**: 给序列里**每个 token 一个独立的噪声水平 (independent per-token noise level)**, 训练一个**因果 (causal)** 的去噪模型。
它把 next-token prediction (teacher forcing) 和 full-sequence diffusion 统一起来, 同时拿到两边的好处。

**两边各缺什么**:
- **Teacher forcing**: 不能做 guidance; 连续数据 (视频) 一旦超过训练长度, 误差累积就会**发散**。
- **Full-sequence diffusion**: 非因果, 所有 token 共用同一个噪声级; 只能生成固定长度, 子序列生成和 guidance 都受限。
- **DF**: guidance、tree search、compositionality、causal uncertainty、flexible horizon 全都支持 (Fig. 1)。

**Noise as masking (§3.1)**:
- 噪声级 k = 0 表示完全可见, k = K 是白噪声, 等于**完全被 mask**。
- Teacher forcing = 沿**时间轴** mask; full-seq diffusion = 沿**噪声轴** mask; DF 两个轴都可以。

**训练 (Alg. 1, Eq. 3.1)**:
- 每条序列、每个 t **独立均匀采样 k_t** (k_{1:T} ~ uniform [K]^T)。
- 对每个 token 单独做 forward diffuse 得到 x_t^{k_t}。
- 模型是**因果 RNN** (Causal DF): latent 更新 z_t = p_θ(z_t | z_{t−1}, x_t^{k_t}, k_t)。
  视频实现是 U-Net + GRU, z 是 GRU 的 hidden state。
- 预测噪声 ε_θ(z_{t−1}, x_t^{k_t}, k_t), **loss = Σ_t ‖ε_t − ε_θ‖²** (MSE)。
- 实际实验里视频用 v-param, planning 用 x0, 都是同一套框架。
- **没有 stop-grad, 没有 EMA teacher**, 梯度沿确定性的 z 做 BPTT。
- **理论 (Thm. 3.1)**: 优化的是一个**重加权的 ELBO**, 在适当条件下**同时**给所有噪声级序列的似然提供下界。
  取 k ∈ {0, K} 就相当于可以 mask 任意 token, 所以模型学到训练集**所有子序列 (all sub-sequences)** 的分布。

**采样 (Alg. 2, scheduling matrix K ∈ [K]^{M×T})**:
- 列 = 时间步 t, 行 = 去噪迭代 m。从全噪声开始, 一行一行往下, 把每个 token 去噪到 K_{m,t}, 最后一行全为 0。
- **换一个 K 就换一种采样方式, 不需要重新训练**:
  - full-sequence: 所有 token 同步去噪;
  - autoregressive: 一个 token 去噪完再去下一个;
  - **pyramid / zig-zag (causal uncertainty)**: 越远的未来噪声越高。
    例子: [x1^K, x2^K, x3^K] → [x1^0, x2^{K/2}, x3^K] → [x1^0, x2^0, x3^{K/2}] → 全为 0。
- **长 rollout 的稳定化 (stabilization)**: 已生成的 clean token **不当作 ground truth**,
  而是告诉模型它带**一点噪声 (small noise level 0 < k ≪ K)**, 再送进 latent。
  这样模型不会把自己的误差当真值, 就**不会发散**。
- **Flexible horizon**: 不用 sliding window 也能超出训练长度生成。

**Guidance 与决策 (§3.4)**:
- 每个 token x_t = [a_t, r_t, o_{t+1}]: action, reward, 下一个观测。
- 以 z 为条件采样一段计划, 执行第一个 action, 再更新 z (类似 MPC)。
- 用 reward 的梯度引导未来 token。因为未来还带着噪声, 梯度可以影响整个计划。
- **MCG (Monte Carlo Guidance)**: 引导某个 token 时, **采样多条可能的未来, 对它们的 guidance 梯度取平均**,
  这样优化的是**期望 reward**。
  Full-seq diffusion 做不了 MCG: 所有 token 同步去噪, 没有"未来仍不确定"可以拿来采样。

**实验 + 数字**:
- **视频 (Minecraft, DMLab)**: 两个 baseline 用同一个 RNN, 分别是 next-frame diffusion (teacher forcing) 和 causal full-seq diffusion。
  它们都很快**发散**, full-seq 还会出现帧间跳变。DF 生成远超训练长度仍**稳定** (训练 72 / 36 帧, 测到 1000+ 帧)。
- **Planning (D4RL Maze2D, sparse reward, Tab. 1)**:
  - Single-task 平均: **DF 141.7**, 去掉 MCG 129.67, Diffuser 119.5。
    但 Diffuser 要靠手写 PD controller, 用它自己生成的 action 只有 8.7, 因为它生成的 state 和 action 不因果一致。
  - Multi-task 平均: **DF 146.2**, Diffuser 129.4。
- **Compositional**: 训练数据是十字形轨迹。保留完整 memory 时复现十字; 去掉 memory 用短计划时, 能把子轨迹拼成 V 形。
- **机器人**: Franka 机械臂借第三个槽交换苹果和橘子, 必须**记住初始位置** (需要 memory)。
  - **DF 成功率 80%**; Diffusion Policy 没有 memory, 失败。
  - 视觉干扰或遮挡时: 把这些观测**标成 k > 0 (当作带噪)**, DF 仍有 **76%**; next-frame diffusion 只有 **48%**。
- **时间序列 (GluonTS, CRPS_sum)**: 和 SOTA 有竞争力。

**局限**: 基于 RNN, 规模小; 更高分辨率需要 transformer, 也没有研究 internet-scale 的扩展性。

## 5. 模拟题 (仿 Quiz 1 格式, 先遮住答案自测)

**JiT**
1. JiT 网络直接输出什么? loss 在哪个空间算?
   A. ε, ε-loss  B. v, v-loss  C. x, x-loss  D. x, v-loss
   → **D**
2. 为什么 ε-pred 和 v-pred 在 JiT-B/16 @256 上失败, 但在 B/4 @64 上能用?
   → 噪声和 v 都在 off-manifold, 是满维的。一个 patch 有 768 维, 约等于网络宽度, 网络保留不下来;
   而 x 在低维流形上。B/4 的 patch 只有 48 维, 远小于宽度, 所以都能用。
3. Toy 实验怎么设计?
   → 2 维数据经随机列正交矩阵埋到 D 维, D 最大到 512; 用 MLP 训练。只有 x-pred 在大 D 时还正常。

**MeanFlow**
1. 写出 MeanFlow identity。
   → u(z_t, r, t) = v(z_t, t) − (t−r)·(d/dt) u(z_t, r, t)
2. JVP 的 tangent 是什么? 对应哪个 Jacobian?
   → **(v, 0, 1)**, 对应 (∂_z u, ∂_r u, ∂_t u)。用错的 tangent 会崩: (v,0,0) 268, (v,1,0) 329。
3. stop-grad 放在哪里? 为什么?
   → 放在 u_tgt 上, 避免 double backprop 穿过 JVP (不需要高阶梯度)。
4. 一步采样公式是什么? 什么情况下退化为 Flow Matching?
   → z_0 = z_1 − u(z_1, 0, 1); r = t 时退化为 Flow Matching (Tab. 1a 中 0% 的那一行, FID 328.91)。
5. 训练图填空: (1) target 的公式, (2) 哪里加 sg, (3) 采样几步。
   → (1) v − (t−r)(v∂_z u + ∂_t u); (2) target 上; (3) 1 NFE。

**Drifting**
1. 漂移场由哪两项组成? 正样本、负样本各是什么?
   → V = V⁺ − V⁻。**吸引项**指向真实数据 (正样本), **排斥项**推离生成样本 (负样本, 复用同一 batch)。
   核函数 k = exp(−‖x−y‖/τ)。
2. 为什么 q = p 时训练停止?
   → V 反对称: V_{p,q} = −V_{q,p}, 所以 p = q ⇒ V = 0。
   只保留吸引项时 FID 是 177.14 (正常是 8.46)。
3. 写出 loss 并标出 stop-grad。
   → ‖f(ε) − **sg**(f(ε) + V(f(ε)))‖²。loss 的值等于 ‖V‖²。
4. V 在什么空间计算?
   → 预训练自监督编码器 (latent-MAE) 的多尺度特征空间, **只在训练时用**; 没有它 ImageNet 训不起来。
   注意: 这**不是** perceptual loss, 因为不需要配对。
5. 推理 NFE 是多少? CFG 怎么做?
   → **1 NFE**。CFG 在训练时完成: 真实无条件样本当作额外负样本, α 作为网络输入。

**Diffusion Forcing**
1. 训练时噪声级怎么分配?
   A. 所有 token 同一个 k  B. 随 t 线性增加  C. 每个 token 独立均匀采样  D. 过去设 0, 未来设 K
   → **C**
2. 长视频 rollout 怎么避免发散?
   → 把已生成的帧当作带**一点噪声** (0 < k ≪ K) 的 token 送进模型, 而不是当作 clean ground truth。
3. 什么是 Monte Carlo Guidance? 为什么 full-seq diffusion 做不了?
   → 采样多条未来, 对 guidance 梯度取平均 (优化期望 reward)。
   full-seq 所有 token 同步去噪, 未来没有可以采样的不确定性。
4. 机器人实验中相机被遮挡, DF 怎么处理?
   → 把该观测标为 k > 0 (带噪), 依赖模型的 prior。成功率 76%, baseline 48%。
5. 每个 token 不同噪声级, 对应哪两种特例?
   → 所有 k 相同 = full-sequence diffusion; 过去 k = 0、当前 token 去噪、未来 k = K = teacher forcing / next-token。

---
来源: doc/08 到 11 的 PDF。每个数字都对照论文正文或表格核对过 (2026-10-28)。
