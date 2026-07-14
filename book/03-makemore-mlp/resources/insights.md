# lecture

这节课表面上是在实现一个只有一层隐藏层的字符语言模型，真正的主线却是：如何把“逐个样本、逐个神经元”的数学公式，变成一个可以批量计算、自动求导、优化和正确评估的训练系统。

如果只记住一句话，我会记住：

> 不要在脑中模拟几万个神经元的每一次标量运算；为 tensor 的每个轴赋予语义，检查相邻运算的 shape contract，然后让矩阵乘法和 autograd 处理其余工作。

## 1. 为什么从 bigram 的计数表转向神经网络？

bigram 只有一个字符的 context，因此计数表只有 27 行。如果直接把计数法推广到更长 context，可能的 context 数量会指数增长：

```text
context 1: 27
context 2: 27² = 729
context 3: 27³ = 19,683
context T: 27ᵀ
```

大部分长 context 在有限数据中根本没有出现，即使出现，计数也很少。问题不仅是表太大，更重要的是计数表无法在相似 context 之间共享统计强度：没见过的那一行就是没见过。

Bengio et al. (2003) 的关键想法是学习 distributed representation：先把离散 token 映射到连续 embedding，再让 MLP 根据这些表示预测下一个 token。这样，一个训练样本不再只更新一条孤立的 n-gram 记录；它会更新共享的字符表示和共享的网络权重，从而影响许多相似 context。

在词级例子中，模型即使没有见过精确的 `A dog was running in a ...`，也可能从 `the/dog/cat/walking/running` 的邻近表示中迁移知识。在字符模型中，`a/e/i/o/u` 因为出现在相似的名字位置和 context 中，也可能获得相近表示。

这里真正对抗“维数灾难”的不是神经网络会记住更多组合，而是：

> 参数共享和连续表示让相似输入共享学习结果，从“逐行记忆”变成“通过表示泛化”。

MLP 的参数量仍会随 context 增长，但第一层大约按 `context_length × embedding_dim × hidden_dim` 线性增长，不再像完整计数表那样按 `vocab_size^context_length` 指数增长。

## 2. 一个名字如何变成监督学习数据？

`block_size = 3` 表示输入三个字符、预测第四个字符。`.` 同时承担 padding/start token 和 end token 的角色。

例如 `emma` 被展开为：

```text
... -> e
..e -> m
.em -> m
emm -> a
mma -> .
```

长度为 `L` 的名字会产生 `L + 1` 个预测样本，因为还要学习名字在何时结束。每一步都把 context 左移一格，再追加刚刚看到的字符：

```python
context = context[1:] + [ix]
```

这个构造揭示了语言模型的基本训练形式：给定过去，预测下一个 token。生成时做的其实是同一件事，只不过真实的下一个字符不再提供给模型，而是把模型自己采样出的字符放回 context。

还有一个容易忽视的数据原则：应该先按完整名字划分 train/dev/test，再从各个 split 构造字符样本。否则同一个名字的前半段可能进入 train、后半段进入 dev，造成明显泄漏。课程的代码遵守了这个顺序；但原始文件中有重复名字，因此严格研究时还应先去重，或把相同名字作为同一 group 划分。

## 3. embedding lookup 其实就是一个特殊的线性层

对字符 `i`，下面两种写法数学上完全相同：

```python
C[i]
F.one_hot(i, num_classes=27).float() @ C
```

one-hot 里只有第 `i` 个位置为 1，所以矩阵乘法恰好取出 `C` 的第 `i` 行。embedding 并不是神秘的新运算，它是 one-hot 线性层的一种高效实现：既然我们知道只有一行会被选中，就没有必要真的构造 27 维 one-hot 再做大量乘零操作。

更深的两点是：

1. 同一个矩阵 `C` 被所有样本和所有 context 位置共享。任何位置上出现的字符都会训练同一行 embedding。
2. `C` 不是预先规定的字母知识，而是与 `W1/W2` 一起通过 next-character loss 学出来的。它保存的是“对预测下一个字符有用的相似性”。

因此 `a/e/i/o/u` 聚在一起，不是模型发现了字典中的“元音定义”，而是这些字符在训练数据中的预测作用具有可替换性。`q` 和特殊字符 `.` 离得很远，也表示它们具有较特殊的条件分布。

二维图很直观，但不要过度解释坐标本身。embedding 的绝对方向没有固定语义；对 embedding 空间作可逆变换，同时对下一层权重作逆变换，网络仍然可以表达同一个函数。旋转和反射甚至会在保持距离的同时彻底改变坐标轴，更一般的可逆变换还可能改变距离。因此，字符聚类只能算当前这组参数中很有启发性的 diagnostic，不是模型函数唯一决定的“真实语言地图”。维度变高后也不能直接看前两维就下结论，通常需要 PCA 等投影，而且投影仍只是诊断工具。

## 4. batch tensor：一次计算 N 个样本，而不是写 N 次网络

课程先用 3 个字符、每个字符 2 维 embedding，因此完整的 shape 流是：

```text
X       : (N, 3)       N 个样本，每个样本 3 个字符索引
C       : (27, 2)      27 个字符的 2 维 lookup table
emb=C[X]: (N, 3, 2)    每个索引被替换成 2 维向量
flat    : (N, 6)       每个样本的 3×2 个特征拼起来
W1      : (6, H)
h       : (N, H)
W2      : (H, 27)
logits  : (N, 27)
Y       : (N,)
loss    : scalar
```

所以你说的 `N × 6` 是对的。更准确地说，`N` 是 batch/sample 轴，`6 = 3 × 2` 是每个样本进入隐藏层的 input features。一次：

```python
h = torch.tanh(flat @ W1 + b1)
```

就把同一个 `W1` 应用到了全部 N 个样本。底层可以把它交给高度优化的矩阵乘法，而不是在 Python 中循环 N 次。GPU/CPU 不需要 N 份模型；所有样本共享同一份参数，只是在 batch 轴上并行。

这也是学习深度学习代码时最有用的抽象方法之一。不要追踪“第 73 个样本的第 418 个神经元现在是多少”，而是追踪相邻模块的契约：

```text
embedding : (N, T)     -> (N, T, D)
flatten   : (N, T, D)  -> (N, T·D)
hidden    : (N, T·D)   -> (N, H)
output    : (N, H)     -> (N, V)
loss      : (N, V),(N) -> scalar
```

但“只看 shape 就永远不会乱”还需要一个修正：相同 shape 也可能有完全不同的语义，例如 `(N, T, D)` 中交换了时间轴和 embedding 轴，shape 有时仍然碰巧合法。可靠做法是同时记录：

- shape；
- 每个轴的语义，例如 `N=batch, T=context, D=embedding, H=hidden, V=vocab`；
- dtype，例如索引和 target 通常是 `long`，权重和 logits 是浮点数；
- 必要的不变量，例如 target 必须在 `[0, V)`。

实际代码里可以使用有语义的变量名、shape 注释、`assert`，并用 `block_size * n_embd` 代替硬编码的 `6` 或 `30`。

## 5. `view` 为什么快，以及它并非无条件零拷贝

`emb` 的逻辑形状是 `(N, 3, 2)`，隐藏层需要 `(N, 6)`。一种方法是切出三个 `(N, 2)` tensor 再 `cat`，但 `cat` 必须为拼接结果分配新的 storage 并复制数据。

对于课程中连续存储的 `emb`：

```python
flat = emb.view(-1, 6)
```

只改变 tensor 的 shape/stride 等元数据，用另一种方式解释同一块底层 storage，不移动元素，因此非常便宜。`-1` 表示让 PyTorch 根据总元素数自动推断 batch size，也避免硬编码 N。

现代 PyTorch 中必须补充一个边界：只有新 shape 与当前 size/stride 兼容时，`view` 才能这样工作。`transpose`、某些切片等操作会得到 non-contiguous tensor，此时 `view` 可能报错。`reshape` 会尽可能返回 view，做不到时才复制；`flatten(1)` 在模型代码里往往最能表达“保留 batch 轴、展开后面的特征轴”，但同样不应该依赖它必定零拷贝。

所以真正的规则是：

> view 是“共享 storage 的不同解释”，不是通用 reshape 魔法；先理解 shape，再留意 stride 和 contiguous 性。

## 6. broadcasting：短代码背后也可能藏着 silent bug

隐藏层的矩阵乘法结果是 `(N, H)`，bias 是 `(H,)`：

```python
flat @ W1 + b1
```

PyTorch 从右侧对齐维度，把 `(H,)` 视作 `(1, H)`，再把同一个 bias row 广播到 N 个样本。这正是我们需要的行为。

但 broadcasting 的危险也在于“代码能运行”不代表“语义正确”。如果某个维度碰巧相等，错误对齐可能不会报错。Karpathy 在课堂上停下来核对 broadcasting，不是多余的细节，而是一种重要习惯：

> 每次依赖隐式 broadcasting，都要能明确说出哪些轴被复制、为什么这正是想要的计算。

## 7. logits、softmax 与 cross entropy 是同一个概率故事

输出层给每个样本产生 27 个 logits：

```text
logits: (N, 27)
```

logit 是任意实数分数，不需要为正，也不需要和为 1。手写概率过程是：

```python
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
loss = -probs[torch.arange(N), Y].log().mean()
```

对单个样本，更紧凑的数学形式是：

```text
loss_i = -log softmax(logits)[target]
       = -z_target + log Σ exp(z_j)
```

`F.cross_entropy(logits, Y)` 直接完成这件事。注意应传入 raw logits，不要先手动调用 softmax。

### 为什么应使用 PyTorch 的 cross entropy？

课堂讲到的三个理由都非常实用：

1. **少建中间 tensor，节省内存和调度开销。** 不必显式保存 `exp(logits)`、`counts`、`probs`、`log(probs)` 等完整结果。
2. **forward 和 backward 可以使用专门优化的实现。** 后向不需要让 autograd 逐个穿过手写的 `exp/div/log`；解析梯度可以被化简，具体 backend 还可能使用融合或其他优化 kernel。这里准确的词是“融合/优化实现”，不是分布式计算意义上的“集群运算”。
3. **数值稳定。** softmax 对所有 logits 加减同一个常数后概率不变，因此实现可以先减去该行最大 logit。最大的值变成 0，其余值不大于 0，避免 `exp(100)` 一类浮点溢出。等价地，它使用稳定的 log-sum-exp 计算。

这是一条普遍工程原则：

> 为了学习，可以把复合运算拆开；实际训练时，应优先使用框架提供的稳定复合算子。

cross entropy 也不只是“判断预测对不对”。同一个 context 可能对应多个合理的下一个字符，它要求模型学习完整的条件概率分布，并惩罚对真实结果的错误自信。因此它同时影响准确性、概率校准和之后的采样质量。

## 8. 为什么先故意 overfit 一个 tiny batch？

课程一开始只用少量名字构造约 32 个样本，却给模型约 3,400 个参数。训练 loss 很快降得非常低，这不是最终目标，而是一个非常有用的 sanity check：

> 如果一个具有大量参数的模型连一个 tiny batch 都拟合不了，优先怀疑数据、target、forward、backward、学习率或参数更新存在 bug，而不是马上训练完整数据集。

这种“先 overfit 一个 batch”的调试法今天仍然非常实用。

但这里即使参数很多，loss 也不能严格降到零，因为完全相同的输入可能有多个 label：

```text
... -> e
... -> o
... -> a
...
```

输入相同时，确定性的网络必须输出同一组概率，不可能同时给每个不同 label 都分配概率 1。最优解是学习这些开头字母的经验条件分布。

这揭示了一个比“过拟合”更深的概念：

> 数据本身含有条件不确定性；不是所有非零 loss 都代表模型能力不足或训练失败。

## 9. mini-batch：不是错误梯度，而是 full gradient 的随机估计

完整训练集约有 20 多万个字符样本。每一步都对全部样本 forward/backward，可以得到当前数据集上的精确梯度，但单步成本太高，因此单位时间内只能走很少几步。

均匀随机抽取 batch 后，mini-batch gradient 是 full gradient 的随机估计。在常见条件下它是无偏的：重复很多次取平均，会回到 full gradient。单个 batch 的方向会有噪声，但通常仍包含足够的下降信号。

因此课堂中的核心取舍是：

```text
full batch
  每步方向更精确
  每步很昂贵，更新次数少

mini-batch
  每步方向更嘈杂
  每步便宜，可以快速做很多次更新
```

Karpathy 的判断是：与其很久才取得一次精确梯度，不如快速取得很多个“足够好”的梯度。在这个模型里，batch size 32 已经足以明显降低 loss。

mini-batch loss 也只是当前随机样本的 loss，所以训练曲线看起来很“厚”。它适合生成梯度和观察大趋势，不适合当作最终模型指标。报告 train/dev loss 时，应该切到评估模式并在完整 split 或足够大的固定子集上计算；绘图时可以对 minibatch loss 做滑动平均。

### 今天再看 batch size：它不只有“精确度”一个轴

batch 越大，梯度方差通常越小，矩阵运算也更容易填满 GPU；但显存占用更大，每处理相同数量数据所做的参数更新更少，而且学习率通常也要一起调整。超过某个任务相关的 critical batch size 后，继续增大 batch 往往只有有限收益。

所以 batch size 同时平衡：

- 梯度噪声；
- 每秒吞吐量和硬件并行度；
- 每看多少数据更新一次参数；
- 显存；
- 学习率和优化器；
- 有时还包括泛化表现。

较新的大模型研究甚至观察到 useful/critical batch size 会随训练阶段变化，这也是 batch-size warmup 等思路的动机。对当前小 MLP 不必实现这些技巧，但应知道不存在脱离模型、数据、优化器和训练阶段的“最佳 batch size”。

## 10. autograd 训练循环中容易漏掉的两个事实

PyTorch 的 `.backward()` 默认把新梯度累加到已有 `.grad`，而不是覆盖。因此普通训练循环必须在每次新的 batch 前清空梯度：

```python
for p in parameters:
    p.grad = None
loss.backward()
```

设为 `None` 是合法且常用的做法；需要梯度的参数会在 backward 时重新创建 `.grad`。如果忘记清空，当前更新会混入之前所有 batch 的梯度，除非你明确就是要做 gradient accumulation。

第二个事实是：只有 `requires_grad=True` 的相关 leaf tensor 才会积累参数梯度。课程手写参数，所以显式设置它；使用 `nn.Linear` 等模块时，权重被包装成 `nn.Parameter`，默认会被 optimizer 管理。

课程使用 `p.data += -lr * p.grad` 是为了把梯度下降完整摊开给你看。现代项目应优先使用 `torch.optim` 的 `optimizer.step()`；若必须手写更新，应放在 `torch.no_grad()` 中，而不要依赖 `.data` 绕过 autograd 的安全检查。

## 11. 学习率不是猜一个数字：先找可用区间，再衰减

课程先尝试极小和极大的学习率：

- 太小：loss 几乎不动；
- 合适：loss 快速下降；
- 太大：loss 剧烈震荡，甚至发散。

然后把 learning rate 在对数尺度上从 `10^-3` 扫到 `10^0`，画出 loss 随 `log10(lr)` 的变化。对学习率按指数取样很重要，因为 `0.001 → 0.01` 与 `0.1 → 1.0` 都是十倍变化；线性网格会把大部分候选浪费在高学习率一端。

这次 valley 表明 `0.1` 是合理量级。训练后期 loss 接近平原时再降到 `0.01`，较小步长可以继续精调，而不在较好的区域附近来回跳动。

这里还有两个实践 insight：

1. 模型变大后收敛可能更慢，短暂训练结果差不能立刻证明架构差；应先确保不同候选都被合理优化。
2. batch size、模型宽度和参数尺度改变后，原来的 learning rate 也未必仍然合适。优化超参数是联动的。

课堂里的手动两段式 decay 很适合建立直觉；正式实验通常会把步数、warmup、decay schedule 都显式配置并记录，像 exercise 中的 warmup + cosine decay。

## 12. train/dev/test：其实有三层“学习”

三个 split 的角色可以这样理解：

```text
train: 梯度下降学习 parameters
dev:   实验者通过比较结果学习 hyperparameters 和设计选择
test:  模型与实验者都未用过的最终审计
```

parameters 是 `C/W1/b1/W2/b2` 中通过 backprop 更新的数值。hyperparameters 不只包括隐藏层宽度和 embedding 维度，还包括：

- context length；
- batch size；
- learning rate 和 schedule；
- 训练步数；
- 初始化；
- dropout/weight decay；
- 优化器；
- 什么时候停止训练。

这正是 exercise 中所有改动应该用 dev 比较的原因。

每查看一次 test 结果并据此修改模型，就有一点关于 test 的信息泄漏进了设计过程。次数足够多后，实验者同样可以 overfit test。即使从未对 test 调用 `.backward()`，仍然可能发生这种“人类在外环中训练”的过拟合。

dev 也不是无限使用而不会过拟合。如果进行了大量实验、差异又很小，最好使用多个随机种子、交叉验证或保留一个新的 holdout 来复核。test 应尽量只在全部选择冻结后使用一次。

此外，split 最重要的不是机械地遵守 `80/10/10`，而是模拟部署时的泛化问题。随机划分适合课程中的同分布名字生成；时间预测、用户个性化或跨域任务可能需要按时间、用户或来源分组划分。

## 13. 如何从 train/dev loss 判断 underfit 与 overfit？

课程先得到 train 和 dev 都约为 `2.3`。两者接近并不表示模型已经很好，而表示它连训练数据都没有充分拟合：当前主要是 underfitting，可以尝试增加容量。

于是隐藏层从 100 增加到 300。但更宽后提升仍有限，Karpathy 没有只盯着 hidden layer，而是怀疑 2 维 embedding 已经成为新的瓶颈。把 embedding 增加到 10 维、隐藏层设为 200，并继续合理优化后，最终得到：

```text
train loss ≈ 2.1260
dev loss   ≈ 2.1701
```

现在 train/dev 开始缓慢分离，说明容量增加确实降低了训练误差，同时过拟合也开始出现。

因此诊断不能只看 gap，还要同时看绝对水平和变化趋势：

```text
train 高，dev 高，二者接近
→ 更像欠拟合或优化不足

train 持续下降，dev 停滞或上升
→ 更像过拟合

train 本身降不动
→ 检查信息、容量、初始化、学习率和实现

train 很低，dev 明显更高
→ 优先考虑数据、正则化和容量控制
```

这正是 exercise 后续调参的逻辑来源：先扩大 context、embedding 和 hidden capacity；当 train/dev gap 扩大后，再加入 dropout 和 weight decay。

## 14. embedding 图中的 `aeiou` 到底说明了什么？

二维 embedding 训练后，`a/e/i/o/u` 靠近，`q` 和 `.` 较独立。它说明反向传播不只是训练最后的分类层，也在重新组织输入表示，使对预测任务作用相似的字符获得相似向量。

这带来一种组合式泛化：如果某个 context 对 `a` 的表示学到了一种有用响应，附近的 `e/i/o/u` 可能产生相似隐藏激活，而不必为每个字符组合从零学习。

但是应避免三种过度解释：

1. 邻近表示的是任务中的 predictive similarity，不等于所有语言学意义都相同。
2. 二维空间是非常强的瓶颈，图像清楚不代表模型最好；增加到 10 维后 loss 更好，却无法直接画出来。
3. embedding 是和后续权重共同定义的；单独看坐标不能完整解释模型行为。

所以 embedding visualization 最适合用于：发现明显结构、异常 token、塌缩或数据问题，而不是给每一个维度编故事。

## 15. 从训练到实际生成：batch 轴变成 1，模型没有变

训练时输入是 `(N, T)`；生成一个名字时输入变成 `(1, T)`：

```text
context  : (1, T)
embedding: (1, T, D)
hidden   : (1, H)
logits   : (1, V)
probs    : (1, V)
```

shape contract 完全相同，只有 batch size 从 N 变成 1。这很好地说明 batch dimension 是外层并行轴，而不是模型逻辑的一部分。

生成循环是：

1. 从全 `.` context 开始；
2. forward 得到 logits；
3. 用稳定的 `softmax` 得到概率；
4. 用 `torch.multinomial` 采样，而不是永远取 argmax；
5. 左移 context 并追加新字符；
6. 采到 `.` 时停止。

采样保留了模型学到的分布和多样性，argmax 则容易反复走同一条最高概率路径。生成结果适合做 qualitative sanity check，但“听起来像名字”不能代替 dev/test NLL。

在现代模块代码中，生成时还应使用 `model.eval()` 关闭 dropout 等训练态行为，并用 `torch.inference_mode()` 或 `torch.no_grad()` 避免建立反向图。要注意二者作用不同：`.eval()` 改变某些层的行为，inference/no-grad 控制 autograd。

## 16. 现在回头看，课堂代码中哪些是教学写法？

为了让每一步都可见，课堂手写了 lookup table、矩阵、bias、softmax 和 SGD。今天写实际模型时通常会作如下替换，但数学没有变化：

```text
C[X]                -> nn.Embedding
x @ W + b           -> nn.Linear
手写 softmax/NLL    -> F.cross_entropy
p.data 手动更新     -> torch.optim
手动 full evaluation -> eval() + inference_mode()
硬编码 6/30         -> block_size * n_embd 或 flatten(1)
```

抽象层次提高后，仍然要保留课堂中建立的检查能力：

- 能写出每层 shape；
- 知道 logits 与 probabilities 的区别；
- 知道 gradients 会累加；
- 知道 bias 如何 broadcast；
- 知道 loss 是否接近合理初值；
- 能先 overfit tiny batch；
- 能分别解释 train 和 dev 的变化。

框架 API 可以隐藏实现细节，但不应隐藏这些不变量。

## 17. 一份可以反复使用的思考清单

看到一个新的前向传播时：

1. 写出每个 tensor 的 shape 和轴语义。
2. 检查相邻层的 output/input contract。
3. 检查 dtype、device 和 broadcasting。
4. 区分 raw logits、probabilities、targets 和 scalar loss。

开始训练前：

1. 检查随机模型 loss 是否在合理数量级。
2. 尝试 overfit 一个 tiny batch。
3. 确认每步清空梯度，并确认参数确实在更新。
4. 用对数尺度寻找学习率量级。

进行实验时：

1. mini-batch loss 用来更新和看趋势，不当最终指标。
2. 同时看 full train loss 与 dev loss。
3. 用 dev 选择 hyperparameters，不用 test。
4. 记录 seed、split、步数、batch size 和 schedule。
5. 小于随机波动的差异不要包装成结论。

部署或采样时：

1. 保持与训练相同的 tokenization、context rolling 和 shape contract。
2. 切换到 eval/inference 状态。
3. 区分概率采样的主观质量与 held-out NLL 的定量质量。

这节 lecture 最深的 insight 最终不是某个 PyTorch API，而是一套分层思维：

> 标量数学定义了模型；tensor shape 把数学组织成批量计算；autograd 给出梯度；mini-batch 和 learning rate 决定优化效率；train/dev/test 决定我们是否真的学到了可泛化的规律。

## 参考与延伸

- [Bengio et al., *A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/v3/bengio03a.html)
- [PyTorch: Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html)
- [PyTorch: `Tensor.view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html)
- [PyTorch: `cross_entropy`](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.cross_entropy.html)
- [PyTorch: Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [McCandlish et al., *An Empirical Model of Large-Batch Training*](https://arxiv.org/abs/1812.06162)
- [Merrill et al., *Critical Batch Size Revisited*](https://arxiv.org/abs/2505.23971)

# exercise

  整条思路可以压缩成一句话：

  > 先用 train/dev loss 判断模型是欠拟合还是过拟合；先补充信息和容量，再用正则化驯服新增容量，最后用初始化和训练配置把这些容量真正训练出来。

  ## 1. 先诊断：2.17 的模型哪里不够？

  课程模型大致是：

  context = 3
  embedding = 10
  hidden = 200

  课程中的 train loss 和 dev loss 相差不大，大约是：

  train ≈ 2.12
  dev   ≈ 2.17

  这很重要。

  如果 train loss 已经特别低、dev loss 很高，那叫过拟合，应当加强正则化。

  但这里 train 和 dev 都不够好，而且差距不大，说明首先是：

  > 模型没有足够的信息或者能力，连训练集的规律都没有充分学会。

  所以第一步不应该急着加 dropout，而应该找到模型的信息瓶颈和容量瓶颈。

  我的实验逻辑大致是：

   观察                                   判断          下一步
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   train/dev 都高且接近                   欠拟合        增加信息和容量
  ─────────────────────────────────────  ────────────  ────────────────────────────────
   context 3 可能看不见足够历史           信息瓶颈      context 增加到 5
  ─────────────────────────────────────  ────────────  ────────────────────────────────
   10 维 embedding 压缩较强               表示瓶颈      embedding 增加到 24
  ─────────────────────────────────────  ────────────  ────────────────────────────────
   200 个隐藏神经元不够组合这些信息       容量瓶颈      加宽隐藏层
  ─────────────────────────────────────  ────────────  ────────────────────────────────
   加宽后 train 继续下降、dev 开始落后    开始过拟合    加 dropout、weight decay
  ─────────────────────────────────────  ────────────  ────────────────────────────────
   大模型训练不稳定或末期震荡             优化问题      改善初始化、优化器和学习率计划

  这就是整条主线。

  ## 2. context 从 3 增加到 5：先提高模型的信息上限

  模型预测下一个字符时，原来只能看到前三个字符：

  ... → next character

  无论隐藏层有多宽，它都不可能利用第四、第五个历史字符，因为这些字符根本没有输入模型。

  这是一个“信息上限”，不是增加神经元可以解决的问题。

  名字里存在不少长度超过 3 的局部模式，例如：

  -ella
  -anna
  -leigh
  -son

  当模型只看到最后三个字符时，一些不同的长前缀会被压缩成相同的 context。增加到 5 后，模型可以区分更多情况。

  所以我优先增加 context，而不是先把隐藏层从 200 暴力增加到几千。

  更深的 insight 是：

  > 容量只能加工已经提供给模型的信息，无法恢复根本没有输入的信息。

  ### 为什么不继续增加到 6 或 8？

  我实际测试了 6 和 8。它们可以进一步降低 train loss，但没有降低 dev loss。

  原因之一是每增加一个 context 字符，第一层都会增加：

  embedding_dim × hidden_dim
  = 24 × 1536
  = 36,864

  个参数。

  从 context 5 增加到 8，会额外增加约 11 万个参数。可是名字中的大多数下一个字符规律仍然比较局部。额外历史提供的有效信号有限，额外参数却很容易记住训练集里的具体名字。

  因此实验表现是：

  更多 context
  → train loss 更低
  → dev loss 没有更低

  这是非常典型的“信息收益已经小于新增方差”的信号。

  所以 5 是当前数据规模和 MLP 配置下的经验甜点，不是说自然语言永远只需要 5 个字符。

  ## 3. embedding 从 10 增加到 24：解除低秩瓶颈

  embedding 不只是“给每个字符配几个数字”。

  字符表有 27 个符号：

  .abcdefghijklmnopqrstuvwxyz

  embedding 维度为 10 时，每个字符必须被压缩到 10 个数中。

  从数学上看，对于 context 中的每一个位置，从字符 one-hot 到隐藏层的变换被分解为：

  one_hot @ embedding_matrix @ hidden_weight

  如果 embedding 只有 10 维，这个变换的秩最多是 10。也就是说，27 种字符对隐藏层产生影响的方式，被限制在一个最多 10 维的子空间里。

  将 embedding 增加到 24 后：

  最大秩：10 → 24

  它已经接近 27 个字符的完整表示能力了。

  这意味着模型既可以学到：

  a、e、i、o、u 都是元音

  这样的共享结构，也有足够空间保留：

  a 和 e 仍然是不同字符

  这样的个体信息。

  所以这里的 insight 不是简单的“embedding 越大越好”，而是：

  > 10 维 embedding 对 27 个符号和更长 context 来说形成了低秩瓶颈；24 维基本消除了这个瓶颈。

  context 变成 5、embedding 变成 24 后，隐藏层输入宽度是：

  5 × 24 = 120

  而不是课程模型中的：

  3 × 10 = 30

  现在模型接收到的信息丰富了很多。

  ## 4. 隐藏层从 200 增加到 1536：增加非线性特征的数量

  每个 tanh 隐藏神经元都可以理解成一个“软模式检测器”。

  例如某些神经元可能逐渐学会响应：

  当前位置像不像名字结尾
  前面是否出现了元音组合
  最近字符是否类似 ell
  当前是否处于名字开头
  某种 embedding 组合是否暗示下一个字符是 a

  输出层再把这些模式检测器组合起来，预测 27 个字符。

  当输入从 30 维增加到 120 维后，仍然只使用 200 个隐藏神经元，会限制模型可以同时学习的模式数量。

  因此我逐渐增加隐藏层宽度。

  一个 512-hidden、context-5 的 MLP，即使还没有 dropout，dev loss 已经可以达到大约 2.00。这说明 context 和 embedding 的改动确实击中了主要瓶颈。

  最终增加到 1536，是为了让单隐藏层拥有更多软特征。因为不能使用卷积、循环或注意力，在“只允许一层隐藏层”的条件下，加宽就是最直接的容量来源。

  参数量变化非常明显：

  课程模型：约 11,897 参数
  最终模型：228,003 参数

  最终模型的参数组成是：

  embedding: 27 × 24                  =     648
  fc1:       120 × 1536 + 1536       = 185,856
  fc2:       1536 × 27 + 27          =  41,499
                                        -------
                                        228,003

  大约是原模型的 19 倍。

  但这里马上出现了新的问题：过拟合。

  ## 5. 为什么不能只是“做得越大越好”？

  增加隐藏层后，train loss 继续下降，但 dev loss 没有以同样速度下降，train/dev gap 开始扩大。

  这说明诊断已经发生了变化：

  开始：容量不足
  后来：容量足够，但模型开始记忆训练数据

  这是调模型时非常重要的状态转换。

  如果还继续盲目增加隐藏神经元，只会让训练集越来越漂亮，验证集却不一定改善。

  因此下一步不是缩回小模型，而是：

  > 保留大模型的表示能力，同时限制它记忆训练数据的方式。

  这就是 dropout 和 weight decay 出现的原因。

  ## 6. 15% dropout：让 1536 个神经元不能互相死记硬背

  训练时，dropout 随机关闭 15% 的隐藏激活。

  对于 1536 个隐藏神经元，每一步平均有大约：

  1536 × 0.85 ≈ 1306

  个参与计算，但每一步被关闭的神经元都不同。

  它迫使模型不能依赖某一个固定的隐藏神经元组合，而要让多个特征彼此备份。

  我尝试了较小的 dropout，例如 5% 和 10%。对于最终的 1536-hidden 模型，15% 的验证结果更好。

  没有 dropout 时，这类调大的 MLP 仍然可以达到大约 2.00，但 train/dev gap 明显更大。加入适度 dropout 后，额外宽度才真正转化成了 dev loss 改善，最终接近 1.95。

  这里最值得记住的 insight 是：

  > 大模型本身不是问题；不受约束的大模型才容易成为问题。

  很多时候，“较大的模型 + 合适的正则化”会优于“为了防止过拟合而使用的小模型”。因为验证和推理时 dropout 会关闭，完整的 1536 个神经元都会参与计算。

  ## 7. weight_decay=1e-4：很轻的参数约束

  除了 dropout，我还对 embedding 和权重矩阵使用了：

  weight_decay = 1e-4

  它会轻微阻止权重无限增大。

  这和 Bengio 论文中的 penalized likelihood 思想一致：我们不仅要求模型拟合数据，还对过大的参数收取一点代价。

  bias 没有使用 weight decay，因为 bias 主要调整整体偏移，不像大型权重矩阵那样承担大量字符组合和交互关系。

  不过这里不要高估 weight decay 的贡献：

  - dropout 是更主要的正则化
  - 1e-4 weight decay 是轻微辅助
  - 它们必须用 dev loss 判断，而不是因为论文里有就默认有效

  ## 8. 初始化：让模型从“合理的无知”开始

  初始化是 Karpathy 课程里很重要、但经常被低估的部分。

  模型有 27 个可能的下一个字符。如果一开始什么也不知道，最合理的是给每个字符差不多相同的概率：

  1 / 27

  对应的 loss 是：

  -log(1 / 27) = log(27) ≈ 3.29584

  所以我把最后一层初始化为：

  fc2.weight std = 0.001
  fc2.bias = 0

  这样初始 logits 都接近零，初始 softmax 接近均匀分布，实际初始 loss 是：

  3.2984

  非常接近理论上的 log(27)。

  如果最后一层随机值太大，模型一开始就会对随机字符产生很高的置信度。交叉熵会很大，训练前几百步只能先浪费时间消除这些错误的自信。

  对 tanh 第一层则使用：

  std = (5/3) / sqrt(fan_in)

  这里 fan_in=120。

  目的在于让初始隐藏激活：

  - 不要全部挤在 0 附近
  - 也不要大量进入 tanh 的 -1/+1 饱和区域
  - 让不同层的激活和梯度保持健康尺度

  关键 insight 是：

  > 初始化不会改变模型理论上能表示什么，但会显著改变优化器能否顺利找到那个解。

  ## 9. 为什么换成 AdamW、batch 1024？

  课程使用较小 minibatch，是为了教学上直观，也可以工作。但最终配置用了：

  AdamW
  batch size = 1024
  betas = (0.9, 0.99)

  原因是不同参数的梯度尺度很不一样：

  - embedding 中有些字符出现频繁，有些很少
  - 第一层有 18 万多个参数
  - 输出层只有 27 类

  AdamW 会根据每个参数最近的梯度尺度调整更新量，使这些不同类型的参数更容易一起训练。

  batch 从 32 增加到 1024 后，梯度噪声显著降低。每一步看到的字符样本更多，dev loss 的下降也更可预测。

  7500 步总共抽取：

  7500 × 1024 = 7,680,000

  个训练样本。训练集有 182,625 个字符预测样本，相当于大约 42 个 epoch 的抽样量。

  这并不改变网络架构，只是更有效地优化同一个 MLP。

  ## 10. warmup + cosine decay：先学习，再收敛

  学习率不是固定的，而是：

  前 100 步：
  0.00002 → 0.002

  之后：
  0.002 → 0.00006

  前 100 步 warmup 是为了避免模型和 Adam 的统计量还没有稳定时，突然使用最大步长。

  之后 cosine decay 的思路是：

  - 早期用大学习率快速找到不错的区域
  - 后期逐渐减小步长，在这个区域里精细调整
  - 避免在最优点附近反复跳来跳去

  实际训练过程很好地展示了这个现象：

  step  500: dev 2.1180
  step 1000: dev 2.0662
  step 2500: dev 2.0148
  step 4500: dev 1.9864
  step 6000: dev 1.9650
  step 7500: dev 1.9536

  模型在第 500 步就已经低于课程的 2.17，但后面更小的学习率又贡献了相当多的改善。

  这说明“模型已经会了”与“模型已经精确收敛”是两回事。

  ## 11. 为什么保存最佳 dev checkpoint？

  训练 loss 通常继续下降，但 dev loss 不保证单调下降。

  所以每 500 步完整计算一次 dev loss，并保存最佳参数：

  if current_dev < best_dev:
      best_state = model.state_dict()

  最后恢复的不是“最后一步”，而是“dev 最好的一步”。

  这次最佳点恰好出现在 7500，但代码不能事先假设这一点。别的种子或者配置可能在 6500、7000 就已经最好。

  test set 只在最终模型确定后查看一次。否则如果看着 test loss 调 context、dropout 和宽度，test set 就已经变成了另一个 validation set。

  ## 12. 失败实验分别告诉了我什么？

  ### context 6/8 没有更好

  它们降低了 train loss，却没有降低 dev loss。

  结论不是“长 context 永远没用”，而是：

  > 在当前数据量和单隐藏层 MLP 中，context 5 之后，新增参数带来的过拟合超过了额外历史带来的信息收益。

  ### 两层隐藏层没有更好

  增加第二个非线性层理论上更有表达能力，但也更难优化，并且消耗更多参数。

  这个名字数据集的主要规律比较局部，宽的一层已经可以表示大量字符组合。第二层没有提供像卷积的局部性、RNN 的状态或 attention 的动态选择这样的新归纳偏置，只是“更深”。

  所以它没有稳定胜过单个宽隐藏层。

  这不证明两层 MLP 永远不好，只说明在这次计算预算和调参范围里，没有理由把它放进最终版本。

  ### Bengio 的 input→output direct connection 没有更好

  direct connection 相当于：

  embedding 输入 ───────────────→ logits
                ↘ hidden layer ↗

  它允许输出层直接学习线性的字符转移规律，类似一个容易优化的低阶 n-gram 通道。

  理论上它可以让模型更快学习“最近字符对下一个字符的直接影响”。但当前隐藏层已经足够宽，可以学习这些简单规律。direct path 可能与隐藏层重复，甚至更容易拟合训练集中的频繁转移。

  实际结果没有稳定 dev 改善；不加 direct 的泛化略好。因此它没有进入最终代码。

  这正好说明：

  > 论文中的组件是待验证的 hypothesis，不是必须照抄的教条。

  ## 13. 从 2.17 到 1.9536 到底有多大？

  最终结果：

  train loss = 1.6830
  dev loss   = 1.9536
  test loss  = 1.9540

  NLL 下降：

  2.1701 - 1.9536 = 0.2165

  perplexity 从：

  exp(2.1701) ≈ 8.76

  下降到：

  exp(1.9536) ≈ 7.05

  降低约 19.5%。

  另一个理解方式是，模型给真实下一个字符的几何平均概率大致从：

  exp(-2.1701) ≈ 11.4%

  提高到：

  exp(-1.9536) ≈ 14.2%

  也就是平均对正确字符赋予了大约 24% 更高的相对概率。这已经是明显提升，不只是随机波动。

  ## 最核心的 insights

  不是“把 3 改成 5、10 改成 24、200 改成 1536”这些数字本身，而是它们背后的顺序：

  1. 先看 train/dev gap，判断当前是欠拟合还是过拟合。
  2. 先解决信息瓶颈：context 不够，模型再大也无能为力。
  3. 再解决表示瓶颈：embedding 太小会形成低秩限制。
  4. 然后增加非线性容量：用更多隐藏神经元加工这些信息。
  5. 容量足够后，问题会从欠拟合转变成过拟合。
  6. 此时才加入 dropout 和 weight decay，而不是一开始就限制模型。
  7. 最后用正确初始化、优化器和学习率计划把潜在能力兑现出来。
  8. 失败实验看 train/dev 的不同变化，它们是在告诉你瓶颈在哪里。

  所以最终模型真正的故事不是“全部调大”，而是：

  > 先让模型拥有足够的信息和容量，故意把它推到能够明显拟合训练集的位置；然后观察泛化差距，再用正则化把多出来的容量约束成真正能够迁移到 dev/test 的规律。
