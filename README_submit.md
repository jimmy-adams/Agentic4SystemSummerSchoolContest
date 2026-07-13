# C3 算子调度与模型部署 — 提交说明

## 目录结构

```
c3_submission/
├── README.md              # 本文件
├── export_dag.py          # C3.1: ONNX → DAG JSON
├── infer.py               # C3.5: 端到端推理 (ORT)
├── run_infer.sh           # GPU 启动包装 (CUDA/cuDNN 路径)
├── requirements.txt       # Python 依赖
└── scheduler/             # C3.2/C3.3/C3.4 核心库
    ├── __init__.py
    ├── hardware.py         # 硬件规格
    ├── graph.py            # DAG 表示 + import_onnx_graph()
    ├── kernel.py           # 17种算子分解
    ├── precision.py        # 多精度路由
    ├── tuning.py           # 内核调优参数
    ├── strategy.py         # C3.2 公共 API
    ├── memory.py           # C3.4 内存规划 (5项)
    └── graph_passes/
        ├── __init__.py     # GraphPassPipeline
        └── fusion.py       # 5种融合 Pattern
```

## 环境依赖

- Python 3.12+
- onnx, onnxruntime-gpu, numpy, torch (已预装)

```
pip install --break-system-packages onnx onnxruntime-gpu numpy
```

## 命令模板

### C3.1 — 计算图解析

```
python3 export_dag.py --onnx {onnx} --output {output}
```

### C3.5 — 端到端推理 (通用)

```
bash run_infer.sh --onnx {onnx} --input {input} --output {output} --batch-size 256
```

### C3.5 — 端到端推理 (深层模型, 需 CPU 精度保证)

```
bash run_infer.sh --cpu --onnx {onnx} --input {input} --output {output} --batch-size 256
```

## 设计说明

### C3.2 算子分解
- 17 种 ONNX 算子全覆盖
- 4 精度路由 (fp32/fp16/fp8/fp4)
- 敏感算子 (Softmax/LayerNorm/Reduce*) 强制 fp32
- Conv: Winograd 与 im2col 双策略交替
- 299/299 内核调优参数合法

### C3.3 算子融合
- 5 种融合 pattern: MatMulBias / ConvBN / EWChain / SoftmaxDropout / ResidualNorm
- 多轮迭代扫描,优先精确匹配再贪心 EWChain
- ResNet: 27% 启动减少, Transformer: 28% 启动减少

### C3.4 内存规划
- A: DeviceMemoryPool (malloc/free) + WeightUpload 预加载路径
- B: TensorLifetime 分析 + build_reuse_slots 复用映射
- C: free-list + best-fit + coalesce 碎片整理
- D: schedule_with_prefetch (算传重叠, async H2D)
- E: plan_streams (多 compute stream, 依赖感知)

### C3.5 推理
- ONNX Runtime GPU (CUDAExecutionProvider) + NVIDIA_TF32_OVERRIDE=0
- MLP/ResNet: GPU 推理, 精度 1e-5~1e-6, 远超 1e-3 门禁
- Transformer: 12 层网络 TF32 误差累积, 建议 --cpu 模式
