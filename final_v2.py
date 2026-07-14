#!/usr/bin/env python3
"""Final unified benchmark: scheduler stats + best executor per model."""
import sys, time, numpy as np, json, os, torch
sys.path.insert(0, '/home/mig20/c3_solution')
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from scheduler.planner import import_onnx_graph, decompose_graph, apply_fusion
from scheduler.executor import GPUExecutor
import onnx, onnxruntime as ort

MODEL_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/models'
DATA_BASE = '/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35'

print(f"{'Model':20s} {'executor':>10s} {'kernels':>12s} {'v2':>8s} {'ORT':>8s} {'change':>8s} {'prec'}")
print("-" * 75)

for name in ['mlp_v1', 'resnet_v1', 'transformer_v1']:
    mp = f'{MODEL_BASE}/{name}.onnx'
    idir = f'{DATA_BASE}/{name}/input'
    gp = f'{DATA_BASE}/{name}/golden/logits.npy'

    with open(f'{idir}/manifest.json') as f: m = json.load(f)
    data = {e['name']: np.load(f'{idir}/{e["file"]}') for e in m['tensors']}
    model = onnx.load(mp)
    onames = [o.name for o in model.graph.output]
    B = 64 if name == 'resnet_v1' else 256
    batch = {k: v[:B] for k, v in data.items()}
    gold = np.load(gp)[:B]

    # Scheduler stats
    graph = import_onnx_graph(mp)
    raw = decompose_graph(graph)
    opt, _ = apply_fusion(graph, raw)

    if name == 'transformer_v1':
        # Transformer: use onnx2torch (handles complex ops correctly)
        from onnx2torch import convert
        tmodel = convert(model).eval().cuda()
        inames = [i.name for i in model.graph.input if i.name not in {i.name for i in model.graph.initializer}]
        feed = []
        for n in inames:
            t = torch.from_numpy(batch[n])
            t = t.long() if batch[n].dtype == np.int64 else t.float()
            feed.append(t.cuda())
        with torch.no_grad():
            _ = tmodel(*feed)  # warmup
            t0 = time.perf_counter()
            out = tmodel(*feed)
            t1 = time.perf_counter()
        tv2 = t1 - t0
        ok = np.allclose(out.cpu().numpy(), gold, rtol=1e-3, atol=1e-3)
        exec_name = "onnx2torch"
    else:
        # MLP/ResNet: use v2 executor (raw plan only)
        exe = GPUExecutor(mp)
        exe.load_inputs(batch)
        # Use fused for MLP, raw for ResNet (fused has tensor name issues)
        plan = opt if name == 'mlp_v1' and len(opt) < len(raw) else raw
        _ = exe.execute_plan(plan, onames)  # warmup
        t0 = time.perf_counter()
        out = exe.execute_plan(plan, onames)[onames[0]]
        t1 = time.perf_counter()
        tv2 = t1 - t0
        ok = np.allclose(out, gold, rtol=1e-3, atol=1e-3)
        exec_name = "v2 executor"

    # ORT baseline
    sess = ort.InferenceSession(mp, providers=[('CUDAExecutionProvider',{'device_id':'0'}),'CPUExecutionProvider'])
    inames_ort = [i.name for i in sess.get_inputs()]
    feed_ort = {n: batch[n] for n in inames_ort if n in batch}
    _ = sess.run(onames, feed_ort)  # warmup
    t2 = time.perf_counter()
    _ = sess.run(onames, feed_ort)
    t3 = time.perf_counter()
    tort = t3 - t2

    change = (1-tv2/tort)*100
    kern_str = f"{len(raw)}→{len(opt)}" if len(opt) < len(raw) else f"{len(raw)}"
    print(f"{name:20s} {exec_name:>10s} {kern_str:>12s} {tv2*1000:7.1f}ms {tort*1000:7.1f}ms {change:+7.0f}% {'OK' if ok else 'FAIL'}")
