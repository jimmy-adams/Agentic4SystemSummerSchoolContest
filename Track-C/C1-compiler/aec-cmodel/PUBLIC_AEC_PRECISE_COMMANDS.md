# C1 Public Cases: fixed address rule and CModel commands

本文档给参赛者说明公开 5 个 C1 样例在 `aec-precise` 中的固定 buffer 地址分配规则，以及每个题目的 `aec-cc` 编译命令和 `aec-precise` 运行命令。

这里不提供正确性比对逻辑。`aec-precise` 会在 stdout 打印 JSON，其中 `steps` 字段可作为当前 CModel 暴露的 warp-level 动态执行步数观察值；输出 buffer 通过 `--dump` 写到文件。

## 固定地址规则

GMEM buffer 地址固定规则：

1. 第一个 buffer 从 byte address `256` 开始。
2. 按每个题目 `manifest.json` 中 `buffers` 的出现顺序分配。
3. 每个 buffer 的 base address 按 `256` bytes 对齐。
4. buffer size 为元素数量乘以 dtype 字节数；公开样例 dtype 均为 `f32`，每个元素 4 bytes。
5. 输出 buffer 也按同样规则分配，并在运行前加载 zero-init 文件。
6. `gmem_ptr` kernel 参数写入对应 buffer 的 GMEM byte address。

PMEM 参数区规则：

1. 按 `manifest.json` 中 `params` 的出现顺序写入。
2. `u64/b64` 参数占 8 bytes，按 8 bytes 对齐。
3. `u32/s32/b32/f32` 参数占 4 bytes，按 4 bytes 对齐。
4. 所有值使用 little-endian。
5. 参数块总大小按 8 bytes 对齐。

本文档命令假设输入文件按如下命名准备：

```text
/tmp/c1_inputs/<case>/pmem.bin
/tmp/c1_inputs/<case>/input_<buffer>.bin
```

输出 dump 在下面命令中写到 `/tmp/*_dump_*.bin`。

## 如何准备输入文件

每个 `input_<buffer>.bin` 是对应 GMEM buffer 的原始二进制镜像。公开样例所有 buffer 都是 `f32`，因此文件内容为连续的 little-endian IEEE-754 binary32。

buffer 初始化规则：

- `init: "zero"`：写入全 0 的 `f32` 数组。
- `init: "rand_uniform"`：使用 `manifest.json` 中的 `seed` 初始化伪随机数生成器，生成区间 `[-1.0, 1.0]` 的均匀随机数，再按 `f32` 写入。

如果使用 Python 生成公开样例输入，可采用下面这段脚本。它同时生成 5 个样例的 `pmem.bin` 和 `input_<buffer>.bin`，路径与本文档后续命令一致：

```bash
python3 - <<'PY'
import random
import struct
from pathlib import Path

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def pack_f32(values):
    return struct.pack("<" + "f" * len(values), *values)

def rand_f32(numel, seed):
    rng = random.Random(seed)
    return [f32(rng.uniform(-1.0, 1.0)) for _ in range(numel)]

def zero_f32(numel):
    return [0.0] * numel

def u32(value):
    return struct.pack("<I", value & 0xffffffff)

def u64(value):
    return struct.pack("<Q", value & 0xffffffffffffffff)

def write_case(case, buffers, pmem_entries):
    out = Path("/tmp/c1_inputs") / case
    out.mkdir(parents=True, exist_ok=True)

    for name, values in buffers.items():
        (out / f"input_{name}.bin").write_bytes(pack_f32(values))

    size = max(offset + len(data) for offset, data in pmem_entries)
    size = (size + 7) & ~7
    pmem = bytearray(size)
    for offset, data in pmem_entries:
        pmem[offset:offset + len(data)] = data
    (out / "pmem.bin").write_bytes(pmem)

write_case(
    "T1_basic_lowering",
    {
        "a": rand_f32(1048576, 1),
        "b": rand_f32(1048576, 2),
        "c": zero_f32(1048576),
    },
    [
        (0, u64(256)),
        (8, u64(4194560)),
        (16, u64(8388864)),
        (24, u32(1048576)),
    ],
)

write_case(
    "T2_scalar_optimization",
    {
        "x": rand_f32(524288, 3),
        "y": rand_f32(524288, 4),
        "out": zero_f32(524288),
    },
    [
        (0, u64(256)),
        (8, u64(2097408)),
        (16, u64(4194560)),
        (24, u32(524288)),
    ],
)

write_case(
    "T3_memory_reuse",
    {
        "x": rand_f32(524288, 5),
        "y": rand_f32(524288, 6),
        "z": rand_f32(524288, 7),
        "out": zero_f32(524288),
    },
    [
        (0, u64(256)),
        (8, u64(2097408)),
        (16, u64(4194560)),
        (24, u64(6291712)),
        (32, u32(524288)),
    ],
)

write_case(
    "T4_register_scheduling",
    {
        "a": rand_f32(524288, 8),
        "b": rand_f32(524288, 9),
        "c": rand_f32(524288, 10),
        "d": rand_f32(524288, 11),
        "out": zero_f32(524288),
    },
    [
        (0, u64(256)),
        (8, u64(2097408)),
        (16, u64(4194560)),
        (24, u64(6291712)),
        (32, u64(8388864)),
        (40, u32(524288)),
    ],
)

write_case(
    "T5_scalar_gemm",
    {
        "A": rand_f32(16384, 12),
        "B": rand_f32(16384, 13),
        "C": zero_f32(16384),
    },
    [
        (0, u64(256)),
        (8, u64(65792)),
        (16, u64(131328)),
        (24, u32(128)),
        (28, u32(128)),
        (32, u32(128)),
    ],
)
PY
```

如果 `aec-precise` 不在仓库默认路径，请把命令中的：

```text
./simple-gpgpu/cmodel/precise/aec-precise
```

替换为实际路径。

## T1 basic lowering

Kernel: `vector_add`

GMEM:

| Buffer | Address | Size |
|---|---:|---:|
| `a` | `256` | `4194304` |
| `b` | `4194560` | `4194304` |
| `c` | `8388864` | `4194304` |

PMEM:

| Param | Offset | Type | Value |
|---|---:|---|---:|
| `param_a` | `0` | `u64` | `256` |
| `param_b` | `8` | `u64` | `4194560` |
| `param_c` | `16` | `u64` | `8388864` |
| `param_n` | `24` | `u32` | `1048576` |

Compile:

```bash
./compiler/aec-cc testcases/T1_basic_lowering/kernel.ptx -O2 -o /tmp/T1_basic_lowering.aecbin --report /tmp/T1_basic_lowering_compile_report.json
```

Run:

```bash
./simple-gpgpu/cmodel/precise/aec-precise \
  --program /tmp/T1_basic_lowering.aecbin \
  --grid 4096,1,1 \
  --block 256,1,1 \
  --gmem-size 12591104 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 50000000 \
  --load pmem:0:/tmp/c1_inputs/T1_basic_lowering/pmem.bin \
  --load gmem:256:/tmp/c1_inputs/T1_basic_lowering/input_a.bin \
  --load gmem:4194560:/tmp/c1_inputs/T1_basic_lowering/input_b.bin \
  --load gmem:8388864:/tmp/c1_inputs/T1_basic_lowering/input_c.bin \
  --dump 8388864:4194304:/tmp/T1_basic_lowering_dump_c.bin
```

## T2 scalar optimization

Kernel: `repeated_expression`

GMEM:

| Buffer | Address | Size |
|---|---:|---:|
| `x` | `256` | `2097152` |
| `y` | `2097408` | `2097152` |
| `out` | `4194560` | `2097152` |

PMEM:

| Param | Offset | Type | Value |
|---|---:|---|---:|
| `param_x` | `0` | `u64` | `256` |
| `param_y` | `8` | `u64` | `2097408` |
| `param_out` | `16` | `u64` | `4194560` |
| `param_n` | `24` | `u32` | `524288` |

Compile:

```bash
./compiler/aec-cc testcases/T2_scalar_optimization/kernel.ptx -O2 -o /tmp/T2_scalar_optimization.aecbin --report /tmp/T2_scalar_optimization_compile_report.json
```

Run:

```bash
./simple-gpgpu/cmodel/precise/aec-precise \
  --program /tmp/T2_scalar_optimization.aecbin \
  --grid 2048,1,1 \
  --block 256,1,1 \
  --gmem-size 6299648 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 50000000 \
  --load pmem:0:/tmp/c1_inputs/T2_scalar_optimization/pmem.bin \
  --load gmem:256:/tmp/c1_inputs/T2_scalar_optimization/input_x.bin \
  --load gmem:2097408:/tmp/c1_inputs/T2_scalar_optimization/input_y.bin \
  --load gmem:4194560:/tmp/c1_inputs/T2_scalar_optimization/input_out.bin \
  --dump 4194560:2097152:/tmp/T2_scalar_optimization_dump_out.bin
```

## T3 memory reuse

Kernel: `repeated_global_load`

GMEM:

| Buffer | Address | Size |
|---|---:|---:|
| `x` | `256` | `2097152` |
| `y` | `2097408` | `2097152` |
| `z` | `4194560` | `2097152` |
| `out` | `6291712` | `2097152` |

PMEM:

| Param | Offset | Type | Value |
|---|---:|---|---:|
| `param_x` | `0` | `u64` | `256` |
| `param_y` | `8` | `u64` | `2097408` |
| `param_z` | `16` | `u64` | `4194560` |
| `param_out` | `24` | `u64` | `6291712` |
| `param_n` | `32` | `u32` | `524288` |

Compile:

```bash
./compiler/aec-cc testcases/T3_memory_reuse/kernel.ptx -O2 -o /tmp/T3_memory_reuse.aecbin --report /tmp/T3_memory_reuse_compile_report.json
```

Run:

```bash
./simple-gpgpu/cmodel/precise/aec-precise \
  --program /tmp/T3_memory_reuse.aecbin \
  --grid 2048,1,1 \
  --block 256,1,1 \
  --gmem-size 8396800 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 50000000 \
  --load pmem:0:/tmp/c1_inputs/T3_memory_reuse/pmem.bin \
  --load gmem:256:/tmp/c1_inputs/T3_memory_reuse/input_x.bin \
  --load gmem:2097408:/tmp/c1_inputs/T3_memory_reuse/input_y.bin \
  --load gmem:4194560:/tmp/c1_inputs/T3_memory_reuse/input_z.bin \
  --load gmem:6291712:/tmp/c1_inputs/T3_memory_reuse/input_out.bin \
  --dump 6291712:2097152:/tmp/T3_memory_reuse_dump_out.bin
```

## T4 register scheduling

Kernel: `mixed_load_compute`

GMEM:

| Buffer | Address | Size |
|---|---:|---:|
| `a` | `256` | `2097152` |
| `b` | `2097408` | `2097152` |
| `c` | `4194560` | `2097152` |
| `d` | `6291712` | `2097152` |
| `out` | `8388864` | `2097152` |

PMEM:

| Param | Offset | Type | Value |
|---|---:|---|---:|
| `param_a` | `0` | `u64` | `256` |
| `param_b` | `8` | `u64` | `2097408` |
| `param_c` | `16` | `u64` | `4194560` |
| `param_d` | `24` | `u64` | `6291712` |
| `param_out` | `32` | `u64` | `8388864` |
| `param_n` | `40` | `u32` | `524288` |

Compile:

```bash
./compiler/aec-cc testcases/T4_register_scheduling/kernel.ptx -O2 -o /tmp/T4_register_scheduling.aecbin --report /tmp/T4_register_scheduling_compile_report.json
```

Run:

```bash
./simple-gpgpu/cmodel/precise/aec-precise \
  --program /tmp/T4_register_scheduling.aecbin \
  --grid 2048,1,1 \
  --block 256,1,1 \
  --gmem-size 10493952 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 50000000 \
  --load pmem:0:/tmp/c1_inputs/T4_register_scheduling/pmem.bin \
  --load gmem:256:/tmp/c1_inputs/T4_register_scheduling/input_a.bin \
  --load gmem:2097408:/tmp/c1_inputs/T4_register_scheduling/input_b.bin \
  --load gmem:4194560:/tmp/c1_inputs/T4_register_scheduling/input_c.bin \
  --load gmem:6291712:/tmp/c1_inputs/T4_register_scheduling/input_d.bin \
  --load gmem:8388864:/tmp/c1_inputs/T4_register_scheduling/input_out.bin \
  --dump 8388864:2097152:/tmp/T4_register_scheduling_dump_out.bin
```

## T5 scalar GEMM

Kernel: `scalar_gemm`

GMEM:

| Buffer | Address | Size |
|---|---:|---:|
| `A` | `256` | `65536` |
| `B` | `65792` | `65536` |
| `C` | `131328` | `65536` |

PMEM:

| Param | Offset | Type | Value |
|---|---:|---|---:|
| `param_A` | `0` | `u64` | `256` |
| `param_B` | `8` | `u64` | `65792` |
| `param_C` | `16` | `u64` | `131328` |
| `param_M` | `24` | `u32` | `128` |
| `param_N` | `28` | `u32` | `128` |
| `param_K` | `32` | `u32` | `128` |

Compile:

```bash
./compiler/aec-cc testcases/T5_scalar_gemm/kernel.ptx -O2 -o /tmp/T5_scalar_gemm.aecbin --report /tmp/T5_scalar_gemm_compile_report.json
```

Run:

```bash
./simple-gpgpu/cmodel/precise/aec-precise \
  --program /tmp/T5_scalar_gemm.aecbin \
  --grid 8,8,1 \
  --block 16,16,1 \
  --gmem-size 204800 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 50000000 \
  --load pmem:0:/tmp/c1_inputs/T5_scalar_gemm/pmem.bin \
  --load gmem:256:/tmp/c1_inputs/T5_scalar_gemm/input_A.bin \
  --load gmem:65792:/tmp/c1_inputs/T5_scalar_gemm/input_B.bin \
  --load gmem:131328:/tmp/c1_inputs/T5_scalar_gemm/input_C.bin \
  --dump 131328:65536:/tmp/T5_scalar_gemm_dump_C.bin
```
