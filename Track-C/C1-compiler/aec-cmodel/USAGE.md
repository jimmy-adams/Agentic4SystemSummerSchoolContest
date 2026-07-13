# aec-precise 使用教程

本文档里的 `aec-precise` 是已经编译好的 CModel 可执行文件。它用于运行 AEC ISA binary，观察程序执行状态，并把 global memory 的指定区域 dump 到文件。

## 输入是什么

`aec-precise` 的核心输入是 AEC binary 文件，假设命名为 `program.bin`。

AEC 指令是固定宽度指令：

- 每条指令 128 bit。
- 每条指令 16 字节。
- `program.bin` 的文件大小必须是 16 的整数倍。


## 和你的编译器一起使用

如果你的编译器能直接输出 AEC binary，就直接把输出文件作为 `--program` 传给 `aec-precise`。

```bash
./aec-precise --program path/to/program.bin ...
```

需要保证 binary 编码和 AEC ISA 约定一致：

- 每条指令 16 字节。
- 每条 128-bit 指令按 4 个 little-endian `uint32` 写入。
- 写入顺序是 `w0,w1,w2,w3`。
- CModel 解码时把 `w0` 视为 `imm`，`w1` 视为 `src2`，`w2` 拆成 `src1/dest`，`w3` 拆成 `ctrl/opcode`。


## 基本命令格式

在本目录执行：

```bash
./aec-precise \
  --program path/to/program.bin \
  --instructions N \
  --grid x,y,z \
  --block x,y,z \
  --gmem-size BYTES \
  --cmem-size BYTES \
  --pmem-size BYTES \
  --smem-size BYTES \
  --lmem-size BYTES \
  --max-steps N \
  --load target:addr:file \
  --dump addr:size:file
```

最小可运行命令通常只需要：

```bash
./aec-precise \
  --program path/to/program.bin \
  --grid 1,1,1 \
  --block 32,1,1
```

如果不写 `--instructions`，CModel 会根据 `program.bin` 文件大小自动计算指令条数。

## 参数说明

| 参数 | 是否常用 | 说明 |
| --- | --- | --- |
| `--program FILE` | 必填 | AEC binary 文件路径，也就是要运行的 `program.bin`。 |
| `--instructions N` | 可选 | 要运行的指令条数。不填时默认等于 `program.bin` 大小除以 16。 |
| `--grid x,y,z` | 常用 | kernel grid 维度，三个正整数，例如 `1,1,1`。默认是 `1,1,1`。 |
| `--block x,y,z` | 常用 | 每个 block 的 thread 维度，三个正整数，例如 `32,1,1`。默认是 `32,1,1`。 |
| `--gmem-size BYTES` | 常用 | global memory 大小，单位字节。默认 65536。 |
| `--cmem-size BYTES` | 可选 | constant memory 大小，单位字节。默认 65536。 |
| `--pmem-size BYTES` | 可选 | parameter memory 大小，单位字节。默认 65536。 |
| `--smem-size BYTES` | 可选 | 每个 block 的 shared memory 大小，单位字节。默认 65536。 |
| `--lmem-size BYTES` | 可选 | 每个 thread 的 local memory 大小，单位字节。默认 0。 |
| `--max-steps N` | 建议设置 | 最大执行步数，防止程序死循环。默认 100000。 |
| `--load target:addr:file` | 可选 | 运行前把一个二进制文件加载到指定 memory。 |
| `--dump addr:size:file` | 可选 | 运行后把 global memory 的一段内容写到文件。 |
| `--help` | 可选 | 打印简短帮助。 |

## memory size 怎么设置

这些 size 参数只是告诉 CModel 每类 memory 要分配多少字节。程序访问超出范围时，CModel 会报执行错误。

| memory | 参数 | 读写属性 | 用途 |
| --- | --- | --- | --- |
| global memory | `--gmem-size` | 可读写 | 数据数组、输出结果、普通全局读写。 |
| constant memory | `--cmem-size` | 只读 | 常量数据。 |
| parameter memory | `--pmem-size` | 只读 | kernel 参数区。 |
| shared memory | `--smem-size` | 可读写 | 每个 block 独立的 shared memory。 |
| local memory | `--lmem-size` | 可读写 | 每个 thread 独立的 local memory。 |

设置原则：

- size 必须覆盖程序会访问的最大地址。
- 如果要用 `--load gmem:256:/tmp/input.bin`，那么 `--gmem-size` 至少要大于等于 `256 + input.bin` 的字节数。
- 如果要 dump `--dump 512:128:/tmp/output.bin`，那么 `--gmem-size` 至少要大于等于 `512 + 128`。
- 如果程序不访问某类 memory，可以使用默认值；如果程序访问 local memory，建议显式设置 `--lmem-size`。

简单调试时可以先给宽松一些：

```bash
--gmem-size 1048576 \
--cmem-size 65536 \
--pmem-size 65536 \
--smem-size 65536 \
--lmem-size 4096
```

## 加载输入数据

`--load` 用于在程序运行前初始化 memory。

格式：

```text
--load target:addr:file
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `target` | 目标 memory，支持 `gmem`、`cmem`、`pmem`。 |
| `addr` | 起始地址，十进制整数。 |
| `file` | 要加载的二进制文件路径。 |

示例：把输入数组加载到 `gmem[256, ...)`：

```bash
--load gmem:256:/tmp/input.bin
```

示例：把 kernel 参数加载到 `pmem[0, ...)`：

```bash
--load pmem:0:/tmp/params.bin
```

可以写多个 `--load`：

```bash
./aec-precise \
  --program /tmp/program.bin \
  --grid 1,1,1 \
  --block 32,1,1 \
  --gmem-size 1048576 \
  --pmem-size 4096 \
  --load gmem:256:/tmp/input.bin \
  --load pmem:0:/tmp/params.bin
```

注意：`addr` 当前按十进制解析。想加载到地址 `0x100` 时，请写成 `256`。

## dump 输出数据

`--dump` 用于在程序运行结束后，把 global memory 的一段内容写到文件。

格式：

```text
--dump addr:size:file
```

字段含义：

| 字段 | 说明 |
| --- | --- |
| `addr` | 要 dump 的 global memory 起始地址，十进制整数。 |
| `size` | 要 dump 的字节数，十进制整数。 |
| `file` | 输出文件路径。 |

示例：把 `gmem[256, 256 + 128)` 写到 `/tmp/output.bin`：

```bash
--dump 256:128:/tmp/output.bin
```

完整示例：

```bash
./aec-precise \
  --program /tmp/program.bin \
  --grid 1,1,1 \
  --block 32,1,1 \
  --gmem-size 4096 \
  --max-steps 1000 \
  --dump 256:128:/tmp/output.bin
```

查看 dump 文件：

```bash
xxd -g4 /tmp/output.bin
```

当前 `aec-precise` 的 CLI 只支持 dump global memory。如果需要观察 `cmem`、`pmem`、`smem` 或 `lmem`，需要修改 CModel 或在程序里把相关数据写回 global memory 后再 dump。

## stdout 输出

程序结束时，`aec-precise` 会在 stdout 打印一行 JSON：

```json
{"status":"done","steps":10,"pc":0,"cta":0,"warp":0,"lane":0,"opcode":0,"error":""}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `status` | 执行状态。 |
| `steps` | 已执行的调度步数。 |
| `pc` | 出错时的 PC；正常结束时通常为 0。 |
| `cta` | 出错时的 CTA index。 |
| `warp` | 出错时的 warp index。 |
| `lane` | 出错时的 lane index。 |
| `opcode` | 出错时的 opcode。 |
| `error` | 错误信息；正常结束时为空字符串。 |

常见 `status`：

| status | 含义 |
| --- | --- |
| `done` | 正常结束。 |
| `invalid` | 指令编码非法，或遇到未实现 opcode。 |
| `fail` | 执行失败，例如 memory 越界。 |
| `timeout` | 超过 `--max-steps` 仍未结束。 |

返回码规则：

- `status` 为 `done` 时，进程返回码是 0。
- 其他运行状态通常返回 1。
- 参数错误、文件错误、程序 binary 非法时通常返回 2。

## 示例：运行一个无输入程序并 dump gmem

假设你的编译器输出了 `/tmp/program.bin`，程序会把结果写到 `gmem[256, 384)`，可以这样运行：

```bash
./aec-precise \
  --program /tmp/program.bin \
  --grid 1,1,1 \
  --block 32,1,1 \
  --gmem-size 4096 \
  --cmem-size 65536 \
  --pmem-size 65536 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 1000 \
  --dump 256:128:/tmp/output.bin
```

运行后查看状态：

```bash
echo $?
```

查看输出文件：

```bash
xxd -g4 /tmp/output.bin
```

## 示例：运行一个带输入数组和参数的程序

假设：

- `/tmp/program.bin` 是你的 AEC binary。
- `/tmp/input.bin` 是输入数组，放到 `gmem[256, ...)`。
- `/tmp/params.bin` 是参数区，放到 `pmem[0, ...)`。
- 程序把结果写到 `gmem[512, 640)`。

命令：

```bash
./aec-precise \
  --program /tmp/program.bin \
  --grid 1,1,1 \
  --block 32,1,1 \
  --gmem-size 1048576 \
  --cmem-size 65536 \
  --pmem-size 4096 \
  --smem-size 65536 \
  --lmem-size 4096 \
  --max-steps 100000 \
  --load gmem:256:/tmp/input.bin \
  --load pmem:0:/tmp/params.bin \
  --dump 512:128:/tmp/output.bin
```

## 常见错误

### `bad program`

通常表示 `--program` 文件不存在、无法读取，或者文件大小不是 16 的整数倍。

### `instruction count exceeds binary`

`--instructions` 写得比 `program.bin` 实际包含的指令数还多。可以去掉 `--instructions`，让 CModel 自动按文件大小计算。

### `bad load`

通常是 `--load` 的目标 memory 名字不对、输入文件不存在，或者加载范围超过对应 memory size。

### `bad dump`

通常是 dump 输出文件无法创建，或者 `addr + size` 超过 `--gmem-size`。

### `load out of bounds` / `store out of bounds or read-only`

程序执行时访问了越界地址，或者试图写只读 memory。检查程序地址计算、`--load` 地址、以及各类 memory size。
