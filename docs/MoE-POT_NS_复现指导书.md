# MoE-POT-Tiny NS 零样本与 FT200 BitaHub 复现指导书

## 目标

在 BitaHub 单卡 RTX 4090 上，使用作者公开的 MoE-POT-Tiny 预训练权重，复现论文 Table 1 中 NS 两列的 zero-shot 和 FT200 结果。

| 数据集 | Zero-shot L2RE | FT200 L2RE |
|---|---:|---:|
| NS 1e-5 | 0.0682 | 0.0456 |
| NS 1e-3 | 0.00768 | 0.00493 |

先记录和展示原始结果。验收预警为 zero-shot 相对偏差不超过 1%、FT200 相对偏差不超过 5%，但预警不替代原始数值。

本实验固定作者仓库提交 `c36ec22aa492b9460d2e9f24cf4f0354d0f07540`，使用独立分支 `repro/tiny-ns-ft200`。不改模型结构或训练公式。

## 参数

| 项目 | 值 |
|---|---|
| 数据集 | NS 1e-5、NS 1e-3 |
| 数据格式 | HDF5，键 `data`，规范 shape `(N,64,64,T)` |
| 数据规模 | 1e-3 原始约 4.1GB，裁剪后两套合计小于 2GB |
| 模型 | MoE-POT-Tiny，总计约 30M，激活约 17M |
| checkpoint | `model_moepot_tiny.pth` |
| 共享环境 | `/data/nestor_env/envs/nestor` |
| 现有数据 | `nestor_1e5`，挂载到 `/data/nestor_1e5` |
| 工作数据 | `nestor_moepot_ns`，挂载到 `/data/nestor_moepot_ns` |
| 模型资源 | `nestor_moepot_tiny`，挂载到 `/model/nestor_moepot_tiny` |
| 代码 | `/code/MoEPOT` |
| GPU | 1 卡 RTX 4090，24GB |
| batch size | 20 |
| learning rate | 1e-3，OneCycle |
| epochs | 200 |
| warmup | 40 epochs |
| 输入/输出时间步 | `T_in=10`，`T_ar=1`，`T_bundle=1` |
| patch size | 8 |
| Tiny 配置 | width 512，4 layers，4 blocks，Top-K 4 |

NS 1e-3 原始轨迹总大小约 4.1GB，裁剪后训练/测试数据约 1GB；全部执行都在云端完成，不需要把数据或权重下载到本地。

## 执行方式

使用开发机 `nohup`，不启用分布式训练。原因是本次只保留两套 NS 数据，裁剪后总量小于 2GB，Tiny 可放入单张 4090；仓库的 zero-shot 入口本身也是单进程。FT200 必须先跑 NS 1e-5，完成后再跑 NS 1e-3，两次都从同一个预训练 checkpoint 重新开始。

## 操作步骤

### 1. 创建并挂载 BitaHub 资源

按新版界面操作：

```text
数据&存储 -> 数据集
数据&存储 -> 存储管理
模型&推理 -> 模型库
开发&训练 -> 开发机
管理&配置 -> 代码管理
```

创建以下资源并挂载到开发机：

```text
/data/nestor_1e5            已有数据，只读输入
/data/nestor_moepot_ns      新建工作数据，读写
/data/nestor_env            共享环境
/model/nestor_moepot_tiny   新建模型资源
/code                       代码目录
```

工作数据卷建议预先创建以下目录：

```bash
mkdir -p /data/nestor_moepot_ns/{raw,inventory,large,runs,cache/huggingface}
```

验收：开发机 WebTerminal 中五个路径都存在，`touch /data/nestor_moepot_ns/.write_test` 成功。

### 2. 放置代码

优先使用工作区生成的完整 `MoEPOT` 仓库，目录名必须保持为 `MoEPOT`，上传后位于 `/code/MoEPOT`。如果从作者仓库重新创建分支，则执行：

```bash
cd /code
git clone https://github.com/haiyangxin/MoEPOT.git
cd /code/MoEPOT
git fetch --depth 1 origin c36ec22aa492b9460d2e9f24cf4f0354d0f07540
git checkout -b repro/tiny-ns-ft200 c36ec22aa492b9460d2e9f24cf4f0354d0f07540
git apply --check /code/repro_changes.patch
git apply /code/repro_changes.patch
```

建议通过 BitaHub WebTerminal 上传约几 MB 的 `repro_changes.patch`，不要上传数据或模型。

验收：

```bash
cd /code/MoEPOT
git branch --show-current
# repro/tiny-ns-ft200
git merge-base --is-ancestor c36ec22aa492b9460d2e9f24cf4f0354d0f07540 HEAD
echo "base commit check: PASS"
test -f scripts/run_zero_shot.sh
```

### 3. 准备共享环境

```bash
cd /code/MoEPOT
ENV_DIR=/data/nestor_env/envs/nestor bash scripts/setup_env_moepot.sh
```

脚本会复用已有环境；若目标路径没有 Python，则在云端创建虚拟环境，并安装 PyTorch 2.0/CUDA 11.7 和本实验依赖。作者原始 `requirements.txt` 中的 `yaml==0.2.5` 不直接使用，复现依赖改用 `requirements-moepot.txt`。

后续每个新终端先执行：

```bash
export ENV_DIR=/data/nestor_env/envs/nestor
export PATH="$ENV_DIR/bin:$PATH"
cd /code/MoEPOT
python scripts/check_env.py
```

验收：输出 `cuda_available: True`、GPU 名称 RTX 4090，最后一行是 `environment check: PASS`。

如现有共享环境被其他项目占用，优先只补缺依赖；如果版本冲突，停止安装并把 `ENV_DIR` 改为 `/data/nestor_env/envs/moepot` 后重新运行，避免破坏 NESTOR 环境。

### 4. 盘点现有 NS 1e-5

```bash
cd /code/MoEPOT
python scripts/preflight_assets.py \
  --root /data/nestor_1e5 \
  --output /data/nestor_moepot_ns/inventory/nestor_1e5.json
cat /data/nestor_moepot_ns/inventory/nestor_1e5.json
```

这一步只读取文件结构和 shape，不加载完整轨迹。目标是找到：

```text
训练: data/(1000, 64, 64, 20) 或可转换成该形状的 u 数组
测试: data/(200, 64, 64, 20) 或包含总计 1200 条轨迹的单个数组
```

如果已经是仓库格式，直接从 JSON 中记录两个 HDF5 的绝对路径。若不是，则把实际源文件路径设为 `NS1E5_SOURCE` 并统一转换：

```bash
export NS1E5_SOURCE="/data/nestor_1e5/实际文件.mat"
python scripts/prepare_fno_ns.py \
  --source "$NS1E5_SOURCE" \
  --variable u \
  --viscosity 1e-5 \
  --output-dir /data/nestor_moepot_ns/large \
  --n-train 1000 \
  --n-test 200 \
  --chunk-size 25
```

如果清单显示变量名是 `data` 而不是 `u`，把 `--variable` 改为 `data`。转换脚本支持 `(64,64,20,N)`、`(N,20,64,64)` 和 `(N,64,64,20)`，并强制前 1000 条训练、最后 200 条测试。

验收：生成 `prepare_manifest.json`，其中 train/test 的 shape 分别为 `[1000,64,64,20]` 和 `[200,64,64,20]`。

### 5. 在云端获取 NS 1e-3

打开论文 README 中的 FNO 数据链接：

```text
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLnZaXrKUZhp7pjt-
```

只选择 `NavierStokes_V1e-3_N5000_T50.mat`，从网页获得该单文件的 Google File ID。不要把整个文件夹递归下载到本地。

在 BitaHub 执行：

```bash
export FNO_1E3_FILE_ID="从单个文件链接中取得的 ID"
gdown --id "$FNO_1E3_FILE_ID" \
  --output /data/nestor_moepot_ns/raw/NavierStokes_V1e-3_N5000_T50.mat
ls -lh /data/nestor_moepot_ns/raw/NavierStokes_V1e-3_N5000_T50.mat
```

下载后先盘点：

```bash
python scripts/preflight_assets.py \
  --root /data/nestor_moepot_ns/raw \
  --output /data/nestor_moepot_ns/inventory/ns1e3_raw.json
```

验收：文件可被 `h5py` 读取，变量 `u` 的总轨迹数为 5000、时间步为 50；常见源形状是 `(64,64,50,5000)`。

### 6. 转换并注册 NS 1e-3

```bash
python scripts/prepare_fno_ns.py \
  --source /data/nestor_moepot_ns/raw/NavierStokes_V1e-3_N5000_T50.mat \
  --variable u \
  --viscosity 1e-3 \
  --output-dir /data/nestor_moepot_ns/large \
  --n-train 1000 \
  --n-test 200 \
  --chunk-size 10
```

采用前 1000 条训练、最后 200 条测试。虽然原训练文件有 4800 条轨迹，论文 Table 6 的 FT200 只使用 1000 条。

然后把两套数据注册到仓库预期路径。下面的 train/test 变量应替换为步骤 4 和本步骤生成的实际文件路径：

```bash
export NS1E5_TRAIN="/data/nestor_moepot_ns/large/ns2d_1e-5_train.hdf5"
export NS1E5_TEST="/data/nestor_moepot_ns/large/ns2d_1e-5_test.hdf5"

python scripts/link_data.py \
  --repo-root /code/MoEPOT \
  --ns1e5-train "$NS1E5_TRAIN" \
  --ns1e5-test "$NS1E5_TEST" \
  --ns1e3-train /data/nestor_moepot_ns/large/ns2d_1e-3_train.hdf5 \
  --ns1e3-test /data/nestor_moepot_ns/large/ns2d_1e-3_test.hdf5
```

验收命令：

```bash
ls -l data/large/ns2d_1e-{5,3}_{train,test}.hdf5
cat data/large/repro_data_links.json
```

四个 shape 必须分别是：

```text
(1000,64,64,20), (200,64,64,20)
(1000,64,64,50), (200,64,64,50)
```

### 7. 下载并校验 Tiny checkpoint

```bash
cd /code/MoEPOT
export HF_HOME=/data/nestor_moepot_ns/cache/huggingface
python scripts/download_tiny_checkpoint.py \
  --repo-id xhy2878/MoEPOT \
  --output /model/nestor_moepot_tiny/model_moepot_tiny.pth
```

脚本只在 Hugging Face 中下载文件名匹配 Tiny 的 checkpoint，不下载全部五个权重；随后计算 SHA256，并按 30M Tiny 结构执行完整 state dict 加载。

如果 `/model/nestor_moepot_tiny` 不可写，改为：

```bash
python scripts/download_tiny_checkpoint.py \
  --output /data/nestor_moepot_ns/models/model_moepot_tiny.pth
export MOEPOT_CHECKPOINT=/data/nestor_moepot_ns/models/model_moepot_tiny.pth
```

再从 BitaHub 页面把该文件登记到 `nestor_moepot_tiny` 模型资源。验收：manifest 中 `model_check` 为 `all Tiny model keys loaded successfully`。

### 8. 运行 zero-shot

```bash
cd /code/MoEPOT
bash scripts/launch_nohup.sh zero
RUN_DIR=/data/nestor_moepot_ns/runs/zero_shot
bash scripts/monitor_run.sh "$RUN_DIR"
```

完成后查看：

```bash
cat "$RUN_DIR/zero_shot.log"
```

最终应出现：

```text
ns2d_fno_1e-5: 0.xxxxx
ns2d_fno_1e-3: 0.xxxxx
```

验收：两个指标都生成，且分别对照 `0.0682`、`0.00768`。如果结果相差数量级、出现 NaN 或 checkpoint 键错误，停止后续 FT200，先检查权重版本和数据布局。

### 9. 各跑 5 个冒烟 epoch

先跑 NS 1e-5：

```bash
cd /code/MoEPOT
bash scripts/run_finetune.sh ns1e-5 smoke
```

完成后：

```bash
RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-5/smoke
test -f "$RUN_DIR/model.pth"
grep '^epoch' "$RUN_DIR/logs.txt" | tail -n 5
```

再跑 NS 1e-3：

```bash
bash scripts/run_finetune.sh ns1e-3 smoke
RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-3/smoke
test -f "$RUN_DIR/model.pth"
grep '^epoch' "$RUN_DIR/logs.txt" | tail -n 5
```

冒烟脚本仍按 200 epoch 构造 OneCycle，因此学习率轨迹与正式训练一致，只是实际执行 5 个 epoch。验收：每个 run 有 5 条 epoch 记录，测试 L2RE 是有限数，`model.pth` 可加载。冒烟结果不能作为论文结果。

### 10. 正式训练 FT200

#### 10.1 启动 NS 1e-5

在 BitaHub 开发机的 WebTerminal 中执行：

```bash
cd /code/MoEPOT
bash scripts/launch_nohup.sh finetune ns1e-5 full
```

启动脚本会立即返回，并输出 PID。默认运行目录为：

```bash
export RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-5/full
```

相关文件：

```text
"$RUN_DIR/pid"         后台进程 PID
"$RUN_DIR/nohup.out"   启动信息及进程级错误
"$RUN_DIR/logs.txt"    epoch、learning rate、训练/测试 L2RE
```

如果关闭浏览器或 WebTerminal，`nohup` 训练仍会继续。

#### 10.2 实时查看 NS 1e-5 状态

只查看一次当前状态：

```bash
bash scripts/monitor_run.sh "$RUN_DIR"
```

持续刷新进程、GPU 显存、GPU 利用率和最近 20 行日志：

```bash
bash scripts/monitor_run.sh "$RUN_DIR" --follow
```

该命令每 60 秒刷新一次。按 `Ctrl-C` 只退出监控界面，不会终止后台训练；需要再次查看时重新执行上面的命令即可。

另开一个 WebTerminal，可只跟踪训练日志：

```bash
RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-5/full
tail -F "$RUN_DIR/logs.txt"
```

查看 PID、进程是否存活和 GPU：

```bash
RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-5/full
PID=$(cat "$RUN_DIR/pid")
echo "pid=$PID"
ps -p "$PID" -o pid,ppid,etime,%cpu,%mem,stat,cmd
kill -0 "$PID" && echo "process is RUNNING" || echo "process is STOPPED"
nvidia-smi
```

期望状态：

```text
process: RUNNING
GPU utilization: 持续出现非零百分比
logs.txt: 每完成一个 epoch 增加一行，epoch 从 0 递增
```

实时监控命令 `monitor_run.sh ... --follow` 在检测到 PID 不存在后显示 `run has stopped` 并自动退出。

#### 10.3 NS 1e-5 完成验收

进程停止后执行：

```bash
test -f "$RUN_DIR/model.pth"
grep '^epoch 199' "$RUN_DIR/logs.txt"
tail -n 20 "$RUN_DIR/logs.txt"
```

只有确认 epoch 199 已出现、`model.pth` 存在后，才启动第二个数据集：

```bash
bash scripts/launch_nohup.sh finetune ns1e-3 full
RUN_DIR=/data/nestor_moepot_ns/runs/finetune/ns1e-3/full
bash scripts/monitor_run.sh "$RUN_DIR" --follow
```

NS 1e-3 完成后：

```bash
test -f "$RUN_DIR/model.pth"
grep '^epoch 199' "$RUN_DIR/logs.txt"
tail -n 20 "$RUN_DIR/logs.txt"
```

两个数据集共用一张 4090，必须串行执行。不要在前一个任务仍显示 `RUNNING` 时启动后一个任务。两次训练都必须从预训练 checkpoint 开始，不能把 NS 1e-5 的 FT200 权重作为 NS 1e-3 的起点。

### 11. 汇总结果

```bash
cd /code/MoEPOT
python scripts/collect_metrics.py \
  --runs-root /data/nestor_moepot_ns/runs
cat /data/nestor_moepot_ns/runs/summary/report.md
```

输出：

```text
runs/summary/metrics.json
runs/summary/metrics.csv
runs/summary/report.md
runs/summary/runtime_estimate.json
```

报告同时显示原始 L2RE、论文值、相对偏差和预警。`pending` 表示对应正式训练尚未完成或日志缺失。

## 正式训练预计用时

当前尚未获得云端实测值，因此先使用“单 epoch 10 秒”作为示意假设。这个假设按当前代码口径包含训练和该 epoch 的测试评估；本仓库不是模板示例中的每 5 epoch 测试一次，而是每个 epoch 都测试。

| 项目 | 示意估算 |
|---|---:|
| 单 epoch 假设 | 10 秒 |
| 单个数据集 FT200 | `10 × 200 / 3600 = 0.556` 小时 |
| 两组 FT200 合计 | 约 1.11 小时 |
| 预计费用 | `1.11 × BitaHub 页面当前 GPU 单价` |

以上仅是未冒烟前的示意值，不代表真实性能，费用必须以 BitaHub 页面显示为准。完成两组 5-epoch 冒烟后执行：

```bash
cd /code/MoEPOT
python scripts/estimate_runtime.py \
  --runs-root /data/nestor_moepot_ns/runs \
  --gpu-price 页面显示的每小时单价
```

脚本把估算写入 `/data/nestor_moepot_ns/runs/summary/runtime_estimate.json`，使用：

```text
单个数据集 FT200 小时数 = 5-epoch 平均单 epoch 秒数 × 200 / 3600
预计费用 = 两个数据集 FT200 小时数之和 × BitaHub 当前 GPU 单价
```

正式训练开始后可用实际 epoch 时间再次更新估算。

## 环境复用

MoE-POT 不是 NESTOR，因此使用为它生成的等价共享环境脚本，不套用 NESTOR starter：

```bash
export ENV_DIR=/data/nestor_env/envs/nestor
bash /code/MoEPOT/scripts/setup_env_moepot.sh
export PATH="$ENV_DIR/bin:$PATH"
```

环境安装一次后，后续开发机和训练终端只需重新设置 `ENV_DIR`、`PATH`，不需要重复安装 PyTorch。

## 验收标准

- 四个 HDF5 文件的 shape 分别为 `(1000/200,64,64,20)` 和 `(1000/200,64,64,50)`，且没有 NaN/Inf。
- Tiny checkpoint 的 SHA256 已记录，全部模型键成功加载。
- zero-shot 两列均完成，参考值为 0.0682 和 0.00768。
- 两套 5-epoch 冒烟均有 5 条完整 epoch 日志和可加载 checkpoint。
- 两组 FT200 都完成到 epoch 199，参考值为 0.0456 和 0.00493。
- 最终模型、日志、指标和报告都在 `/data/nestor_moepot_ns/runs`。

## 操作 Tips：查看云端文件

```bash
DATA_ROOT=/data/nestor_moepot_ns
ls -lh /data/nestor_1e5
ls -lh /model/nestor_moepot_tiny
ls -lh "$DATA_ROOT/runs/zero_shot"
ls -lh "$DATA_ROOT/runs/finetune/ns1e-5/full"
ls -lh "$DATA_ROOT/runs/finetune/ns1e-3/full"
cat "$DATA_ROOT/runs/summary/report.md"
cat "$DATA_ROOT/runs/summary/runtime_estimate.json"
tail -n 10 "$DATA_ROOT/runs/finetune/ns1e-5/full/logs.txt"
tail -n 10 "$DATA_ROOT/runs/finetune/ns1e-3/full/logs.txt"
```

## 常见问题

`ModuleNotFoundError: MoEPOT`：确认代码目录名是 `/code/MoEPOT`，并从仓库根目录运行脚本。

checkpoint 分类头尺寸不匹配：确认评测使用 `--n_class 6`；本分支已修正 `evaluate.py` 从命令行参数读取类别数。

HDF5 shape 不匹配：重新查看 `asset_inventory.json`。标准转换后的轨迹应为 `(N,64,64,T)`，不含通道维。

Google Drive 无法下载：可在 BitaHub 网页侧上传该单个 MAT 文件，或稍后重试单文件链接；不要用递归目录下载。

共享环境版本冲突：设置 `ENV_DIR=/data/nestor_env/envs/moepot` 创建独立环境，避免修改现有 NESTOR 依赖。

CUDA OOM：先确认使用的是 Tiny 和 batch 20，并关闭其他占用 4090 的进程；不要把正式复现配置静默改成更小 batch。

训练异常中断：查看同目录 `nohup.out`、`logs.txt` 和 `pid`；正式结果只认完整执行到 epoch 199 的 `full` 目录。
