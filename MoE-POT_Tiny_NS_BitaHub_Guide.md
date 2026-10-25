# MoE-POT-Tiny NS 零样本与 FT200 BitaHub 操作指南

本指南按“完成一步、验收一步、再进入下一步”组织。每一步都先写操作，再写验收标准；验收通过后再继续。

## 目标

从公开的 MoE-POT-Tiny checkpoint 开始，在 BitaHub 单卡 RTX 4090 开发机上：

1. 记录 NS 1e-5、NS 1e-3 微调前的 zero-shot L2RE。
2. 分别对 NS 1e-5、NS 1e-3 独立执行 200 epoch 微调。
3. 比较论文 Table 1 中 MoE-POT-Tiny 的 zero-shot 和 FT200 结果。

```text
论文 zero-shot：
  NS 1e-5 = 0.0682
  NS 1e-3 = 0.00768

论文 FT200：
  NS 1e-5 = 0.0456
  NS 1e-3 = 0.00493
```

先记录原始结果。辅助预警线为 zero-shot 相对偏差不超过 1%、FT200 相对偏差不超过 5%；预警不替代实际数值。

## 新版 BitaHub 界面说明

本任务全程使用开发机，不使用分布式训练。

```text
开发&训练 -> 开发机：启动、上传、装环境、预检、训练
模型&推理 -> 模型库：保存公开下载的 Tiny checkpoint
数据&存储 -> 数据集：保存 NS 数据、环境和训练结果
数据&存储 -> 存储管理：存储桶
管理&配置 -> 代码管理：公开 fork 可选，本流程不依赖代码映射
```

## 参数

```text
作者代码：https://github.com/haiyangxin/MoEPOT.git
作者提交：c36ec22aa492b9460d2e9f24cf4f0354d0f07540
checkpoint：Hugging Face xhy2878/MoEPOT/model_moepot_tiny.pth
工作数据集：nestor_moepot_ns
已有 NS 1e-5：nestor_1e5
共享环境：nestor_env
模型资源：nestor_moepot_tiny
开发机：nestor_moepot_ns_ft
GPU：1 卡 RTX 4090
epochs：200
batch_size：20
learning_rate：1e-3
warmup_epochs：40
noise_scale：0.0
T_in：10
T_ar：1
T_bundle：1
```

NS 1e-5 使用 1000 条训练、最后 200 条测试、20 个时间步；NS 1e-3 使用前 1000 条训练、最后 200 条测试、50 个时间步。

## 执行方式

两套 NS 数据裁剪后小于 20GB，Tiny 可在单张 4090 上运行，因此使用开发机 `nohup`。两次 FT200 共用同一张卡，必须串行；正式队列会在 NS 1e-5 完成验收后才启动 NS 1e-3。

## 本地文件

代码包：

```text
moepot_bita_code.zip
```

代码包约数 MB，包含作者代码、MoE-POT 配置、数据脚本、环境脚本、训练入口、监控和结果汇总；不包含 HDF5 数据或 checkpoint。

代码有两种获取方式：

- 方式 A：本次分支已推送到你的公开 GitHub fork，开发机直接 clone。需要你后续提供公开仓库 URL。
- 方式 B（当前没有 fork URL 时使用）：开发机先 clone 作者公开仓库，再上传 `moepot_bita_code.zip` 覆盖复现文件。

两种方式都不使用 patch。

## 操作步骤

### 1. 登录并创建数据库、模型

#### 1.1. 创建共享环境数据集

进入：

```text
数据&存储 -> 数据集 -> 创建私有数据集
```

填写：

```text
数据集名称：nestor_env
存储桶：nestor-env-bucket
```

如果存储桶名称已存在，使用 `nestor-env-<你的用户名>`。该数据集只放 Python 环境，不放训练数据。

#### 1.2. 创建本次工作数据集

进入：

```text
数据&存储 -> 数据集 -> 创建私有数据集
```

填写：

```text
数据集名称：nestor_moepot_ns
存储桶：nestor-moepot-ns-bucket
```

如果名称已存在，使用 `nestor-moepot-ns-<你的用户名>`。它保存：

```text
/data/nestor_moepot_ns/raw
/data/nestor_moepot_ns/inventory
/data/nestor_moepot_ns/large
/data/nestor_moepot_ns/runs
/data/nestor_moepot_ns/metadata
```

#### 1.3. 创建模型资源

进入：

```text
模型&推理 -> 模型库 -> 创建私有模型
```

填写：

```text
模型名称：nestor_moepot_tiny
存储桶：nestor-moepot-tiny-bucket
```

不要选择平台“模型微调”。此时可以创建空模型，checkpoint 在第 7 步从公开 Hugging Face 仓库下载。

页面验收：

- 数据集页面能看到 `nestor_env` 和 `nestor_moepot_ns`。
- 模型页面能看到 `nestor_moepot_tiny`。
- 存储桶状态正常。

### 2. 创建开发机

进入：

```text
开发&训练 -> 开发机 -> 新建开发机
```

按成功示例填写：

```text
开发机名称：nestor_moepot_ns_ft
资源分区：包含 nestor_env、nestor_moepot_ns、nestor_moepot_tiny 的分区
资源类型：公共资源
实例规格：1 卡 RTX 4090
实例数量：1
系统盘：平台默认
镜像：pytorch:2.1.2-cuda11.8-cudnn8-py310-ubuntu22.04
访问方式：JupyterLab、SSH
启动命令：sleep 12h
```

如果有 CUDA 12.1 的 PyTorch 3.10 镜像，可以选择；本指南使用上述镜像验证。

#### 2.1. 存储挂载

开发机创建页只使用界面的三种挂载类型：

```text
数据库：nestor_moepot_ns -> /data/nestor_moepot_ns
数据库：nestor_1e5       -> /data/nestor_1e5
数据库：nestor_env       -> /data/nestor_env
模型：  nestor_moepot_tiny -> /model/nestor_moepot_tiny
```

`nestor_1e5` 用于读取已有 NS 1e-5，按只读使用。工作数据和环境需要读写。

创建页面没有“代码映射/训练代码”字段，不要寻找该字段。代码在开发机启动后再进入 `/code`。

验收标准：

- 开发机状态为“运行中”。
- 能看到 JupyterLab 和 SSH 入口。
- 页面上四个挂载项的名称和路径与上表一致。

#### 2.2. 当前开发机连接信息

本次创建好的开发机：

```text
Host：xj-member.bitahub.com
Port：42289
User：root
```

本地终端 SSH 登录：

```bash
ssh -p 42289 root@xj-member.bitahub.com
```

密码以 BitaHub 开发机详情页显示为准，不写入代码包或文档。开发机重启或重新创建后，端口可能变化，届时以页面实际信息为准。

验收标准：

- 使用上述命令可以进入开发机 shell。
- `hostname` 命令正常返回。
- `/data/nestor_moepot_ns`、`/data/nestor_1e5`、`/data/nestor_env` 和 `/model/nestor_moepot_tiny` 均可访问。

### 3. 创建必要目录

打开 JupyterLab Terminal：

```bash
mkdir -p /code /output
ls -ld /code /output /data /model
```

验收标准：四个目录都存在。数据和结果持久保存在 `/data`，checkpoint 在 `/model`，`/code` 和 `/output` 位于开发机系统盘。

### 4. 获取代码

#### 4.1. 方式 A：公开 fork

如果你已经把本分支推送到公开 GitHub fork，执行：

```bash
cd /code
git clone <YOUR_PUBLIC_FORK_URL> MoEPOT
cd /code/MoEPOT
git switch repro/moepot-tiny-ns-bita-v2
```

验收：

```bash
ls /code/MoEPOT/setup_env.sh /code/MoEPOT/run_moepot.sh
ls /code/MoEPOT/configs/repro_tiny_ft200_ns.yaml
```

#### 4.2. 方式 B：作者公开仓库 + 代码包

第 1 步 clone 作者公开仓库：

```bash
cd /code
git clone https://github.com/haiyangxin/MoEPOT.git
cd /code/MoEPOT
git fetch --depth 1 origin c36ec22aa492b9460d2e9f24cf4f0354d0f07540
git switch -c repro/moepot-tiny-ns-bita-v2 c36ec22aa492b9460d2e9f24cf4f0354d0f07540
```

第 2 步在本地 PowerShell 上传代码包；如果使用 JupyterLab 文件浏览器，也可以直接上传到 `/code/`：

```powershell
cd "本地代码包所在目录"
scp -P 42289 moepot_bita_code.zip root@xj-member.bitahub.com:/code/
```

第 3 步回到 JupyterLab Terminal 解压覆盖：

```bash
unzip -o /code/moepot_bita_code.zip -d /code
cd /code/MoEPOT
ls setup_env.sh run_moepot.sh run_all_ft200.sh monitor.sh
```

验收标准：

```bash
test -f /code/MoEPOT/scripts/prepare_assets.py
test -f /code/MoEPOT/scripts/download_tiny_checkpoint.py
test -f /code/MoEPOT/MoE-POT_Tiny_NS_BitaHub_Guide.md
echo "CODE_OK"
```

### 5. 安装共享环境

```bash
cd /code/MoEPOT
bash /code/MoEPOT/setup_env.sh
```

脚本优先复用镜像内的 PyTorch，环境固定保存到：

```text
/data/nestor_env/envs/nestor
```

后续终端使用：

```bash
export ENV_DIR=/data/nestor_env/envs/nestor
export PATH="$ENV_DIR/bin:$PATH"
```

验收标准，输出包含：

```text
torch: 2.1.x
cuda_available: True
gpu: NVIDIA GeForce RTX 4090
environment check: PASS
Environment ready: /data/nestor_env/envs/nestor
```

安装中断后重新运行同一命令。

### 6. 云端获取并处理数据

#### 6.1. 盘点已有 NS 1e-5

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash prepare_data.sh \
  --ns1e5-root /data/nestor_1e5 \
  --inventory-only
```

查看：

```bash
cat /data/nestor_moepot_ns/inventory/ns1e5.json
```

目标是识别规范 HDF5，或包含 1200 条轨迹、20 个时间步的 `u`/`data` 数组。此时不要求转换完成。

#### 6.2. 下载 NS 1e-3

打开论文 README 中的 FNO 数据文件夹：

```text
https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLnZaXrKUZhp7pjt-
```

只获取 `NavierStokes_V1e-3_N5000_T50.mat` 的单文件链接，并复制其 File ID：

```bash
mkdir -p /data/nestor_moepot_ns/raw
gdown --id "<FNO_1E3_FILE_ID>" \
  --output /data/nestor_moepot_ns/raw/NavierStokes_V1e-3_N5000_T50.mat
ls -lh /data/nestor_moepot_ns/raw
```

验收标准：文件存在，能被 HDF5/MAT 读取，变量为 `u`，总轨迹数 5000、时间步 50。

#### 6.3. 转换并注册两套数据

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash prepare_data.sh \
  --ns1e5-root /data/nestor_1e5 \
  --ns1e3-source /data/nestor_moepot_ns/raw/NavierStokes_V1e-3_N5000_T50.mat \
  --work-root /data/nestor_moepot_ns
```

如果 NS 1e-5 盘点后发现需要显式指定原始文件，追加：

```bash
--ns1e5-source /data/nestor_1e5/<实际文件>
```

验收：

```bash
ls -lh /code/MoEPOT/data/large/ns2d_1e-{5,3}_{train,test}.hdf5
cat /data/nestor_moepot_ns/metadata/repro_data_links.json
```

shape 必须为：

```text
(1000,64,64,20), (200,64,64,20)
(1000,64,64,50), (200,64,64,50)
```

脚本末尾应出现 `PREPARE_OK`。

### 7. 云端下载并校验 Tiny checkpoint

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
export HF_HOME=/data/nestor_moepot_ns/cache/huggingface
python scripts/download_tiny_checkpoint.py \
  --repo-id xhy2878/MoEPOT \
  --output /model/nestor_moepot_tiny/model_moepot_tiny.pth
```

脚本只选择 Tiny 权重，不下载其他模型，并执行 SHA256 和完整 state dict 加载。

如果模型卷不可写：

```bash
python scripts/download_tiny_checkpoint.py \
  --output /data/nestor_moepot_ns/models/model_moepot_tiny.pth
export MOEPOT_CHECKPOINT=/data/nestor_moepot_ns/models/model_moepot_tiny.pth
```

验收标准：manifest 中 `model_check` 为 `all Tiny model keys loaded successfully`。

### 8. 云上预检

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash run_moepot.sh --check-only
```

验收标准：

```text
CHECK OK: CUDA available
四个 HDF5 shape 正确
checkpoint_sha256: ...
model parameters: ...
forward output: (1, 128, 128, 1, 4) finite=True
CHECK OK
```

### 9. 记录 zero-shot

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash run_moepot.sh --eval-only --dataset all
cat /data/nestor_moepot_ns/runs/zero_shot/pretrained_eval.json
```

验收标准：JSON 中同时有：

```text
ns2d_fno_1e-5
ns2d_fno_1e-3
```

分别对照 `0.0682` 和 `0.00768`。如果差一个数量级、出现 NaN 或 checkpoint 键错误，停止后续微调。

### 10. 两组 5 epoch 冒烟

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash run_moepot.sh --dataset all --smoke-epochs 5
```

脚本先完成 NS 1e-5，再完成 NS 1e-3。查看：

```bash
cat /data/nestor_moepot_ns/runs/ns1e-5/smoke/status.json
cat /data/nestor_moepot_ns/runs/ns1e-3/smoke/status.json
tail -n 5 /data/nestor_moepot_ns/runs/ns1e-5/smoke/history.csv
tail -n 5 /data/nestor_moepot_ns/runs/ns1e-3/smoke/history.csv
```

验收标准：

- 两个 status 都是 `finished`。
- 每个 history 有 5 行数据。
- `test_l2_full` 为有限数。
- 没有 NaN 或 CUDA OOM。
- 每个 run 都有 `latest.pth`、`final.pth`、`best.pth` 和 `training_curve.png`。

冒烟结果不代表论文 FT200 结果。

## 正式训练预计用时

未冒烟时的示意假设：

```text
单 epoch：10 秒
单个数据集 FT200：10 x 200 / 3600 = 0.56 小时
两个数据集合计：约 1.12 小时
```

完成冒烟后执行：

```bash
cd /code/MoEPOT
python scripts/estimate_runtime.py \
  --runs-root /data/nestor_moepot_ns/runs \
  --gpu-price <页面当前每小时价格>
```

结果在：

```text
/data/nestor_moepot_ns/runs/summary/runtime_estimate.json
```

以上为估算，费用以 BitaHub 页面显示为准。

## 环境复用

环境只在第 5 步安装一次。后续 JupyterLab Terminal 或 SSH 新会话执行：

```bash
export ENV_DIR=/data/nestor_env/envs/nestor
export PATH="$ENV_DIR/bin:$PATH"
export MOEPOT_DATA_ROOT=/data/nestor_moepot_ns
```

如果 checkpoint 放在数据工作区而不是模型挂载，还要设置：

```bash
export MOEPOT_CHECKPOINT=/data/nestor_moepot_ns/models/model_moepot_tiny.pth
```

环境安装中断时直接重新运行 `bash /code/MoEPOT/setup_env.sh`，不需要重复创建 `/data/nestor_env`。

### 11. 串行运行两次 FT200

先确认没有旧训练进程：

```bash
ps -ef | grep -E 'finetune.py|run_all_ft200.sh' | grep -v grep
```

没有输出后启动串行队列：

```bash
mkdir -p /data/nestor_moepot_ns/runs/queue
nohup bash /code/MoEPOT/run_all_ft200.sh \
  > /data/nestor_moepot_ns/runs/queue/console.log 2>&1 &
echo $!
```

关闭 Jupyter 页面不会中断训练。查看状态：

```bash
bash /code/MoEPOT/monitor.sh
bash /code/MoEPOT/monitor.sh --follow
```

实时看当前数据集日志：

```bash
tail -f /data/nestor_moepot_ns/runs/ns1e-5/ft200/training.log
```

NS 1e-5 结束后，队列会自动切换到：

```text
/data/nestor_moepot_ns/runs/ns1e-3/ft200/training.log
```

`Ctrl-C` 只退出 `tail` 或监控，不会停止训练。

### 12. 完成与结果验收

队列结束后查看：

```bash
cat /data/nestor_moepot_ns/runs/queue/status.json
cat /data/nestor_moepot_ns/runs/ns1e-5/ft200/status.json
cat /data/nestor_moepot_ns/runs/ns1e-3/ft200/status.json
tail -n 5 /data/nestor_moepot_ns/runs/ns1e-5/ft200/history.csv
tail -n 5 /data/nestor_moepot_ns/runs/ns1e-3/ft200/history.csv
ls -lh /data/nestor_moepot_ns/runs/ns1e-5/ft200/{final,best,epoch_0200}.pth
ls -lh /data/nestor_moepot_ns/runs/ns1e-3/ft200/{final,best,epoch_0200}.pth
```

验收标准：

- 两个 FT200 status 都是 `finished`，最后 epoch 为 199。
- 两个 history 都有 200 行数据。
- 两个 run 都有 `final.pth`、`best.pth`、`epoch_0200.pth` 和 `training_curve.png`。
- FT200 参考值为 `0.0456` 和 `0.00493`；5% 偏差只作预警。

汇总结果：

```bash
cd /code/MoEPOT
python scripts/collect_metrics.py \
  --runs-root /data/nestor_moepot_ns/runs
cat /data/nestor_moepot_ns/runs/summary/report.md
```

输出：

```text
metrics.json
metrics.csv
report.md
```

## 验收标准

- 代码目录为 `/code/MoEPOT`，公开代码来源和提交清晰可追溯。
- 四个 HDF5 的 shape 分别为 `(1000/200,64,64,20)` 和 `(1000/200,64,64,50)`。
- Tiny checkpoint 完整加载，单批前向输出有限。
- zero-shot 原始值和论文参考值并列记录。
- 两组冒烟各有 5 行 history，状态为 `finished`。
- 两组 FT200 各有 200 行 history，最后 epoch 为 199。
- 每个正式 run 都有 `latest.pth`、`best.pth`、`final.pth`、`epoch_0200.pth` 和 `training_curve.png`。
- 最终汇总报告生成 `metrics.json`、`metrics.csv` 和 `report.md`。

## 暂停与恢复

训练中断后，如果 `latest.pth` 存在，可恢复单个数据集：

```bash
cd /code/MoEPOT
export ENV_DIR=/data/nestor_env/envs/nestor
bash run_moepot.sh \
  --dataset ns1e-5 \
  --epochs 200 \
  --resume-training /data/nestor_moepot_ns/runs/ns1e-5/ft200/latest.pth
```

恢复时保持 `epochs=200`、`batch_size=20`、同一份配置和同一个运行目录。

## 常见问题

### `/code/MoEPOT` 不存在

开发机启动后重新执行第 4 步。作者仓库是公开的，clone 不需要 GitHub Token。

### `git clone` 无法解析 GitHub

先执行：

```bash
git ls-remote https://github.com/haiyangxin/MoEPOT.git HEAD
```

这是开发机网络问题，不是模型或 checkpoint 问题。改用 JupyterLab 上传完整 `moepot_bita_code.zip` 后解压到 `/code/MoEPOT`。

### `prepare_data.sh` 找不到 NS 1e-5 源文件

查看：

```bash
cat /data/nestor_moepot_ns/inventory/ns1e5.json
```

将实际的 MAT/HDF5 文件路径通过 `--ns1e5-source <path>` 传入。

### HDF5 shape 不匹配

必须使用以下口径：

```text
NS 1e-5：训练 1000、测试 200、时间步 20
NS 1e-3：训练 1000、测试 200、时间步 50
空间尺寸：64 x 64
```

### checkpoint 分类头尺寸不匹配

使用公开 Tiny checkpoint，不要加载 Small 或 Medium；预检固定 `n_class=6`。

### CUDA OOM

停止训练并确认使用 Tiny、batch_size 20，关闭其他 4090 进程。正式复现不要静默改 batch size。

## 操作 Tips：查看云端文件

数据：

```text
/data/nestor_moepot_ns/large
/data/nestor_moepot_ns/metadata
```

模型：

```text
/model/nestor_moepot_tiny/model_moepot_tiny.pth
```

运行结果：

```text
/data/nestor_moepot_ns/runs/zero_shot
/data/nestor_moepot_ns/runs/ns1e-5/smoke
/data/nestor_moepot_ns/runs/ns1e-3/smoke
/data/nestor_moepot_ns/runs/ns1e-5/ft200
/data/nestor_moepot_ns/runs/ns1e-3/ft200
/data/nestor_moepot_ns/runs/summary
```

开发机运行中可直接在 JupyterLab 文件浏览器查看 `training_curve.png`、`history.csv` 和 `status.json`。

开发机关机后，可在 BitaHub 的 `nestor_moepot_ns` 数据集页面查看 `runs` 下的小文件。最终模型通常继续留在云端，不需要下载到本地。

如需从当前开发机下载轻量结果：

```bash
scp -P 42289 \
  root@xj-member.bitahub.com:/data/nestor_moepot_ns/runs/summary/report.md \
  .
```
