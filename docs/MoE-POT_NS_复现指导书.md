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

本指南统一使用以下名称。名称、角色和路径必须对应，后续脚本会按这些路径寻找数据和模型。

> **重要：本步骤全部在 BitaHub 页面中完成。** 开发机启动之前没有 JupyterLab、
> WebTerminal 或云端 shell，因此无法执行 `test`、`mkdir`、`nvidia-smi` 等命令。
> 本文所有终端命令都从“第 2 步：启动开发机并验收资源”开始执行。页面暂时看不到
> Jupyter 入口是正常现象，不代表资源创建失败。

| 资源类型 | BitaHub 中的名称 | 用途 | 挂载路径 | 权限 |
|---|---|---|---|---|
| 数据库 | `nestor_1e5` | NS 1e-5 原始或预处理数据 | `/data/nestor_1e5` | 只读 |
| 数据库 | `nestor_moepot_ns` | NS 1e-3、规范化 HDF5、权重缓存、日志和结果 | `/data/nestor_moepot_ns` | 读写 |
| 数据库 | `nestor_env` | 共享 Python/PyTorch 环境 | `/data/nestor_env` | 首次安装时读写 |
| 模型 | `nestor_moepot_tiny` | MoE-POT-Tiny checkpoint | `/model/nestor_moepot_tiny` | 初始上传时读写 |
| 文件存储 | `nestor_moepot_storage` | 代码补丁或其他文件中转；如界面不要求可不创建 | 以平台实际路径为准 | 读写 |
| 开发机 | `moepot-tiny-ns-4090` | 执行数据转换、zero-shot、冒烟和 FT200 | 代码映射到 `/code` | 读写 |

BitaHub 开发机挂载对话框中的资源类型只有“数据库、模型、文件存储”三个选项。
因此：

- 在“数据&存储 -> 数据集”中创建或引用的数据资源，在挂载时选择 `数据库`。
- Tiny checkpoint 资源选择 `模型`。
- 代码仓库不是数据库或模型挂载项，应在开发机配置的“代码映射/训练代码”中选择
  `MoEPOT-public`，映射路径填写 `/code`。
- 只有在需要上传补丁或临时文件时，才使用 `nestor_moepot_storage` 并选择
  `文件存储`；本流程的核心数据仍放在 `nestor_moepot_ns` 数据库中。

按新版界面操作，不使用旧版“项目”入口：

```text
开发&训练 -> 开发机
模型&推理 -> 模型库
数据&存储 -> 数据集
数据&存储 -> 存储管理
管理&配置 -> 代码管理
```

#### 1.1 确认现有 `nestor_1e5` 数据集

进入：

```text
数据&存储 -> 数据集
```

找到名称为 `nestor_1e5` 的数据集。不要新建同名资源，也不要在这里移动或覆盖原文件。打开数据集后确认其挂载路径为：

```text
/data/nestor_1e5
```

该数据集中可以暂时还是原始 MAT、工作 HDF5 或其他结构；第 5 步会盘点并转换，本步骤只负责挂载。

页面验收：数据集状态正常、名称确认为 `nestor_1e5`、挂载路径已设置为
`/data/nestor_1e5`。此时不要尝试执行终端命令。

#### 1.2 新建工作数据集 `nestor_moepot_ns`

进入：

```text
数据&存储 -> 数据集 -> 新建/创建数据集
```

填写：

```text
数据集名称：nestor_moepot_ns
用途/备注：MoE-POT-Tiny NS 1e-5/1e-3 zero-shot 与 FT200
期望挂载路径：/data/nestor_moepot_ns
权限：开发机读写
```

如果界面要求版本名称或备注，可填写：

```text
版本/标签：tiny-ns-ft200
```

该数据集不是原始数据备份，而是本次实验的持久化工作区，后续保存：

```text
/data/nestor_moepot_ns/raw                       NS 1e-3 下载文件
/data/nestor_moepot_ns/inventory                 数据盘点 JSON
/data/nestor_moepot_ns/large                     规范化 train/test HDF5
/data/nestor_moepot_ns/models                    权重卷不可写时的 checkpoint
/data/nestor_moepot_ns/runs                      zero-shot、冒烟、FT200 和汇总结果
/data/nestor_moepot_ns/cache/huggingface         Hugging Face 下载缓存
```

#### 1.3 确认共享环境数据集 `nestor_env`

进入：

```text
数据&存储 -> 数据集
```

优先查找名称为 `nestor_env` 的共享环境数据集。如果已经存在，直接复用；如果不存在，则创建：

```text
数据集名称：nestor_env
用途：AI4S 项目共享 Python/PyTorch 环境
期望挂载路径：/data/nestor_env
权限：首次安装环境时读写，后续训练可只读
```

本实验只在该环境内安装一次 PyTorch 和依赖，不重复创建每套数据自己的环境。实际 Python 路径为：

```text
/data/nestor_env/envs/nestor/bin/python
```

#### 1.4 创建底层存储桶

如果创建数据集或模型时，BitaHub 要求先选择或新建“存储桶”，进入：

```text
数据&存储 -> 存储管理
```

新建并统一使用：

```text
存储桶名称：nestor_moepot_storage
用途：承载 nestor_moepot_ns、nestor_moepot_tiny 及代码上传中转
权限：本次开发机读写
```

如果平台使用“对象存储”或“文件存储”两个不同入口，选择能够持久化到开发机的文件存储；不要选择实例重启后消失的临时存储。训练结果仍统一写入 `/data/nestor_moepot_ns/runs`，不要把结果写到页面无法再次挂载的临时目录。

#### 1.5 新建 Tiny 模型资源 `nestor_moepot_tiny`

进入：

```text
模型&推理 -> 模型库
```

创建：

```text
模型名称：nestor_moepot_tiny
模型版本/标签：tiny-ns-ft200
模型说明：MoE-POT-Tiny 30M，预训练 checkpoint
挂载路径：/model/nestor_moepot_tiny
权限：首次下载或上传 checkpoint 时读写
```

此时可以先创建空模型资源，第 8 步再从 Hugging Face 下载 `model_moepot_tiny.pth` 并登记。

#### 1.6 配置代码仓库

进入：

```text
管理&配置 -> 代码管理
```

点击“新建授权代码”或同等入口后，按 BitaHub 表单逐项填写：

| BitaHub 表单字段 | 填写内容 | 说明 |
|---|---|---|
| 授权代码名称 | `MoEPOT-public` | 建议使用这个名称区分作者公开仓库和后续本地复现分支 |
| 授权仓库类型 | 页面分组标题，不需要单独输入 | 在下方的仓库类型下拉项中选择 |
| 请选择授权仓库类型 | `GitHub` | 代码实际托管在 GitHub；如果另有公开/私有选项，选择公开仓库 |
| 授权仓库地址 | `https://github.com/haiyangxin/MoEPOT.git` | 作者公开仓库；不要填写本地目录或 `repro_changes.patch` |
| Access Tokens key | 按下方规则填写 | 不要把真实 Token 写进本指南、日志、代码或聊天消息 |

本实验使用的作者仓库是公开仓库，因此 Access Token 按以下顺序处理：

1. 如果 `Access Tokens key` 标为可选并且允许留空，先留空并点击授权或连接测试。
2. 如果 BitaHub 强制要求填写，使用你自己的 GitHub Personal Access Token。
3. 如果 GitHub 提供细粒度 Token，只授权 `haiyangxin/MoEPOT` 的只读仓库内容权限。
4. 如果只能使用 Classic Token，公开仓库只读通常不需要勾选额外权限。
5. 不要把 Token 发给我，也不要执行 `git remote -v` 后公开包含 Token 的 URL。

##### 1.6.1 优先创建 Fine-grained Personal Access Token

在浏览器中登录你自己的 GitHub 账号，按以下路径进入设置：

```text
GitHub 右上角头像
-> Settings
-> Developer settings
-> Personal access tokens
-> Fine-grained tokens
-> Generate new token
```

填写或选择：

| GitHub 字段 | 推荐值 |
|---|---|
| Token name | `bitahub-moepot-readonly-20261025` |
| Expiration | `7 days`，复现完成后可提前撤销 |
| Resource owner | 你自己的个人账号 |
| Repository access | 如显示 `Public Repositories (read-only)`，选择它 |
| Repository permissions | 不需要额外添加权限，全部保持 `No access` |
| Contents | 不需要手动配置 |
| 其他权限 | 全部保持 `No access` |

GitHub 官方文档明确说明：每个 Fine-grained Token 都自动包含 GitHub 上所有公开
仓库的只读访问。因此 `haiyangxin/MoEPOT` 这种公开仓库**不需要**在
`Repository permissions` 中再寻找 `Contents: Read-only`。看不到这个选项是正常的，
不要把 `Contents` 改成 `Read and write`，也不需要新增任何权限。

BitaHub 只需要读取公开仓库 `haiyangxin/MoEPOT`，不需要修改 GitHub 仓库、
Actions、Secrets 或用户资料。

如果在 `Repository access` 中：

- 有 `Public Repositories (read-only)`：直接选择。
- 只有 `All repositories` 和 `Only select repositories` 两个选项：不要为了作者仓库
  授权全部私有仓库；保留最小范围设置，并将 `Repository permissions` 全部留为
  `No access`，公开仓库读取能力仍然自动存在。

点击 `Generate token` 后会显示一次类似下面的 Token：

```text
github_pat_xxxxxxxxxxxxxxxxxxxxxxxx
```

真实 Token 只显示一次。立即复制，填入 BitaHub 的 `Access Tokens key`：

```text
Access Tokens key：github_pat_填入完整Token
```

粘贴时只填写 Token 本身：

- 不要添加 `Bearer `、`token ` 或引号。
- 不要带入换行和多余空格。
- 不要把这个值写入指南、项目文件、shell 历史或聊天消息。
- 不要使用别人的 Token。

##### 1.6.2 Fine-grained Token 不可用时的 Classic Token

如果 BitaHub 明确提示“无法识别该 API key”或“无法识别 Access Tokens key”，
而同一个 Fine-grained Token 通过 GitHub API 验证正常，这说明是 BitaHub 对
`github_pat_...` 格式的兼容问题，不需要继续重复创建 Fine-grained Token。

另外，如果 Fine-grained Token 授权后仍返回 `403`，也使用下面的 Classic Token
方案。回到：

```text
GitHub 右上角头像
-> Settings
-> Developer settings
-> Personal access tokens
-> Tokens (classic)
-> Generate new token (classic)
```

填写：

```text
Note：bitahub-moepot-readonly
Expiration：7 days
Scopes：public_repo
```

`public_repo` 是访问公开仓库所需的最小 Classic scope。不要勾选同时覆盖私有仓库的
`repo`，更不要勾选 `admin:org`、`delete_repo`、`workflow` 等无关权限。
生成的 Classic Token 通常以 `ghp_` 开头，将完整 Token 原样填入 BitaHub 的
`Access Tokens key`，不要添加 `Bearer`、`token` 或引号。

如果 BitaHub 在你已经改用 `ghp_...` Classic Token 后仍提示无法识别，应停止反复
生成 Token，记录页面原始报错，并检查 BitaHub 的“授权仓库类型”是否确实选择了
GitHub，而不是其他代码平台。

Token 的独立连通性验证需要终端，因此不在本步骤执行。开发机启动后，可在第 3 步
使用 GitHub API 验证并排查 `401/403`；这一步是可选的，因为 BitaHub 提交授权时
也会进行一次连接检查。

完成第 3 步并确认 `/code/MoEPOT` 可用后，应在 GitHub 中撤销这个短期 Token：

```text
Settings -> Developer settings -> Personal access tokens
-> 找到 bitahub-moepot-readonly-20261025
-> Revoke/Delete
```

授权完成后返回：

```text
管理&配置 -> 代码管理
```

确认列表中显示：

```text
授权代码名称：MoEPOT-public
仓库类型：GitHub
仓库地址：https://github.com/haiyangxin/MoEPOT.git
连接状态：正常/已授权
```

如果创建开发机时需要选择授权代码，选择 `MoEPOT-public`，代码映射路径填写：

```text
/code
```

BitaHub 通常会把仓库内容放到 `/code/MoEPOT`。开发机启动前只记录这个目标路径，
不要尝试检查目录。启动后如果平台自动生成的是其他目录，例如
`/code/MoEPOT-main`，在第 3 步进入 WebTerminal 后规范目录名。

不要直接把内容复制到 `/code` 根目录，因为后续命令和 Python 导入都要求仓库目录名为
`MoEPOT`。开发机中的最终路径必须是：

```text
/code/MoEPOT
```

目录名不能保留为 `MoEPOT-main`、`moe-pot` 或其他名称，因为作者代码使用 `MoEPOT.models...` 导入。

作者仓库授权只提供作者 `main` 分支，不包含本次生成的本地复现分支。因此还需要在
开发机启动后，通过 WebTerminal 的文件上传功能，把本地 `repro_changes.patch`
上传到云端 `/code/repro_changes.patch`。创建分支和应用补丁的具体命令统一放在
第 3 步执行。

如果 BitaHub 的代码授权允许选择分支，选择作者仓库的 `main` 分支；不要选择其他实验分支。

#### 1.7 创建开发机 `moepot-tiny-ns-4090`

进入：

```text
开发&训练 -> 开发机 -> 新建开发机
```

填写：

```text
开发机名称：moepot-tiny-ns-4090
实例规格：1 卡 RTX 4090
实例数量：1
训练代码/工作目录映射：/code
共享内存：建议 102400 MB
```

该实验使用单卡开发机，不需要创建分布式训练任务。zero-shot、冒烟和两次 FT200 都在此开发机中顺序执行。

#### 1.8 配置开发机资源挂载

启动开发机之前，先在其创建页面或“存储挂载”设置中配置以下四项数据库/模型资源：

| 挂载顺序 | 资源名称 | 挂载类型下拉选项 | 挂载路径 | 权限 |
|---:|---|---|---|---|
| 1 | `nestor_1e5` | 数据库 | `/data/nestor_1e5` | 只读 |
| 2 | `nestor_moepot_ns` | 数据库 | `/data/nestor_moepot_ns` | 读写 |
| 3 | `nestor_env` | 数据库 | `/data/nestor_env` | 读写 |
| 4 | `nestor_moepot_tiny` | 模型 | `/model/nestor_moepot_tiny` | 读写 |

代码仓库 `MoEPOT-public` 不在这四个资源挂载项中。在开发机创建页面的
“代码映射/训练代码”字段选择 `MoEPOT-public`，填写映射路径：

```text
/code
```

如果页面还要求选择文件存储资源，才添加：

| 资源名称 | 挂载类型下拉选项 | 挂载路径 | 权限 |
|---|---|---|---|
| `nestor_moepot_storage` | 文件存储 | 按平台生成的路径填写 | 读写 |

此时只把挂载配置保存到开发机定义中，不要尝试在浏览器页面中执行目录命令。

#### 1.9 启动前页面检查

点击“创建/保存”开发机之前，在页面中逐项确认：

- 现有数据集名称为 `nestor_1e5`。
- 新工作数据集名称为 `nestor_moepot_ns`。
- 共享环境数据集名称为 `nestor_env`。
- 模型资源名称为 `nestor_moepot_tiny`。
- 开发机名称为 `moepot-tiny-ns-4090`，规格为 1 卡 RTX 4090。
- 三个数据库资源和一个模型资源已加入挂载列表。
- 代码选择 `MoEPOT-public`，并映射到 `/code`。
- 数据、环境、模型和代码不在实例的临时系统盘中。

以上检查只看页面字段，不需要终端。确认无误后点击创建并启动开发机。

### 2. 启动开发机并验收资源

#### 2.1 启动 `moepot-tiny-ns-4090`

进入：

```text
开发&训练 -> 开发机
```

找到开发机 `moepot-tiny-ns-4090`，点击“启动”或“运行”。在实例状态变为
`运行中/RUNNING`、初始化完成后，页面才会出现 `JupyterLab`、`WebTerminal`
或同等入口。

打开 `JupyterLab` 或 `WebTerminal`，进入终端。后续命令都在这个已启动的开发机
终端中执行。

页面验收：

- 开发机状态为运行中。
- 实例规格显示 1 卡 RTX 4090。
- 页面上可以看到 JupyterLab 或 WebTerminal 入口。

#### 2.2 初始化持久化目录

在已启动开发机的终端中执行：

```bash
mkdir -p /data/nestor_moepot_ns/{raw,inventory,large,runs,cache/huggingface}
mkdir -p /data/nestor_moepot_ns/models
mkdir -p /code
```

#### 2.3 挂载验收

依次检查资源路径、写权限和 GPU：

```bash
test -d /data/nestor_1e5 && echo "PASS nestor_1e5"
test -d /data/nestor_moepot_ns && echo "PASS nestor_moepot_ns"
test -d /data/nestor_env && echo "PASS nestor_env"
test -d /model/nestor_moepot_tiny && echo "PASS nestor_moepot_tiny"
test -d /code && echo "PASS code"

touch /data/nestor_moepot_ns/.write_test
test -f /data/nestor_moepot_ns/.write_test && echo "PASS write access"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h /data/nestor_moepot_ns /model/nestor_moepot_tiny /code
```

验收结果应为：

```text
五个路径全部存在
工作数据卷写入成功
nvidia-smi 显示 NVIDIA GeForce RTX 4090 和约 24GB 显存
df -h 可读取三个持久化或映射目录
```

如果某个路径不存在，回到开发机挂载设置检查“资源名称”和“容器内挂载路径”是否同时填写；只填写资源名称或只填写路径都不够。

### 3. 放置代码

#### 3.1 可选：验证 GitHub Token

开发机启动以后才能在 BitaHub 终端验证 Token。如果第 1.6 步授权失败，而 Token
还在手上，可以执行：

```bash
read -r -s -p "GitHub Token: " GITHUB_TOKEN
echo
curl -i \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  https://api.github.com/repos/haiyangxin/MoEPOT
unset GITHUB_TOKEN
```

成功时应看到 `HTTP/2 200`，并返回：

```text
"full_name": "haiyangxin/MoEPOT"
"private": false
```

常见错误：

- `401 Bad credentials`：Token 不完整、粘贴错误、包含多余空格或已经过期。
- `403` 或 `404`：仓库访问范围设置有误，或者 Token 已被撤销。
- BitaHub 提示无法识别 `github_pat_...`，但本命令返回 `200`：改用第 1.6.2 节的
  Classic Token `public_repo`。

#### 3.2 查找或克隆授权代码

第 1.6 步完成代码授权、第 2 步启动开发机后，先检查 BitaHub 是否已经把授权仓库
放到 `/code/MoEPOT`，并规范可能的自动目录名：

```bash
cd /code
if [[ -d /code/MoEPOT-main && ! -e /code/MoEPOT ]]; then
  mv /code/MoEPOT-main /code/MoEPOT
fi
if [[ ! -d /code/MoEPOT/.git ]]; then
  echo "未发现 /code/MoEPOT，使用公开仓库地址克隆"
  git clone https://github.com/haiyangxin/MoEPOT.git
fi
cd /code/MoEPOT
test -d .git
git rev-parse --show-toplevel
```

#### 3.3 应用复现补丁并创建本地分支

通过 WebTerminal 上传补丁，并确认文件存在：

```bash
ls -lh /code/repro_changes.patch
test -s /code/repro_changes.patch
```

固定作者提交并创建本地复现分支：

```bash
git fetch --depth 1 origin c36ec22aa492b9460d2e9f24cf4f0354d0f07540
if git show-ref --verify --quiet refs/heads/repro/tiny-ns-ft200; then
  git switch repro/tiny-ns-ft200
else
  git switch -c repro/tiny-ns-ft200 c36ec22aa492b9460d2e9f24cf4f0354d0f07540
fi
git apply --check /code/repro_changes.patch
git apply /code/repro_changes.patch
```

不要上传数据或模型到 `/code`；这里只上传约几十 KB 的补丁。

验收：

```bash
cd /code/MoEPOT
git branch --show-current
# repro/tiny-ns-ft200
git merge-base --is-ancestor c36ec22aa492b9460d2e9f24cf4f0354d0f07540 HEAD
echo "base commit check: PASS"
test -f scripts/run_zero_shot.sh
```

### 4. 准备共享环境

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

### 5. 盘点现有 NS 1e-5

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

### 6. 在云端获取 NS 1e-3

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

### 7. 转换并注册 NS 1e-3

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

然后把两套数据注册到仓库预期路径。下面的 train/test 变量应替换为步骤 5
和本步骤生成的实际文件路径：

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

### 8. 下载并校验 Tiny checkpoint

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

### 9. 运行 zero-shot

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

### 10. 各跑 5 个冒烟 epoch

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

### 11. 正式训练 FT200

#### 11.1 启动 NS 1e-5

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

#### 11.2 实时查看 NS 1e-5 状态

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

#### 11.3 NS 1e-5 完成验收

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

### 12. 汇总结果

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
