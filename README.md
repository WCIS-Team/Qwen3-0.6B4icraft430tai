# Qwen3-0.6B4icraft430tai · 迁移交付件

> 这条线的仓：把 `Qwen/Qwen3-0.6B` 推到 iCraft（30TAI／ZG330）的 `_ZG` 件，量完 PC 侧数值，再把件送上板做完四档预检。**结构几何＝随库 `board/config.json` 实测键值**：28 层／`hidden_size` 1,024／`num_attention_heads` 16／`num_key_value_heads` 8／`head_dim` 128／`intermediate_size` 3,072／`vocab_size` 151,936（含 pad）／`tie_word_embeddings` true／`torch_dtype` bfloat16／`max_position_embeddings` 40,960；分词侧＝`tokenizer.json` 里 BPE **vocab 151,643＋merges 151,387＋added tokens 26**、`normalizer` NFC。**参数量按上游标注 ≈ 0.6 B，本仓没有逐张量复算过**。
> **图上编译与逐 token 数值已闭环；板上生成从未执行**（写本页时仍是这个状态，见「状态」节）。
> 全部读数与流程＝本目录 [`pipeline.md`](pipeline.md)（正本，§0～§9：现在到哪一步／主机分工／阶段 1～6（含 §4b mask 补丁）／产物清单／还没做）；**本页只做索引、来源、许可与上手，不复制读数表**。

## 来源与许可

| 面 | 出处 | 许可 |
|---|---|---|
| 模型本体 | `Qwen/Qwen3-0.6B`（ModelScope `https://modelscope.cn/models/Qwen/Qwen3-0.6B`／HuggingFace `https://huggingface.co/Qwen/Qwen3-0.6B`，取法＝官方工具，见下） | Apache-2.0（Copyright Alibaba Cloud） |
| 结构与几何读数 | 上游 `config.json`（28 层／`num_key_value_heads` 8／`head_dim` 128／`vocab_size` 151,936），不是从目录名或加载行为推的 | 同上 |
| 导出与编译工具链 | iCraft **3.39.0** CLI（ZG 档 imodel 只能在 Windows CLI 出）＋ `genai` 导出（fp16/cuda 档） | 厂商专有，**不随本仓分发**；本仓只放我方脚本与配置 |
| 板端运行时 | FMSH 悟净 FMQL30TAI676H（Ubuntu 20.04／Python 3.8.10／ZG330 NPU）；`modelzoo_utils` 来自厂商 Qwen2 demo 包 | 厂商专有，不随本仓分发 |
| **本仓所有我方件**（改写脚本、`_ZG.json`、板侧驱动与探针、读数留底） | WCIS 项目 | Apache-2.0，见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE) |

上游取回（官方工具，不用 `curl`／`wget`）——[`compile/dl_qwen3.py`](compile/dl_qwen3.py) 就是这条命令的现役脚本，改 `local_dir` 即可：

```bash
python -c "from modelscope import snapshot_download; \
print(snapshot_download('Qwen/Qwen3-0.6B', local_dir='./Qwen3-0.6B', allow_patterns=[ \
'config.json','generation_config.json','tokenizer_config.json','tokenizer.json', \
'vocab.json','merges.txt','special_tokens_map.json','chat_template.json','*.safetensors']))"
```

## 不随本仓分发的东西，以及怎么再拿到

体积件与"源上有正本"的分词件一律挡在库外（[`.gitignore`](.gitignore)），下面这张表就是取回合同：**逐件带字节数与 md5，拿到后先对上这两个数再用**。

| 件 | 字节数 | md5 | 为什么不进库 | 怎么再拿到 |
|---|---|---|---|---|
| `board/tokenizer.json` | 11,422,654 | `6423133b9cc1a2077b57822c30c211aa` | 上游重取件 | 上游仓库根目录那颗；**与板上现役那颗不是同一份**（见「已知坑」第 2 条） |
| `export/q3i_fp16_cuda/tokenizer.json` | 11,422,650 | `0fa489dfe491c3835067a33b4d7dea6e` | 同上 | 导出目录里那份副本 |
| `export/q3i_fp16_cuda/model.onnx` | 248,431 | `450ada9d95f7d5453e5b3b2b45dbb69f` | 导出产物，可复跑 | 按 [`pipeline.md`](pipeline.md) §2 用 genai 导出（fp16/cuda 档） |
| `export/q3i_fp16_cuda/model.onnx.data` | 1,202,585,600 | `b2e1e072856a8254df728cb63879eee0` | 权重体，>1 GB | 同上（`.onnx` 与 `.data` 成对产出） |
| `Q3V1-decoder_ZG.raw`（板端参数体） | 1,603,399,624 | `1081ddd7222dc2cadd5bee8dc27b781f` | >1.5 GB，GitHub 单文件上限 100 MB 会拒整笔 push | 两条路：① 从板上 `/root/qwen3/jr_path/decode/Q3V1-decoder_ZG.raw` 取回；② 按 [`pipeline.md`](pipeline.md) §4 重跑 `generate` 出件。**判据不变**：`.raw` 的 md5 必须等于库里 `board/Q3I-decoder_ZG.json` 内那处 `params_md5` |
| 五段中间件（parsed／optimized／quantized／adapted 的 json＋raw） | 各 1.5 GB 级 | — | 体积、可复跑 | 编译根按 [`pipeline.md`](pipeline.md) §4 重跑；cwd 必须是 `compile/` |
| 上游权重快照（`*.safetensors`） | 1.2 GB 级 | — | 上游重取件 | 上面 `modelscope` 那条命令 |

**库里唯一别处拿不到的产物**＝`board/Q3I-decoder_ZG.json`（2,679,170 B／`47699b3fd3d87d9056169d9072517f38`）：图上编译的输出、上游没有这颗、也是重编一次的产物名，2.7 MB 直接进库。

## ⚠️ 板侧两条运行依赖（装不上就白跑）

### 1. 分词：`tokenizers` 要 ≥ 0.15，实测用 0.20.3

板上出厂那颗 `tokenizers 0.13.3` 读不了这颗 `tokenizer.json`——`Tokenizer.from_file` 直接报
`data did not match any variant of untagged enum ModelWrapper`；升到 0.20.3 后**同一颗文件**直接可读。

失败点用最小用例锁在 `model.merges` 的**序列化格式**上（2026-11-04 板上跑 `tok_bisect.py`；2026-11-06 本机直读四颗文件核对）：

| 文件 | `model.merges` 写法 | `ignore_merges` | 板上 0.13.3 |
|---|---|---|---|
| 厂商 Qwen2 那颗（`1c74fd33…`，151,387 条） | `"Ġ Ġ"` 空格连接字符串 | 无此键 | **读通**（demo 真跑过本机分词） |
| 板上现役这颗 Qwen3（`7875a23b…`，151,387 条） | `["Ġ","Ġ"]` 成对数组 | `False` | 报 `ModelWrapper` |
| 库里这颗 Qwen3 留档（`6423133b…`） | 成对数组 | `False` | 同上 |
| 同项目 ERNIE 那颗（182,926 条） | 成对数组 | `False` | 同上 |
| 板上合成用例 | `[["a","b"]]` ⇒ FAIL；`[]` ⇒ OK | 不带 | 两半都是当场跑的 |

⇒ **A（实测）**：0.13.3 认不了成对数组写法的 `merges`，升版即解，与文件本身好坏无关。
⇒ **未取证**：`model.ignore_merges` 这个键（上表里两颗新文件有、厂商那颗没有）从没被单独摘出来对照过，所以"只差 merges 格式"这句写不到 A。补这一格只要 30 秒——把那颗文件摘掉 `ignore_merges` 再加载一次：

```bash
python3 - <<'EOF'
import json
from tokenizers import Tokenizer
d = json.load(open('/root/qwen3/config/qwen3/tokenizer.json', encoding='utf-8'))
d['model'].pop('ignore_merges', None)
json.dump(d, open('/tmp/no_im.json', 'w', encoding='utf-8'))
try:
    Tokenizer.from_file('/tmp/no_im.json'); print('no ignore_merges -> OK')
except Exception as e:
    print('no ignore_merges -> FAIL', str(e)[:90])
EOF
```

两条脚本随库：[`board/tok_bisect.py`](board/tok_bisect.py)（就是 11-04 板上跑的那份，逐字未改，`merges` 两半是它里面的合成用例）＋[`board/tok_load_test.py`](board/tok_load_test.py)（下面验收那 7 组样例）。

装 0.20.3（cp38／aarch64；PyPI 上带 cp38 aarch64 轮子的最后几版＝0.15.2／0.19.1／0.20.3，**0.21 起无 cp38**）：

```bash
# 有网的机器上取轮子（项目内部留底那份：2,893,923 B / fc747d88688d1b1609b670feb04ce53a）
python3 -m pip download "tokenizers==0.20.3" --no-deps --only-binary=:all: \
  --python-version 3.8 --implementation cp --abi cp38 --platform manylinux_2_17_aarch64 -d whl
# 板上（root，离线装）
python3 -m pip install --no-index --no-deps tokenizers-0.20.3-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

**验收**（2026-11-04 板上实跑）：`load OK`，7 组样例 ids 与本机 `tokenizers 0.23.3` 逐字相同——`你好，世界`→`[108386, 3837, 99489]`；`<|im_start|>user\nHi<|im_end|>`→`[151644, 872, 198, 13048, 151645]`。

对照：同项目 ERNIE 那条线没升 `tokenizers`，改用纯标准库实现（`board/e41_tok.py`＋预打数据包），两条路都通，**别混着用**。

### 2. NPU 绑定：板上 `import icraft` 要拼一颗 3.39 的包

板上 pip 里的 `icraft` 是 **3.33.1**（`xir.so` 是 2024-12 的），系统 `.deb` 里的库是 **3.39.0**（与出这颗 `_ZG` 件的 CLI 同版）。旧绑定引用 `icraft::xir::Params::load()`，新版 `libicraft_xir.so` 里已无此符号 ⇒ 直接 `import icraft` 报 undefined symbol。

解法（板上已实测通）：用 **3.39 的 `.so` ＋ 3.33 的 `__init__.py`／`.pyi`** 拼一个包放在 `/root/icraft39`，再 `PYTHONPATH=/root/icraft39`（本仓 `board/npuserv_adapter.py` 里已 `sys.path.insert`，不必每次设环境变量）。`.so` 正本在板子的系统包里，本仓不带（`*.so`／`*.whl` 在 `.gitignore` 里）。

## 布局

| 子目录／件 | 内容 | 入库 |
|---|---|---|
| `pipeline.md` | 流程正本（§0～§9，含 §4b）：现在到哪一步／主机分工／阶段 1 导出／阶段 2 两处结构改写／阶段 3 五段编译／§4b mask 补丁／阶段 4 PC 前向出 token／阶段 5 数值校验与口径实验／阶段 6 上板（含门禁）／产物清单／还没做 | 是 |
| `compile/` | 本线现役脚本 **24 件**＝18 `.py`＋4 `.ps1`＋2 `.toml`：`dl_qwen3.py`（上游取回）、`meta_variant.py`／`detranspose_embed.py`／`padfix.py`／`patch_mask0.py`（图改写）、`mk_parse_toml.py`／`Q3I_parse.toml`／`Q3I_gen_mask0zg.toml`（编译配置）、`host_gen.py`（本机 Host 后端逐级链驱动）、`cmp_chain.py`／`cmp_logits.py`／`verify_onnx.py`／`ref_hf.py`（数值对照）、`dump_mask.py`／`masktrace.py`／`fmtrace.py`／`so_probe.py`／`so_ctx.py`／`decode_ids.py`／`vendorize.py`（清点与探针）、`gen_fixed.ps1`／`fixcheck_run.ps1`／`variants_run.ps1`／`setup-school.ps1`（跑批） | 是 |
| `board/` | 上板件与板侧脚本 9 件：`Q3I-decoder_ZG.json`（2.7 MB，图上编译产物）、`config.json`（模型结构，本页上面那行几何数字就是从它读的）、`npuserv_adapter.py`（npuserv 单槽服务里的 Qwen3 adapter）、`qwen_board.py`（paramiko 长跑执行器，**口令从环境变量取、本仓这份不带明文**）、`mem_watch.sh`（心跳采样）、`sercap.ps1`／`ser_probe.ps1`（串口兜底取证）、`tok_bisect.py`／`tok_load_test.py`（分词依赖那两条证据脚本） | json／脚本入库，`tokenizer.json` 否 |
| `patch/` | 两支板侧驱动：`q3_infer_board.py`（本仓这份实算 **11,109 B／`1ea294244363291747988d871085b848`**＝与 HA4FPAI 里那颗同 md5；专为 Qwen3 写的 decode-only 驱动，28 层／8 KV head／head_dim 128，`token_dynamic_fm` 喂 3 个 fp32。与 10-30 那版记账 10,586 B／`e2364dddefcafb40e8293b9b0ffab75a` 的差＝11-04 给 `:95`～`:99` 加的 tokenizer 加载兜底那五行，装上 0.20.3 后不再触发、留着无害）、`qwen3_infer_pad3.py`（23,602 B／`ce087fbcf3c5305ca0d41e3c3421e30a`，厂商驱动的同源改造副本；`:469` 的 chat template 仍是厂商 Qwen2 形，与本仓 `export/…/chat_template.jinja` 没对过） | 是 |
| `export/q3i_fp16_cuda/` | genai 导出件里的小件 3 份：`chat_template.jinja`（4,256 B）、`genai_config.json`、`tokenizer_config.json` | 是 |
| `ref/` | 数值基准留底：`padfix_mask0_25.json`、25 token 链的 `steps.json`／`tokens.txt` | 是 |
| `LICENSE`／`NOTICE`／`.gitignore`／`MANIFEST.md5` | Apache-2.0 正文／来源声明／出库规则／逐件 md5 清单 | 是 |
| `Q3V1-decoder_ZG.raw`、`model.onnx(.data)`、`tokenizer.json`、五段中间件、上游权重 | 见上面那张表 | **否**（体积件／可重取件） |

## 上手（要在板上真跑，得推三件＋注册一条）

1. **npuserv 服务端**：`npuserv/`（`server.py`＋`adapter_lib.py`＋`wire.py`＋`models.json`）——它在同项目的 HA4FPAI 仓里，本仓不带。
2. **本仓两件**：`board/npuserv_adapter.py` 与 `patch/q3_infer_board.py`（后者 `:31` 的 `sys.path.append` 指向厂商 Python 运行时 `/root/qwen2/modelzoo_utils`，那颗目录下没有这份件）。
3. **registry 条目**（`models.json` 里 `id`＝`qwen3-0.6b`）——板上路径版：

   ```json
   {
     "id": "qwen3-0.6b",
     "artifacts": {
       "decode_json":     "/root/qwen3/jr_path/decode/Q3V1-decoder_ZG.json",
       "decode_raw":      "/root/qwen3/jr_path/decode/Q3V1-decoder_ZG.raw",
       "tokenizer":       "/root/qwen3/config/qwen3/tokenizer.json",
       "modelzoo_utils":  "/root/qwen2/modelzoo_utils"
     },
     "defaults": { "max_new": 32, "stop_ids": [151645, 151643], "pos_rule": "vendor",
                   "vocab_size": 151936, "n_layers": 28, "n_kv_heads": 8, "head_dim": 128,
                   "max_decode_length": 512, "config_path": "/root/qwen3/config/qwen3" },
     "env": { "PYTHONPATH": "/root/icraft39" }
   }
   ```

4. **点火前三道门禁**（判据正本＝[`pipeline.md`](pipeline.md) §7，这里只列名字）：板上位流必须含 NPU（读 `/mnt/boot` 位流 md5，且要先 `mount -o ro`）、`CmaFree` 起跑值（CMA 是单一 256 MiB 池，与视频链同池，`udmabuf0` 开机就占走 128 MiB）、板子占用锁（同项目那台主机桌面上的 `113.txt` 要等于 `off`）。

## 状态（诚实版）

- **PC 侧**：五段编译全过、出 `Q3I-decoder_ZG` 件；Host 后端逐级链与校机 HF 基准对齐，`ref/` 那三份是 25 token 链的留底。**判据复算 PASS**：`.raw` 实算 md5 ＝ `.json` 内 `params_md5` ＝ 清单记账值三者逐字相等（10-28）。
- **板侧（2026-10-14）**：三件（`.json`／`.raw`／`tokenizer.json`）SFTP 上板、两端 `md5sum` 逐字相同；就地四档预检 s1／s2／s3 全 `exit=0`，**s3 的 `apply` 实测 75.1 s** ⇒ 这颗 `_ZG` 件在 ZG330 上**完成过 NPU 编程**。
- **没做的**：**生成那一步从未执行**——所以上板前置里"件内数值在 NPU 上对不对"这条还没答。本仓那两支驱动都写好未跑（`patch/q3_infer_board.py` 从未上板）。
- **运行依赖已解**（2026-11-04 板上实测）：`import icraft`（`/root/icraft39`）通；`tokenizers` 升 0.20.3 后这颗 `tokenizer.json` 板上直接可读、ids 与本机逐字相同；npuserv 的 server 骨架在板上 import 通过、`/models` 应答正确。
- **`/load` 已点火、但生成退化**（2026-11-06 板上实测，详见下节）：`/load` 成功（apply 171.5 s）、`/generate` 能出 token，但**输出恒定为同一 token，未达可用**；根因定位到 decode 步 KV 回喂（非 pos_rule、非 tokenizer）。**此项已搁置，交 Qwen 模型线。**

## 板上首次运行与退化问题（2026-11-06，板 113）

首次经 npuserv 真跑这颗件（走 registry、两条软链已建）。三组都"能加载、能出 token"，但生成退化：

| 场景 | 加载 | 生成 | 结果 |
|---|---|---|---|
| ids=[9707,1879] max_new=8 decode:false | apply 171.5 s → ready | 13.7 s | ids 全是 `508`（8 个） |
| text="你好" max_new=16 pos_rule=vendor | apply 170.9 s → ready | 30.0 s | 16 个 `198`（`\n`）×16 |
| text="你好" max_new=16 pos_rule=uniform | apply 172.2 s → ready | 30.6 s | 16 个 `198`（**与 vendor 完全相同**） |

**判据与结论**：

1. **不是 tokenizer**：text 路径 prompt=9 token 正常编码、decode 也正常出文本（`"\n"×16`）。
2. **不是 pos_rule**：vendor 与 uniform 两臂逐 token 相同（都退化到 `198`）。
3. **是 decode 步 KV 回喂缺陷**：首 token 随 prompt 变（ids→508、text→198），其后恒定重复，说明解码步看不到已生成 token。对照代码：adapter vendor 臂**整块替换** KV，与 `patch/q3_infer_board.py:run_decode` 返回 `arrays[1:]` 整块替换一致；uniform 臂走 `kv[slot] = out[p]` **切片写回**；**两种结构完全不同却同错 ⇒ 缺陷在 KV feed/布局（`past_key_values.i.key/value` 的命名/形状/dtype，或 out 索引 `p` 的语义），不在更新策略**。

**板端单元级实验（2026-11-07，已做）**：用"只读 logits、不落地整块 KV"的低内存探针，在板上得到三条硬结论——

1. **图确实消费 past KV，位置也生效**：零 KV vs 人为把槽位 1 填 5.0（同 token/同位置）→ `max|Δlogits|=22.76`、argmax 变（508→13）；同 token 位置 2 vs 9 → `Δ=15.99`、argmax 变；`token_dynamic_fm` 第三元素 0→7 → `Δ=0.000`（**未被图使用**）。证据：`workenv/ha-npu-ignite-2026-11-06/board-q3_logits_probe.log`。
2. **零 KV 时 argmax 就是 508**（与真实 ids 退化输出同 token）⇒ 真实运行里**历史 KV 实际等价于空**，不是"KV 没被消费"。
3. **图把新 KV 写在 `out[pos]`（原生 1-based，slot 0 不用）**：prompt token 用 `tdf=[1,1,0]` 时新 KV 落在 `out[1]`（`out[0]` 全零）；decode token 在 pos=2 时落在 `out[2]`。证据：`board-q3_slot_probe.log`。

**根因判定**：驱动 prompt 段把 `out[i+1]` **下移一位**写进 `kv slot i`（0-based），而解码段按 `seq_pos=prompt_length+step-1`（1-based）续写 ⇒ **两段 KV 索引基准不一致**，历史 KV 被读到错误槽位/空槽位，等价空上下文。Qwen2 用同样的下移能出通顺中文，说明它对这 1 位错位不敏感；Qwen3 不敏感度不足，于是恒重复。**不是 tokenizer、不是 pos_rule、也不是"图不消费 KV"**。

**修复已落地并验收（2026-11-07，板 113）**：统一成**原生 1-based**——prompt 也按 `out[pos]→slot pos` 原样透传（不位移），解码 `pos = prompt_len + 1` 起步。改动三处：`patch/q3_infer_board.py`（`run_prompt` 不再下移、`generate` 用 `prompt_length + step`）、`board/npuserv_adapter.py`（新增 `pos_rule="native"` 并设为默认：prompt/decode 都 `tdf=[1,slot+1,0]`、写回 `out[p]` 同一槽）、板上 `models.json`（qwen3 的 defaults.pos_rule 改 native）。

**A/B 与端到端证据（同一次加载、同一 prompt）**：

| 测试 | old（shift/vendor） | fixed（native） |
|---|---|---|
| ids `[9707,1879]` max_new=16 | `508×16`（复现历史退化） | `! This is a simple example of a Python script that prints "Hello World" …`（32 token 连贯） |
| text「你好」max_new=24 | `198×24`（`"\n"` 复读） | `好的，用户发来消息“你好”，我需要先确认他们的意图。可能他们只是打招呼，或者` |
| text「用一句话说明什么是边缘计算。」max_new=24 | —（未跑） | `嗯，用户让我用一句话说明什么是边缘计算。首先，我需要确定边缘计算的定义。边缘` |

驱动独立跑（`q3_infer_board.py` 修后，prompt tokens=9）同样得到「好的，用户发来消息“你好”…」。板上速度约 **0.8–1.2 token/s**（开了 1 GB swapfile 后；swap 实测用量 ~85 MB）。

**跑通条件（板端 OS 改动）**：整段 generate 峰值 `anon-rss≈580 MB`，1 GB 内存无 swap 时会被 OOM 连杀；本轮开了 `/swapfile`（1 GB，`chmod 600`、`mkswap`、`swapon`，并写入 `/etc/fstab` 持久化）。回滚＝`swapoff /swapfile && rm /swapfile` 并删 fstab 那行。**npuserv 代码未改**（只改 registry 的 `defaults.pos_rule` 与模型仓 adapter/驱动）。

**证据**：`workenv/ha-npu-ignite-2026-11-06/board-q3_logits_probe.log`、`board-q3_slot_probe.log`、`board-native-prod_text.json`、`board-native-prod_ids.json`、`board-native-prod_cn.json`、`board-native-q3_fixed_run.log`；探针脚本 `workenv/migrate-real-preview/q3_*.sh`、`npuserv_probe_adapter.py`。

**状态**：`能加载、能出连贯中文（0.6B 正常水平）`；退化根因与修复均已板端验证。

### 质量与速度评测（2026-11-07，板 113，`pos_rule=native`，swap 在用）

| Q | prompt | max_new | tok/s | 输出（节选） |
|---|---|---|---|---|
| Q1 | 你好 | 32 | 0.93 | 好的，用户发来消息“你好”，我需要先确认他们的意图。可能他们只是打招呼，或者想进行某种互动。 |
| Q2 | 你是谁？ | 32 | 0.82 | 好的，用户问我是谁。我需要先确认用户的身份，然后给出一个自然的回答。 |
| Q3 | 用一句话说明什么是边缘计算。 | 32 | 0.79 | 嗯，用户让我用一句话说明什么是边缘计算。首先，我需要确定边缘计算的定义。边缘计算通常指的是将数据处理和计算… |
| Q4 | 翻译：今天会议推迟到下午三点。 | 32 | 0.77 | 好的，用户让我翻译这句话：“今天会议推迟到下午三点。”首先，我需要确认这句话的准确意思。… |
| Q5 | 1+1等于几？ | 16 | 0.63 | 嗯，用户问的是1加1等于几。这个问题看起来挺 |
| Q6 | 写一句关于秋天的诗。 | 32 | 0.80 | 好的，用户让我写一句关于秋天的诗。首先，我需要确定秋天的主要元素。秋天通常与落叶、凉风、丰收有关… |

**读法（重要边界）**：六问**全部连贯、无复读、无退化**，但**全部是"思考式前缀"**（"好的，用户…我需要先确认…"）——Qwen3-0.6B 先出 reasoning、再给正式答案；32 token 预算下正式答案还没开始。实际产品要 **max_new ≥ 128**，或按 Qwen3 chat template 关/隔离开 thinking，不能拿 32 token 的输出当"答非所问"。翻译 Q4 同样停在"复述原句"阶段，未进入英文输出。

**速度**：0.63–0.93 tok/s（本评测在 swap 打开、`swap_used≈85MB` 状态下跑）；对比无 swap 时 Qwen3 8 token / 13.7 s ≈ **1.7 tok/s** ⇒ **swap 在用会让生成慢约一倍**。若要在板上提速，优先做"窄回读"（当前每步整块回读 58 MB KV）而不是靠 swap；swap 只该当 OOM 缓冲。证据：`workenv/ha-npu-ignite-2026-11-06/board-eval-{1..6}.json`。

### 窄回读优化（2026-11-07，已落地）

- **问题**：驱动 `_forward` 每步 `Tensor2Numpy(outputs)` 读 57 个输出（logits + 56×1 MiB KV = **58 MiB/步**），但每步其实只有**一个槽**是新的。
- **实现**：新增 `patch/kv_io.py`——按字节偏移只读槽位（每 head 256 B × 8 = **2 KiB/张量**，56 张量 ≈ **112 KiB/步**）；`q3_infer_board.py` 增 `_forward_narrow`（只读 logits + 新槽），`run_prompt`/`run_decode` 改走窄路径；`board/npuserv_adapter.py` 的 `_step` 同步窄回读。本地单测 `board/test_kv_readback.py` **11/11**：offset 0 与非零（777）、pos 0/1/7/511 的窄读都与整块读逐字节一致。
- **实测**（板 113，native，swap 打开）：32 token / 30.8 s = **1.04 tok/s**（整块回读同条件 0.93 tok/s，约 +12%）。
- **边界（实测教训）**：窄回读**没有**消掉 swap 的必要性——`swapoff` 后重新 load+generate 仍被 OOM kill（`worker closed the connection mid-request`），且板子随后进入约 **15 分钟内存抖动**（sshd 无响应、ping 正常，后自行恢复）。**板端 1 GB 内存跑 Qwen3 仍需 1 GB swap 当 OOM 缓冲**；要再提速应减少 device 侧 56×1 MiB 输出分配（驱动/图层面），不是继续抠 host 回读。
- 证据：`workenv/ha-npu-ignite-2026-11-06/board-narrow-n_gen_on.json`、`board-narrow-n_load_{on,off}.json`、`board-narrow-server.log`。

## 三条已知坑

1. **件名不一致**：板上叫 `Q3V1-decoder_ZG.*`、库里叫 `Q3I-decoder_ZG.*`。10-28 逐字节核过：`.raw` 两边同为 `1081ddd7…`；`.json` 把板上那颗里唯一一处 `Q3V1` 换成 `Q3I` 后与库里这颗**逐字节相同** ⇒ 同一张图只差名字串。**按 `Q3I` 这名重推板＝把正本 raw 复制改名，不用重编**（`_ZG.json` 内没有 `.raw` 路径字段，只有 `params_md5`／`params_bytes`）。
2. **`tokenizer.json` 有两颗、值不同**：库里留档那颗（`6423133b…`／11,422,654 B）与板上那颗（`7875a23b…`／11,418,266 B）**不是同一份**；板上那颗与厂商 Qwen2 导出目录里的、与本机 `compile/fmodel/T0/` 的那颗三者同 md5。要复现板上行为以**板上那颗**为准，别拿库里这颗对数。
3. **首步（位置 0）的成因与本线不同**：本项目 ERNIE 那条线实测过"槽 0 是全零 pad、图真会去 attend 它"，修法＝token 0 连喂两次＋每步把槽 1 复制进槽 0。本仓 `patch/` 那两支驱动里 `padfix` 这条**在 Qwen3 上从未板上验证**，写成待验不是写成结论。

## 相关（项目内部路径，外部读者可忽略）

`wiki/PL/MOD/llm/qwen3-0.6b.md`（操作记录与对照表）／`wiki/PL/MOD/llm/Qwen3-Instruct按Qwen3配方迁移.md`（SOP）／`wiki/PL/MOD/llm/Qwen3按厂商Qwen2配方迁移.md`（原理与证据分级）／`wiki/PS/板端我方代码与产物台账.md`（板端逐件登记，`#208`～`#213` 是本线上板那批）。


## K128：KV 深度重编译（2026-11-07）——2.13 tok/s（原 0.93）

`max_tokens_length` 512→128 重编译（pipeline.md §9 全流程），KV [1,8,128,128]，
每步 KV DMA 112MB→28MB。语义与校机 HF 参考 26/26 全等；板上 max_new=64 →
**2.13 tok/s**（原 1.04）。上下文窗口同步缩到 128（prompt+生成 ≤128 token，
语音单轮场景够用）。registry defaults 已指 `decode_k128` 件；回退＝删 defaults
的 `json/raw/max_decode_length`。

**瓶颈判定（实测定案）**：swap 与速度**无关**（512 件有 swap 1.04 / 无 swap 1.08、
0.90 tok/s）；K128 后每步 ~0.47s = KV DMA 28MB ≈0.15s + **NPU 计算 ~0.3s**——计算
成为新瓶颈，再提速需减层数/换量化，超出本件范围。LLM 常驻与视觉相机互斥（apply 吃
CMA，实测详见 `docs/COEXISTENCE.md`）。

**瓶颈判定（实测定案）**：swap 与速度**无关**（512 件有 swap 1.04 / 无 swap 1.08、
0.90 tok/s）；K128 后每步 ~0.47s = KV DMA 28MB ≈0.15s + **NPU 计算 ~0.3s**——计算
成为新瓶颈，再提速需减层数/换量化，超出本件范围。LLM 常驻与视觉相机互斥（apply 吃
CMA，实测详见 `docs/COEXISTENCE.md`）。
