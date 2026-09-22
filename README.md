# 论文助手

上传学术 PDF，按章节对照阅读，并针对这一篇追问。回答里的引用带页码，点一下即可跳回原文。

当前版本包括解析、翻译、混合检索和问答。问答可以针对本篇精读，也可以检索本地库或 arXiv。

## 功能

- **章节树**：用 Docling 做版面解析，失败时回退 PyMuPDF。按标题拆成节，抽出插图，同一文件按 SHA-256 去重。
- **对照阅读**：左侧目录、章节正文与原 PDF，中英文可切换。翻译按节切块，公式、表格和插图标记尽量保留。
- **公式重识别**：对裁出的公式图做视觉还原，写成 LaTeX。失败不影响论文进入可阅读状态。
- **带引用提问**：一个输入框。精读本篇时走检索并标出节名与页码；也可以查本地已入库论文，或检索 arXiv。
- **混合检索**：章节作为父块，再切成子块。稠密向量与 BM25 稀疏向量在本地 Qdrant 里用 RRF 融合。未安装检索依赖时退化为词法检索。

扫描件不跑 OCR。抽不出足够可复制文本时，论文标为需要可复制文本，并且不会翻译，以免编造内容。

## 技术栈

| 部分 | 选型 |
|---|---|
| API | Python 3.11+、FastAPI、Uvicorn |
| PDF | Docling（主路径）、PyMuPDF（回退与抽图） |
| 模型 | OpenAI 兼容接口，默认 DeepSeek |
| 检索 | sentence-transformers、FastEmbed BM25、本地 Qdrant |
| 问答 | LangGraph（`ASK_MODE=agent`）；未安装时回退为一次检索加一次生成 |
| 前端 | React 19、TypeScript、Vite、pdf.js、KaTeX |
| 观测 | Langfuse（可选） |

数据落在本地 `data/papers/{id}/`（PDF、章节、译文、切块、插图）。`data/`、`.env` 和 PDF 已在 `.gitignore` 中，不要提交。

## 环境要求

- Python 3.11 或更高
- Node.js 18 或更高
- 一个 OpenAI 兼容的 API Key（翻译、公式重识别、问答）
- 首次建索引会下载嵌入模型；访问不了 Hugging Face 时可设置 `HF_ENDPOINT=https://hf-mirror.com`

## 快速开始

在仓库根目录：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[parse,rag,agent,dev]"
copy .env.example .env
npm install
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[parse,rag,agent,dev]"
cp .env.example .env
npm install
```

编辑 `.env`，至少填入 `OPENAI_API_KEY`。然后开两个终端。

API（Windows 可直接用仓库脚本；其它系统请在已激活的虚拟环境里跑 uvicorn）：

```powershell
npm run dev:api
```

```bash
python -m uvicorn dl_agent.api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload --reload-dir src
```

前端：

```bash
npm run dev:web
```

- 界面：<http://localhost:5173>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

Vite 把 `/api` 代理到 FastAPI，并去掉前缀。浏览器只访问 5173 即可。

可选依赖可以分开装：`pip install -e ".[parse]"` 只装解析，`.[rag]` 只装检索，`.[agent]` 只装 LangGraph，`.[obs]` 只装 Langfuse。

## 配置

完整模板见 [`.env.example`](.env.example)。常用项：

| 变量 | 作用 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` | 翻译与问答。默认 `https://api.deepseek.com/v1` 与 `deepseek-chat` |
| `FORMULA_VISION_ENABLED` / `FORMULA_VISION_MODEL` | 公式看图还原 LaTeX，默认开启，模型为 `deepseek-flash` |
| `DATA_DIR` | 论文与索引目录，默认 `./data` |
| `ASK_MODE` | `agent`（默认）或 `simple` |
| `ASK_ENABLE_EXTERNAL` | 是否让精读流程额外调用 arXiv，默认关闭。提问里明确要求检索 arXiv 时仍会走独立检索 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | 有公钥和私钥才上报；未安装 SDK 时自动跳过 |

单篇 PDF 上限 50MB。

## 目录

```text
frontend/                 React 工作区（目录、对照阅读、PDF、提问）
src/dl_agent/
  api/                    FastAPI 路由
  knowledge/              解析、切块、混合检索
  translate/              按节翻译
  understand/             问答；agent/ 为 LangGraph
  mcp_gateway/            arXiv Atom，超时回落 OpenAlex
  harness/                模型调用
  observability/          Langfuse
tests/                    pytest
docs/                     实现说明（见下方）
deploy/docker-compose.yml PostgreSQL 与 Redis，当前应用尚未接入
```

## 主要接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/papers` | 上传 PDF，后台解析并建索引 |
| `GET` | `/papers` | 已上传论文 |
| `GET` | `/papers/{id}/sections` | 章节树 |
| `GET` | `/papers/{id}/figures` | 插图 |
| `GET` | `/papers/{id}/source` | 原 PDF |
| `POST` | `/papers/{id}/translations` | 按需翻译 |
| `POST` | `/papers/{id}/ask` | 针对当前论文提问 |
| `POST` | `/library/ask` | 本地库 |
| `POST` | `/arxiv/ask` | arXiv |

## 测试

```bash
pytest
```

前端纯函数测试使用 Node 内置测试运行器：

```bash
node --experimental-strip-types --test frontend/src/lib/*.test.ts
```

## 已知范围

- 不识别扫描件。
- 参考文献之后的附录默认不进入翻译。
- 引用只跳到 PDF 页，不在页面上高亮框。
- 问答一次返回整段回答，不流式推送工具调用过程。
- 精读只查当前这篇；跨篇走本地库，外部论文走 arXiv，三者的引用不会混用。
- 向量库是本机 Qdrant 文件，没有多用户与云端集合。

更细的设计与边界见 [docs/L1论文解析与翻译.md](docs/L1论文解析与翻译.md) 和 [docs/L2论文检索回答模块.md](docs/L2论文检索回答模块.md)。

## 许可证

仓库尚未选择开源许可证。上传 GitHub 前请补上 `LICENSE`，否则他人默认不能复用这些代码。
