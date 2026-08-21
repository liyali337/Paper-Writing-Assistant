# Asta MCP 接入指南（Cursor / Codex / 同类应用）

本文说明如何把 **Ai2 Asta Scientific Corpus Tool**（官方学术 MCP，数据来自 Semantic Scholar）接到 Cursor、Codex 以及其它支持 MCP 的应用上，使 Agent 能检索论文、查看引用与作者信息。

官方文档：[Asta Scientific Corpus Tool](https://allenai.org/asta/resources/mcp)

---

## 1. 它是什么

Asta 是 Allen Institute for AI 提供的**远程 MCP 服务**，把 Semantic Scholar 学术库暴露成标准工具。应用作为 MCP 客户端连接即可，**本机不用安装学术数据库**。

| 项目 | 值 |
|---|---|
| 服务地址 | `https://asta-tools.allen.ai/mcp/v1` |
| 传输协议 | Streamable HTTP |
| 鉴权头 | `x-api-key: <你的 Key>` |
| 费用 | 免费；Key 用于提高速率上限 |

**能做：** 按关键词/标题搜论文、用 DOI / arXiv ID 取详情、追引用、查作者、检索正文片段。

**不能做：** 不保证每篇论文当天必到；不会主动推送「今日新论文」（需要你或系统去查询）；默认给元数据和片段，不是整篇 PDF 下载器。

---

## 2. 申请 Key

1. 打开 [Request an Asta tool key](https://share.hsforms.com/1L4hUh20oT3mu8iXJQMV77w3ioxm)
2. 提交申请，拿到 Key
3. Key 只放在本机配置或本机环境变量里，**不要提交到 git，不要发到聊天/截图**

推荐环境变量名：`ASTA_API_KEY`

### 2.1 写入本机环境变量（zsh / macOS）

```bash
echo 'export ASTA_API_KEY="你的key"' >> ~/.zshrc
source ~/.zshrc
```

验证：

```bash
echo "ASTA_API_KEY is set: ${ASTA_API_KEY:+yes}"
```

**macOS 注意：** 从程序坞 / Spotlight 打开的 Cursor、Codex 桌面应用，经常读不到 `~/.zshrc`。可选处理：

- 当前会话让 GUI 应用也能读到：

  ```bash
  launchctl setenv ASTA_API_KEY "$ASTA_API_KEY"
  ```

  然后 **彻底退出应用再打开**（Cursor 用 `Cmd + Q`，不要只关窗口）。

- 或者：在该应用的**本机全局配置**里直接填写 Key（见下文各产品配置）。全局配置在用户主目录，不进项目仓库。

---

## 3. 通用接入要点

所有 MCP 宿主（Cursor、Codex、Claude Code、Windsurf 等）都是同一套远程服务，差别只是配置文件格式：

1. 填 URL：`https://asta-tools.allen.ai/mcp/v1`
2. 请求头带 `x-api-key`
3. 不要用 `command` / `npx` 去「安装」Asta（它不是本地进程）
4. 配置后重启宿主；在 **Agent 模式**下使用（纯问答模式往往不会调 MCP）

字段名容易写错：

| 产品 | URL 字段 | Header 写法 |
|---|---|---|
| Cursor / Claude Desktop 一类 JSON | `url` | `"headers": { "x-api-key": "..." }` |
| Windsurf 官方示例 | `serverUrl` | 同上 |
| Codex | `url` | `http_headers` 或 `env_http_headers` |
| Claude Code CLI | `--url` | `-H "x-api-key: ..."` |

---

## 4. Cursor

### 4.1 配置文件位置

| 范围 | 路径 | 说明 |
|---|---|---|
| 全局（推荐，所有项目可用） | `~/.cursor/mcp.json` | macOS 即 `/Users/<你>/.cursor/mcp.json` |
| 仅当前项目 | 项目根目录 `.cursor/mcp.json` | 不要把真实 Key 提交进 git |

两个文件会合并；同名 server 以**项目级**为准。

### 4.2 推荐配置（Key 直接写在本机全局文件）

适合不想处理 macOS 环境变量的情况：

```json
{
  "mcpServers": {
    "asta": {
      "url": "https://asta-tools.allen.ai/mcp/v1",
      "headers": {
        "x-api-key": "这里粘贴你的真实key"
      }
    }
  }
}
```

Key 两边不要多空格，不要加 `Bearer`。

### 4.3 用环境变量插值

若已确保 Cursor 进程能读到 `ASTA_API_KEY`：

```json
{
  "mcpServers": {
    "asta": {
      "url": "https://asta-tools.allen.ai/mcp/v1",
      "headers": {
        "x-api-key": "${env:ASTA_API_KEY}"
      }
    }
  }
}
```

`${env:ASTA_API_KEY}` 是 Cursor 的插值语法；也可以直接换成真实 Key（见 4.2）。

**不要**把 Ai2 官网 Windsurf 示例里的 `serverUrl` 原样贴进 Cursor，Cursor 要用 `url`。

### 4.4 在界面里打开 MCP

新版 Cursor **没有**固定的「Tools & MCP」菜单名。按下面任一方式：

1. 最左侧边栏打开 **Customize（自定义）** → **MCPs**
2. `Cmd + Shift + P`，搜索 `MCP`，选 `View: Open MCP Settings` 或类似项
3. `Cmd + ,` 打开设置后，用**顶部搜索框**搜 `MCP`（不要在 VS Code 那种 Editor 设置里找）

保存配置后 **`Cmd + Q` 完全退出再打开**。在 Customize → MCPs 中确认 `asta` 已启用。

### 4.5 使用

切到 **Agent** 模式，例如：

- 「用 Asta 搜 2024 年以后的 diffusion policy 论文，只要标题、年份、摘要」
- 「查这篇的引用：ARXIV:2303.08774」
- 「按标题找 Attention Is All You Need」

第一次调用会弹出工具确认，允许即可。

---

## 5. Codex（CLI / IDE / ChatGPT 桌面端共用）

Codex 把 MCP 写在 TOML 里。ChatGPT 桌面应用、Codex CLI、IDE 扩展通常共用同一份配置。

| 范围 | 路径 |
|---|---|
| 全局 | `~/.codex/config.toml` |
| 项目 | `.codex/config.toml`（仅受信任项目） |

官方说明：[Codex MCP](https://developers.openai.com/codex/mcp)

### 5.1 推荐：Key 从环境变量读取

```toml
[mcp_servers.asta]
url = "https://asta-tools.allen.ai/mcp/v1"
enabled = true
env_http_headers = { "x-api-key" = "ASTA_API_KEY" }
```

`env_http_headers` 的值是**环境变量名**，不是 Key 本身。先保证 shell 里有 `export ASTA_API_KEY=...`。

### 5.2 直接写死 Header（仅本机、不要提交）

```toml
[mcp_servers.asta]
url = "https://asta-tools.allen.ai/mcp/v1"
enabled = true
http_headers = { "x-api-key" = "你的key" }
```

### 5.3 使用

重启 Codex / 新开一轮对话，在 Agent 里同样用自然语言检索。可用 `codex mcp` 相关命令查看已配置的 server（以本机 `codex mcp --help` 为准）。

---

## 6. 其它同类应用

### 6.1 Claude Code

```bash
claude mcp add -t http -s user asta https://asta-tools.allen.ai/mcp/v1 \
  -H "x-api-key: $ASTA_API_KEY"
```

`-s user` 表示用户级，所有项目可用。

### 6.2 Windsurf

Ai2 官网示例使用 `serverUrl`：

```json
{
  "mcpServers": {
    "asta": {
      "serverUrl": "https://asta-tools.allen.ai/mcp/v1",
      "headers": {
        "x-api-key": "<YOUR_API_KEY>"
      }
    }
  }
}
```

若当前 Windsurf 版本已改用 `url`，与 Cursor 相同即可。

### 6.3 Claude Desktop / 通用 JSON MCP 客户端

与 Cursor 相同，使用 `url` + `headers`：

```json
{
  "mcpServers": {
    "asta": {
      "url": "https://asta-tools.allen.ai/mcp/v1",
      "headers": {
        "x-api-key": "<YOUR_API_KEY>"
      }
    }
  }
}
```

配置文件路径因产品而异（例如 Claude Desktop 在 macOS 上为 `~/Library/Application Support/Claude/claude_desktop_config.json`）。原则不变：**远程 URL + `x-api-key`**。

### 6.4 自研应用

自研后端可以：

1. 当 MCP Client，用官方 MCP SDK 连同一地址；或
2. 不走 MCP，直接调 [Semantic Scholar Graph API](https://www.semanticscholar.org/product/api)

定时「每日拉新论文」更适合走后端任务，而不是依赖 IDE 聊天。Cursor / Codex 适合人在编辑器里临时检索。

---

## 7. 可用工具一览

连接成功后，Agent 应能看到类似工具（名称以服务端为准）：

| 工具 | 作用 |
|---|---|
| `search_papers_by_relevance` | 按关键词检索，可过滤年份、会议/期刊 |
| `search_paper_by_title` | 按标题查找 |
| `get_paper` | 用 DOI、arXiv、PMID 等取一篇详情 |
| `get_citations` | 查看引用该论文的文献 |
| `search_authors_by_name` | 按姓名搜作者 |
| `get_author_papers` | 列出某作者论文 |
| `snippet_search` | 从论文正文检索约 500 词片段 |

`get_paper` 支持的 ID 示例：`DOI:10.18653/v1/N18-3011`、`ARXIV:2106.15928`、`PMID:19872477`，以及 semanticscholar.org / arxiv.org 等 URL。

可选：安装社区 skill [asta-skill](https://github.com/Agents365-ai/asta-skill) 帮助模型选对工具。真正发请求的仍是上述 MCP 服务。

---

## 8. 验证与排障

### 8.1 验证

1. 宿主里 Asta 显示为已连接 / 已启用
2. Agent 模式能列出上述工具
3. 试一句：「用 Asta 搜 3 篇 transformer 论文，返回标题和年份」

### 8.2 Cursor 日志

1. `Cmd + Shift + U` 打开 Output
2. 下拉选择 **MCP Logs**

### 8.3 常见问题

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 找不到 Tools & MCP | 菜单已改名 | 用 Customize → MCPs，或命令面板搜 MCP |
| `${env:ASTA_API_KEY}` 无效 | GUI 应用读不到 `.zshrc` | 改成本机全局配置里直写 Key，或 `launchctl setenv` 后重启 |
| 401 / 鉴权失败 | Key 错、多余空格、写成了 Bearer | 检查 `x-api-key` 原样粘贴 |
| 连得上但不调工具 | 当前是 Ask / 普通聊天 | 切换到 Agent |
| 429 / 限流 | 无 Key 或请求过密 | 确认已带 Key；减少并发、缩小 `limit` |
| Cursor 用了 `serverUrl` 连不上 | 字段名不匹配 | 改为 `url` |
| 项目里配置了但 git 泄露风险 | Key 写在仓库内 mcp.json | 改为 `~/.cursor/mcp.json` 或环境变量 |

---

## 9. 安全约定

- Key 只放：本机全局配置、本机环境变量、本机 LaunchAgent。
- 项目内的 `.cursor/mcp.json` / `.codex/config.toml` 若要提交，只保留 URL 与 `${env:...}` / `env_http_headers`，不要写真实 Key。
- 不要把 Asta 当全文盗链或绕过出版社付费墙的工具；开放获取论文再走 arXiv / 出版社 OA 链接。

---

## 10. 配置速查

```text
服务:  https://asta-tools.allen.ai/mcp/v1
Header: x-api-key: <KEY>
申请:  https://share.hsforms.com/1L4hUh20oT3mu8iXJQMV77w3ioxm

Cursor:  ~/.cursor/mcp.json          字段 url + headers
Codex:   ~/.codex/config.toml        [mcp_servers.asta] + env_http_headers
Claude:  claude mcp add -t http ...
```
