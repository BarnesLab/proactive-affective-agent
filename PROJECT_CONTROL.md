# PROJECT_CONTROL.md — 跨平台任务控制协议

**所有 agent（OpenClaw/Boo、Claude Code、Codex）在启动或推进实验任务前，必须先读这个文件。**

## 控制文件位置

```
~/.openclaw/workspace/project-control.json
```

## 规则

### 启动任务前必须检查：

```python
import json
with open(os.path.expanduser('~/.openclaw/workspace/project-control.json')) as f:
    ctrl = json.load(f)

# 1. 全局暂停？
if ctrl['global_pause']:
    print("全局暂停中，不启动任何任务")
    exit()

# 2. 模型级暂停？
if ctrl['model_pause'].get('claude') and 你要跑的是_claude/sonnet:
    print("Claude 模型暂停中，跳过")
if ctrl['model_pause'].get('openai') and 你要跑的是_openai/gpt:
    print("OpenAI 模型暂停中，跳过")

# 3. 项目级暂停？
project = ctrl['projects'].get('proactive-affective-agent', {})
if project.get('paused'):
    print("项目暂停中，不启动")
```

### 暂停时的行为：
- **不启动新任务**
- **已在跑的任务让它自然结束**（不主动 kill）
- **不要自动恢复**，等用户说"继续"

### 用户怎么触发：
用户会通过 Telegram 对 Boo（OpenClaw）说，Boo 会更新 `project-control.json`。
也可以直接编辑该文件。

### 常见指令：
- "暂停所有" → `global_pause: true`
- "暂停 Claude 实验" → `model_pause.claude: true`
- "暂停 OpenAI 实验" → `model_pause.openai: true`
- "继续" → 恢复对应的 pause flag

## 为什么

实验跑 LLM API 消耗 token 额度，用户需要随时让出额度给正常工作。
所有 agent 必须尊重这个控制文件，否则会抢占用户的工作额度。
