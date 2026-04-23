# 课程资料 RAG 系统

一个基于检索增强生成（RAG）的系统，通过语义搜索和 AI 生成来回答有关课程材料的问题。

## 概述

这是一个全栈 Web 应用，使用户能够查询课程资料并获得智能的上下文感知回答。系统使用 ChromaDB 进行向量存储，MiniMax API 进行 AI 生成，并提供 Web 界面进行交互。

## 前置要求

- Python 3.13 或更高版本
- uv（Python 包管理器）
- MiniMax API 密钥
- **Windows 用户**：使用 Git Bash 运行应用命令 - [下载 Git for Windows](https://git-scm.com/downloads/win)

## 安装

1. **安装 uv**（如未安装）
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **安装 Python 依赖**
   ```bash
   uv sync
   ```

3. **设置环境变量**

   在项目根目录创建 `.env` 文件：
   ```bash
   ANTHROPIC_API_KEY=your_minimax_api_key_here
   ```

## 运行应用

### 快速启动

使用提供的 shell 脚本：
```bash
chmod +x run.sh
./run.sh
```

### 手动启动

```bash
cd backend
$env:PATH += ";C:\Users\86787\.local\bin" 
uv run uvicorn app:app --reload --port 8000
```

应用启动后可访问：
- Web 界面：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

## 查询流程

用户查询经过以下层级：

```
前端 (HTML/JS) → FastAPI 后端 → RAGSystem → AIGenerator → ToolManager → VectorStore → ChromaDB
```

### 第一步：前端提交
**文件：** `frontend/script.js:45-96`

用户输入问题并点击发送（或按回车）。调用 `sendMessage()` 函数：
1. **第 46 行**：`const query = chatInput.value.trim()` - 获取输入框中的查询内容
2. **第 63-72 行**：发送 POST 请求到 `/api/query`，包含 `{query, session_id}`
3. **第 76 行**：`const data = await response.json()` - 接收响应
4. **第 85 行**：`addMessage(data.answer, 'assistant', data.sources)` - 使用 markdown 渲染显示回答和来源

### 第二步：API 端点
**文件：** `backend/app.py:56-74`

`POST /api/query` 端点：
1. **第 62-63 行**：如果未提供 `session_id`，则调用 `rag_system.session_manager.create_session()` 创建新会话
2. **第 66 行**：调用 `rag_system.query(request.query, session_id)`
3. **第 68-72 行**：返回 `QueryResponse(answer, sources, session_id)`

### 第三步：RAG 系统编排
**文件：** `backend/rag_system.py:102-140`

调用 `RAGSystem.query()` 方法（第 102 行）：
1. **第 114 行**：构建提示词 `prompt = f"""Answer this question about course materials: {query}"""`
2. **第 118-119 行**：调用 `self.session_manager.get_conversation_history(session_id)` 获取会话历史
3. **第 122-127 行**：调用 `self.ai_generator.generate_response()` 并传入工具
4. **第 130 行**：调用 `self.tool_manager.get_last_sources()` 提取来源
5. **第 137 行**：调用 `self.session_manager.add_exchange(session_id, query, response)` 更新会话历史
6. **第 140 行**：返回 `(response, sources)`

### 第四步：AI 生成与工具执行
**文件：** `backend/ai_generator.py:42-133`

调用 `AIGenerator.generate_response()` 方法（第 42 行）：

**第一次 API 调用：**
- **第 77 行**：`self.client.chat.completions.create(**api_params)` - 发送请求到 MiniMax API
- **第 79-80 行**：如果 `response.choices[0].finish_reason == "tool_calls"`，调用 `_handle_tool_execution()`

**工具执行路径 `_handle_tool_execution()`（第 84-133 行）：**
- **第 116-120 行**：遍历 `tool_calls`，调用 `tool_manager.execute_tool(tool_call.function.name, **eval(tool_call.function.arguments))`
- **第 132 行**：执行第二次 API 调用 `self.client.chat.completions.create(**final_params)`
- **第 133 行**：返回 Claude 综合后的回答

### 第五步：工具管理器执行
**文件：** `backend/search_tools.py:138-143`

调用 `ToolManager.execute_tool()` 方法（第 138 行）：
1. **第 140-141 行**：检查工具是否存在
2. **第 143 行**：调用 `self.tools[tool_name].execute(**kwargs)` 执行 `CourseSearchTool.execute()`

### 第六步：课程搜索工具
**文件：** `backend/search_tools.py:55-89`

调用 `CourseSearchTool.execute()` 方法（第 55 行）：
1. **第 69-73 行**：调用 `self.store.search(query, course_name, lesson_number)`
2. **第 91-117 行**：调用 `_format_results()` 用课程/课时上下文标题格式化结果
3. **第 115 行**：将来源存储在 `self.last_sources = sources` 供后续检索
4. **第 117 行**：返回格式化字符串给 AI Generator

### 第七步：向量存储搜索
**文件：** `backend/vector_store.py:61-100`

调用 `VectorStore.search()` 方法（第 61 行）：
1. **第 80-83 行**：调用 `_resolve_course_name()` 通过 `course_catalog` collection 解析课程名称
2. **第 86 行**：调用 `_build_filter()` 为 `course_content` collection 构建过滤字典
3. **第 93-97 行**：调用 `self.course_content.query()` 在 `course_content` 上执行相似性搜索
4. **第 98 行**：返回 `SearchResults.from_chroma(results)`

### 第八步：ChromaDB 检索
**Collections：**
- `course_catalog`：存储课程元数据（标题、讲师、课时）—— 位于 `backend/vector_store.py` 第 51 行创建
- `course_content`：存储分块的课程内容，包含 course_title、lesson_number、chunk_index 元数据 —— 位于第 52 行创建

### 关键实现细节

- **每次课程查询两次 API 调用**：`backend/ai_generator.py` 第 77 行和第 132 行
- **基于工具的 RAG**：Claude 自主决定何时调用 `search_course_content`（`finish_reason == "tool_calls"`）
- **会话历史**：`MAX_HISTORY=2` 保留最近 2 轮对话用于上下文
- **嵌入模型**：`all-MiniLM-L6-v2`，块大小 800 字符，重叠 100 字符
