# Microsoft Azure OpenAI 配置指南

## 1. 获取 Azure OpenAI 配置信息

### 从 Azure 门户获取：
1. 登录 [Azure 门户](https://portal.azure.com)
2. 找到你的 Azure OpenAI 资源
3. 记录以下信息：
   - **资源名称** (Resource Name)
   - **API 密钥** (API Key)
   - **部署名称** (Deployment Name)

## 2. 环境变量配置

在 `.env` 文件中添加：

```bash
# Microsoft Azure OpenAI API 配置
OPENAI_API_KEY=your_azure_openai_api_key_here
OPENAI_BASE_PATH=https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name
```

### 示例：
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_BASE_PATH=https://my-openai-resource.openai.azure.com/openai/deployments/gpt-4-deployment
```

## 3. 代码配置

### 对于 Assistant API：
在 `app/api/v1/endpoints/ai_service.py` 中，将模型名称改为你的部署名称：

```python
assistant = client.beta.assistants.create(
    name="English Learning Assistant",
    instructions="You are a helpful English learning assistant...",
    model="your-deployment-name"  # 使用你的部署名称
)
```

### 对于 Chat Completions：
```python
response = client.chat.completions.create(
    model="your-deployment-name",  # 使用你的部署名称
    messages=[{"role": "user", "content": "Hello"}]
)
```

### 对于 TTS (Text-to-Speech)：
```python
response = client.audio.speech.create(
    model="tts-1",  # Azure OpenAI 支持 tts-1
    voice="alloy",
    input="Hello world"
)
```

### 对于 Whisper (Speech-to-Text)：
```python
response = client.audio.transcriptions.create(
    model="whisper-1",  # Azure OpenAI 支持 whisper-1
    file=audio_file
)
```

## 4. 支持的模型

Azure OpenAI 支持以下模型：
- **GPT-4**: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- **GPT-3.5**: `gpt-35-turbo`
- **TTS**: `tts-1`
- **Whisper**: `whisper-1`

## 5. 验证配置

启动服务后，检查日志确认连接成功：
```
OpenAI Base Path: https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name
OpenAI API Key: sk-xxxxxxxx...
```

## 6. 常见问题

### Q: 出现 401 错误？
A: 检查 API 密钥是否正确

### Q: 出现 404 错误？
A: 检查 base URL 和部署名称是否正确

### Q: 出现模型不支持错误？
A: 确认你的 Azure OpenAI 资源已部署了相应的模型

## 7. 优势

使用 Microsoft Azure OpenAI 的优势：
- **数据隐私**: 数据不会离开 Azure 区域
- **企业级安全**: 符合企业安全要求
- **成本控制**: 更灵活的定价和配额管理
- **集成**: 与 Azure 其他服务无缝集成 