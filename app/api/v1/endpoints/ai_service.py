from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Any, Literal, Optional
import time
import traceback
import json
import os
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
import logging
import redis
import itertools
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.load import dumpd, load
import requests

from app.core.config import settings
from app.services.r2_service import r2_service # Import the R2 service instance

router = APIRouter()

# Debug: Print environment variables
print(f"=== Environment Variables Debug ===")
print(f"OPENAI_API_KEY: {settings.OPENAI_API_KEY[:10]}..." if settings.OPENAI_API_KEY else "OPENAI_API_KEY: None")
print(f"OPENAI_BASE_PATH: {settings.OPENAI_BASE_PATH}")
print(f"OPENAI_MODEL_NAME: {settings.OPENAI_MODEL_NAME}")
print(f"CLOUDFLARE_R2_ACCOUNT_ID: {settings.CLOUDFLARE_R2_ACCOUNT_ID}")
print(f"CLOUDFLARE_R2_ACCESS_KEY_ID: {settings.CLOUDFLARE_R2_ACCESS_KEY_ID}")
print(f"CLOUDFLARE_R2_SECRET_ACCESS_KEY: {settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY[:10]}..." if settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY else "CLOUDFLARE_R2_SECRET_ACCESS_KEY: None")
print(f"CLOUDFLARE_R2_BUCKET_NAME: {settings.CLOUDFLARE_R2_BUCKET_NAME}")
print(f"R2_PUBLIC_DOMAIN: {settings.R2_PUBLIC_DOMAIN}")
print(f"=== Search Configuration ===")
print(f"ENABLE_DUCKDUCKGO_SEARCH: {settings.ENABLE_DUCKDUCKGO_SEARCH}")
print(f"ENABLE_GOOGLE_CUSTOM_SEARCH: {settings.ENABLE_GOOGLE_CUSTOM_SEARCH}")
print(f"ENABLE_SERPAPI_FALLBACK: {settings.ENABLE_SERPAPI_FALLBACK}")
print(f"GOOGLE_CUSTOM_SEARCH_API_KEY: {'Configured' if settings.GOOGLE_CUSTOM_SEARCH_API_KEY else 'Not configured'}")
print(f"GOOGLE_CUSTOM_SEARCH_ENGINE_ID: {'Configured' if settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID else 'Not configured'}")
print(f"SERPAPI_API_KEY: {'Configured' if settings.SERPAPI_API_KEY else 'Not configured'}")
print(f"DUCKDUCKGO_TIMEOUT: {settings.DUCKDUCKGO_TIMEOUT}s")
print(f"===================================")

# 检查必需的环境变量
if not settings.OPENAI_MODEL_NAME:
    raise ValueError("OPENAI_MODEL_NAME environment variable is required but not set. Please set it in your .env file or environment.")

# Initialize the OpenAI client
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_PATH,
)

# 本地文件路径，用于保存 assistant 和 thread 的 ID
DEV_IDS_FILE = "dev_assistant_ids.json"

# Redis 连接（可根据实际情况配置 host/port/db）
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
REDIS_CHAT_KEY = 'chat_history:default'  # 单用户 demo，生产可用 user_id
REDIS_CHAT_ID_KEY = 'chat_history:default:next_id'
MAX_HISTORY = 20  # 只保留最近 20 条历史消息

def load_dev_ids():
    """从本地文件加载 assistant 和 thread 的 ID"""
    if os.path.exists(DEV_IDS_FILE):
        try:
            with open(DEV_IDS_FILE, 'r') as f:
                data = json.load(f)
                return data.get('assistant_id'), data.get('thread_id')
        except Exception as e:
            print(f"Error loading dev IDs: {e}")
    return None, None

def save_dev_ids(assistant_id: str, thread_id: str):
    """保存 assistant 和 thread 的 ID 到本地文件"""
    try:
        data = {
            'assistant_id': assistant_id,
            'thread_id': thread_id
        }
        with open(DEV_IDS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved dev IDs to {DEV_IDS_FILE}")
    except Exception as e:
        print(f"Error saving dev IDs: {e}")

# 全局变量来存储 assistant 和 thread
# 开发阶段：从本地文件加载或创建新的 assistant 和 thread
assistant_id, thread_id = load_dev_ids()

def create_search_tool():
    """
    创建搜索工具，三层级降级：DuckDuckGo -> Google Custom Search -> SerpAPI
    """
    # 初始化搜索服务状态跟踪
    if not hasattr(create_search_tool, 'search_stats'):
        create_search_tool.search_stats = {
            'duckduckgo_failures': 0,
            'google_custom_search_failures': 0,
            'serpapi_failures': 0,
            'last_duckduckgo_failure': 0,
            'last_google_custom_search_failure': 0,
            'last_serpapi_failure': 0,
            'duckduckgo_successes': 0,
            'google_custom_search_successes': 0,
            'serpapi_successes': 0
        }
    
    def google_custom_search(query: str) -> str:
        """
        使用 Google Custom Search API 进行搜索
        """
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': settings.GOOGLE_CUSTOM_SEARCH_API_KEY,
                'cx': settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID,
                'q': query,
                'num': 5  # 返回前5个结果
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data and data['items']:
                # 提取搜索结果
                results = []
                for item in data['items']:
                    title = item.get('title', '')
                    snippet = item.get('snippet', '')
                    link = item.get('link', '')
                    results.append(f"{title}: {snippet} ({link})")
                
                return " | ".join(results)
            else:
                return "No results found"
                
        except Exception as e:
            print(f"Google Custom Search failed: {e}")
            return None
    
    def validate_api_keys():
        """验证 API keys 的有效性"""
        valid_keys = {}
        
        # 验证 Google Custom Search
        if settings.GOOGLE_CUSTOM_SEARCH_API_KEY and settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID:
            try:
                result = google_custom_search("test")
                if result and result != "No results found":
                    valid_keys['google_custom_search'] = True
                else:
                    print("Google Custom Search API key validation failed")
                    valid_keys['google_custom_search'] = False
            except Exception as e:
                print(f"Google Custom Search API key validation error: {e}")
                valid_keys['google_custom_search'] = False
        else:
            valid_keys['google_custom_search'] = False
        
        # 验证 SerpAPI
        if settings.SERPAPI_API_KEY:
            try:
                test_search = SerpAPIWrapper(serpapi_api_key=settings.SERPAPI_API_KEY)
                result = test_search.run("test")
                if result and not "Invalid API key" in str(result):
                    valid_keys['serpapi'] = True
                else:
                    print("SerpAPI API key validation failed")
                    valid_keys['serpapi'] = False
            except Exception as e:
                print(f"SerpAPI API key validation error: {e}")
                valid_keys['serpapi'] = False
        else:
            valid_keys['serpapi'] = False
        
        return valid_keys
    
    # 验证 API keys（只在第一次调用时）
    if not hasattr(create_search_tool, 'valid_api_keys'):
        print("Validating search API keys...")
        create_search_tool.valid_api_keys = validate_api_keys()
        print(f"API key validation results: {create_search_tool.valid_api_keys}")
    
    def search_with_fallback(query: str) -> str:
        """
        搜索函数，三层级降级：DuckDuckGo -> Google Custom Search -> SerpAPI
        """
        import time
        import random
        current_time = time.time()
        
        # 检查各服务的健康状态
        skip_duckduckgo = (
            create_search_tool.search_stats['duckduckgo_failures'] > 3 and 
            current_time - create_search_tool.search_stats['last_duckduckgo_failure'] < 300  # 5分钟内失败超过3次
        )
        
        skip_google_custom_search = (
            create_search_tool.search_stats['google_custom_search_failures'] > 2 and 
            current_time - create_search_tool.search_stats['last_google_custom_search_failure'] < 600  # 10分钟内失败超过2次
        )
        
        # 第一层：DuckDuckGo
        if settings.ENABLE_DUCKDUCKGO_SEARCH and not skip_duckduckgo:
            try:
                if not hasattr(create_search_tool, 'duckduckgo_search'):
                    create_search_tool.duckduckgo_search = DuckDuckGoSearchAPIWrapper()
                
                start_time = time.time()
                result = create_search_tool.duckduckgo_search.run(query)
                search_time = time.time() - start_time
                
                # 重置失败计数，增加成功计数
                create_search_tool.search_stats['duckduckgo_failures'] = 0
                create_search_tool.search_stats['duckduckgo_successes'] += 1
                
                # 处理 DuckDuckGo 返回结果
                if result:
                    if isinstance(result, str) and result.strip():
                        print(f"DuckDuckGo search successful in {search_time:.2f}s")
                        return f"[DuckDuckGo] {result}"
                    elif isinstance(result, list) and result:
                        # 如果是列表，取第一个结果
                        first_result = result[0] if isinstance(result[0], str) else str(result[0])
                        if first_result.strip():
                            print(f"DuckDuckGo search successful in {search_time:.2f}s")
                            return f"[DuckDuckGo] {first_result}"
                else:
                    print("DuckDuckGo search returned empty result")
            except Exception as e:
                error_msg = str(e)
                create_search_tool.search_stats['duckduckgo_failures'] += 1
                create_search_tool.search_stats['last_duckduckgo_failure'] = current_time
                
                if "Ratelimit" in error_msg or "429" in error_msg or "202" in error_msg:
                    print(f"DuckDuckGo rate limited: {e}")
                    # 增加延迟时间避免速率限制
                    time.sleep(random.uniform(3, 8))
                else:
                    print(f"DuckDuckGo search failed: {e}")
        
        # 第二层：Google Custom Search（如果 API key 有效且未被跳过）
        if (settings.ENABLE_GOOGLE_CUSTOM_SEARCH and 
            create_search_tool.valid_api_keys.get('google_custom_search', False) and 
            not skip_google_custom_search):
            try:
                start_time = time.time()
                result = google_custom_search(query)
                search_time = time.time() - start_time
                
                # 重置失败计数，增加成功计数
                create_search_tool.search_stats['google_custom_search_failures'] = 0
                create_search_tool.search_stats['google_custom_search_successes'] += 1
                
                # 处理 Google Custom Search 返回结果
                if result and result != "No results found":
                    print(f"Google Custom Search successful in {search_time:.2f}s")
                    return f"[Google Custom Search] {result}"
                else:
                    print("Google Custom Search returned no results")
            except Exception as e:
                create_search_tool.search_stats['google_custom_search_failures'] += 1
                create_search_tool.search_stats['last_google_custom_search_failure'] = current_time
                print(f"Google Custom Search failed: {e}")
        
        # 第三层：SerpAPI（原始）
        if (settings.ENABLE_SERPAPI_FALLBACK and 
            create_search_tool.valid_api_keys.get('serpapi', False)):
            try:
                if not hasattr(create_search_tool, 'serpapi_search'):
                    create_search_tool.serpapi_search = SerpAPIWrapper(serpapi_api_key=settings.SERPAPI_API_KEY)
                
                start_time = time.time()
                result = create_search_tool.serpapi_search.run(query)
                search_time = time.time() - start_time
                
                # 重置失败计数，增加成功计数
                create_search_tool.search_stats['serpapi_failures'] = 0
                create_search_tool.search_stats['serpapi_successes'] += 1
                
                # 处理 SerpAPI 返回结果
                if result:
                    if isinstance(result, str) and result.strip():
                        print(f"SerpAPI search successful in {search_time:.2f}s")
                        return f"[SerpAPI] {result}"
                    elif isinstance(result, list) and result:
                        # 如果是列表，合并所有结果
                        if all(isinstance(item, str) for item in result):
                            combined_result = " ".join(result)
                            if combined_result.strip():
                                print(f"SerpAPI search successful in {search_time:.2f}s")
                                return f"[SerpAPI] {combined_result}"
                        else:
                            # 如果列表包含非字符串，转换为字符串
                            combined_result = " ".join([str(item) for item in result])
                            if combined_result.strip():
                                print(f"SerpAPI search successful in {search_time:.2f}s")
                                return f"[SerpAPI] {combined_result}"
                else:
                    print("SerpAPI search returned empty result")
            except Exception as e:
                create_search_tool.search_stats['serpapi_failures'] += 1
                create_search_tool.search_stats['last_serpapi_failure'] = current_time
                print(f"SerpAPI search failed: {e}")
        
        return "Error: All search services failed. Please try again later."
    
    return Tool(
        name="search",
        func=search_with_fallback,
        description="Searches the web using DuckDuckGo (primary), Google Custom Search (secondary), or SerpAPI (fallback). Use this tool to find current information about recent events, news, or any topic you need to research."
    )

# === LangGraph ReAct Agent (with web search) ===
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_PATH,
    model_name=settings.OPENAI_MODEL_NAME,  # 使用环境变量配置的模型
    temperature=0.7,
)

# 使用工厂函数创建搜索工具
search_tool = create_search_tool()
tools = [search_tool]

react_agent = create_react_agent(llm, tools)

# 正确的 StateGraph 初始化
graph = StateGraph(dict)
graph.add_node("agent", react_agent)
graph.set_entry_point("agent")
app_graph = graph.compile()

class PromptRequest(BaseModel):
    prompt: str
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "onyx"
    # 可选人声: alloy, echo, fable, onyx, nova, shimmer

class SpeechResponse(BaseModel):
    url: str
    object_name: str
    words: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    content: str
    assistant_id: str
    thread_id: str

class TranscriptionResponse(BaseModel):
    content: str

class AgentChatRequest(BaseModel):
    prompt: str
    model_name: str | None = None
class AgentChatResponse(BaseModel):
    content: str
    # history: list

# 全局只初始化一次 LLM、agent、graph、tools
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_PATH,
    model_name=settings.OPENAI_MODEL_NAME,
    temperature=0.7,
)
react_agent = create_react_agent(llm, tools)
graph = StateGraph(dict)
graph.add_node("agent", react_agent)
graph.set_entry_point("agent")
app_graph = graph.compile()

def get_next_msg_id():
    return int(redis_client.incr(REDIS_CHAT_ID_KEY))

def get_redis_chat_key(session_id=None):
    if session_id:
        return f'chat_history:{session_id}'
    return REDIS_CHAT_KEY

# Redis List 版：获取最近 N 条历史

def get_chat_history(limit=MAX_HISTORY, session_id=None):
    redis_key = get_redis_chat_key(session_id)
    messages = redis_client.lrange(redis_key, -limit, -1)
    
    def safe_load_message(data):
        try:
            msg_dict = json.loads(data)
            
            # 新格式：手动序列化的 dict
            if "type" in msg_dict:
                if msg_dict["type"] == "human":
                    return HumanMessage(content=msg_dict.get("content", ""))
                elif msg_dict["type"] == "ai":
                    return AIMessage(content=msg_dict.get("content", ""))
                # 不加载 tool 消息，避免 tool_calls 序列问题
            
            # 旧格式兼容
            elif msg_dict.get("role") == "user" or msg_dict.get("role") == "human":
                return HumanMessage(content=msg_dict.get("content", ""))
            elif msg_dict.get("role") == "ai":
                return AIMessage(content=msg_dict.get("content", ""))
            # 不加载 tool 消息
                
        except Exception as e:
            print(f"Failed to load message: {e}")
            return None
    
    loaded_messages = []
    for m in messages:
        msg = safe_load_message(m)
        if msg is not None:
            loaded_messages.append(msg)
    
    return loaded_messages

def append_chat_message(msg, session_id=None):
    redis_key = get_redis_chat_key(session_id)
    
    # 如果是 LangChain Message 对象，手动提取字段
    if hasattr(msg, 'type') and hasattr(msg, 'content'):
        msg_dict = {
            "type": msg.type,
            "content": msg.content,
            "id": get_next_msg_id()
        }
        # 为 tool 消息添加特殊字段
        if msg.type == "tool" and hasattr(msg, 'tool_call_id'):
            msg_dict["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, 'name'):
            msg_dict["name"] = msg.name
        # 为 tool 消息添加标记，表示这是工具调用结果
        if msg.type == "tool":
            msg_dict["is_tool_result"] = True
    else:
        # 兼容普通 dict
        msg_dict = msg
        if isinstance(msg_dict, dict) and not msg_dict.get("id"):
            msg_dict["id"] = get_next_msg_id()
    
    redis_client.rpush(redis_key, json.dumps(msg_dict))
    redis_client.ltrim(redis_key, -MAX_HISTORY, -1)

def save_chat_history(history, session_id=None):
    """
    全量重置某会话的历史，仅用于后台清空/批量导入等特殊场景。
    日常对话请用 append_chat_message。
    """
    redis_key = get_redis_chat_key(session_id)
    # 最多只保存 MAX_HISTORY 条
    history = history[-MAX_HISTORY:]
    pipe = redis_client.pipeline()
    pipe.delete(redis_key)
    if history:
        pipe.rpush(redis_key, *[json.dumps(m) for m in history])
    pipe.execute()

logger = logging.getLogger(__name__)

@router.post("/generate/speech", response_model=SpeechResponse)
async def generate_speech_from_prompt(request: PromptRequest):
    """
    Receives a prompt, generates speech, transcribes it for timestamps,
    uploads it to R2, and returns a presigned URL and the word timestamps.
    可选 voice: alloy, echo, fable, onyx, nova, shimmer
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    if r2_service is None:
        raise HTTPException(status_code=500, detail="R2 service is not configured on the server.")

    try:
        endpoint_start_time = time.time()
        print(f"\nProcessing request for prompt: '{request.prompt[:30]}...'")

        # Step 1: Generate audio from OpenAI TTS
        start_tts = time.time()
        tts_response = client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=request.prompt,
            response_format="mp3"
        )
        audio_bytes = tts_response.read()
        end_tts = time.time()
        print(f"  - Step 1 (TTS Generation) took: {end_tts - start_tts:.4f} seconds")

        # Step 2: Transcribe the generated audio to get timestamps
        start_transcription = time.time()
        audio_file_tuple = ("prompt_audio.mp3", audio_bytes)
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_tuple,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        word_timestamps = [word.model_dump() for word in transcription_response.words]
        end_transcription = time.time()
        print(f"  - Step 2 (Transcription) took: {end_transcription - start_transcription:.4f} seconds")

        # Step 3: Upload the audio bytes to R2
        start_upload = time.time()
        object_name = r2_service.upload_file_from_bytes(
            data=audio_bytes,
            content_type="audio/mpeg"
        )
        end_upload = time.time()
        print(f"  - Step 3 (R2 Upload) took: {end_upload - start_upload:.4f} seconds")

        # Step 4: Generate a presigned URL for the uploaded object
        start_presign = time.time()
        presigned_url = r2_service.generate_presigned_url(object_name)
        end_presign = time.time()
        print(f"  - Step 4 (Presigned URL) took: {end_presign - start_presign:.4f} seconds")

        if not presigned_url:
            raise HTTPException(status_code=500, detail="Could not generate presigned URL for the audio file.")

        endpoint_end_time = time.time()
        print(f"  - Total time inside endpoint: {endpoint_end_time - endpoint_start_time:.4f} seconds")
        
        return {"url": presigned_url, "object_name": object_name, "words": word_timestamps}

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/generate/chat", response_model=ChatResponse)
async def generate_chat_response(request: ChatRequest):
    """
    接收一个文本，使用 OpenAI 的 Chat API 进行对话，将结果返回给前端。
    """
    print(f"OpenAI Base Path: {settings.OPENAI_BASE_PATH}")
    print(f"OpenAI API Key: {settings.OPENAI_API_KEY[:10]}..." if settings.OPENAI_API_KEY else "OpenAI API Key: Not set")
    
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    if not settings.OPENAI_MODEL_NAME:
        raise HTTPException(status_code=500, detail="OpenAI model name is not configured on the server.")
    
    try:
        # 使用 Chat API 进行对话
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful English learning assistant. Help users improve their English skills through conversation, grammar correction, and vocabulary building."
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print(f"Chat API response: {result}")
        
        return {
            "content": result,
            "assistant_id": "chat-api",  # 使用固定值表示这是 Chat API
            "thread_id": "chat-api"      # 使用固定值表示这是 Chat API
        }
        
    except Exception as e:
        tb = traceback.format_exc()
        # 尝试获取 openai 的 response body
        error_detail = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_detail += f"\nResponse body: {e.response.text}"
        if hasattr(e, 'status_code'):
            error_detail += f"\nStatus code: {e.status_code}"
        print(f"An error occurred in chat endpoint: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/generate/transcription", response_model=TranscriptionResponse)
async def generate_transcription(file: UploadFile = File(...)):
    """
    接收一个音频文件，调用 OpenAI 的转写接口，返回转写后的文字。
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    try:
        audio_bytes = await file.read()
        audio_file_tuple = (file.filename, audio_bytes)
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file_tuple,
            response_format="text"
        )
        print(f"Transcription response: {transcription_response}")
        return {"content": transcription_response}
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_detail += f"\nResponse body: {e.response.text}"
        if hasattr(e, 'status_code'):
            error_detail += f"\nStatus code: {e.status_code}"
        print(f"An error occurred in transcription endpoint: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/generate/agent_chat", response_model=AgentChatResponse)
async def generate_agent_chat_response(request: AgentChatRequest, session_id: str = None):
    """
    用 LangGraph ReAct Agent 实现的对话接口，聊天历史持久化到 Redis，并限制历史条数
    支持多会话，session_id 为空时兼容老逻辑
    """
    try:
        history = get_chat_history(session_id=session_id)
        # 为新消息分配唯一 id
        user_msg = HumanMessage(content=request.prompt)
        append_chat_message(user_msg, session_id=session_id)
        
        # 添加当前用户消息到历史
        messages = history + [user_msg]
        messages = messages[-MAX_HISTORY:]
        
        state = {"messages": messages}
        result = app_graph.invoke(state)
        new_history = result["messages"]
        
        # 只追加新生成的 ai/human/tool 消息（不重复追加 user）
        new_msgs = []
        for m in new_history:
            if hasattr(m, "type") and m.type != "human":
                new_msgs.append(m)
        
        for msg in new_msgs:
            append_chat_message(msg, session_id=session_id)
        
        ai_reply = next((m.content for m in reversed(new_msgs) if m.type == "ai"), "")
        logger.info(f"[agent_chat] ai_reply: {ai_reply}")
        return {"content": ai_reply}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"LangGraph agent error: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history")
async def get_chat_history_api(
    before_msg_id: int | None = None,
    limit: int = 10,  # 默认10
    session_id: str = None,
):
    """
    游标分页拉取聊天历史消息，只返回 user/human/ai 消息
    支持多会话，session_id 为空时兼容老逻辑
    """
    limit = min(limit, 50)  # 最大50
    messages = redis_client.lrange(get_redis_chat_key(session_id), 0, -1)
    
    def safe_load_message(data):
        try:
            msg_dict = json.loads(data)
            
            # 新格式：手动序列化的 dict
            if "type" in msg_dict:
                if msg_dict["type"] in ("human", "ai"):
                    return {
                        "role": msg_dict["type"], 
                        "content": msg_dict.get("content", ""), 
                        "id": msg_dict.get("id")
                    }
            
            # 旧格式兼容
            elif msg_dict.get("role") in ("user", "human", "ai"):
                return {
                    "role": "human" if msg_dict.get("role") in ("user", "human") else "ai",
                    "content": msg_dict.get("content", ""), 
                    "id": msg_dict.get("id")
                }
        except Exception as e:
            print(f"Failed to load message in history API: {e}")
            return None
    
    # 处理消息
    display_history = []
    for m in messages:
        msg_dict = safe_load_message(m)
        if msg_dict:
            display_history.append(msg_dict)
    
    # 按 id 倒序排列（最新在前）
    display_history = sorted(display_history, key=lambda x: x.get("id", 0) or 0, reverse=True)
    if before_msg_id:
        display_history = [msg for msg in display_history if (msg.get("id") or 0) < before_msg_id]
    paged = display_history[:limit]
    return {"history": paged}

# Assistant API 管理端点

class CreateAssistantRequest(BaseModel):
    name: str = "English Learning Assistant"
    instructions: str = "You are a helpful English learning assistant. Help users improve their English skills through conversation, grammar correction, and vocabulary building."
    model: str = settings.OPENAI_MODEL_NAME

class AssistantResponse(BaseModel):
    assistant_id: str
    name: str
    instructions: str
    model: str

class ThreadResponse(BaseModel):
    thread_id: str

class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: int

class MessagesResponse(BaseModel):
    messages: List[MessageResponse]

@router.post("/assistant/create", response_model=AssistantResponse)
async def create_assistant(request: CreateAssistantRequest):
    """
    创建一个新的 Assistant
    """
    global assistant_id
    
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    try:
        assistant = client.beta.assistants.create(
            name=request.name,
            instructions=request.instructions,
            model=request.model
        )
        assistant_id = assistant.id
        
        return {
            "assistant_id": assistant.id,
            "name": assistant.name,
            "instructions": assistant.instructions,
            "model": assistant.model
        }
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        print(f"An error occurred creating assistant: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/thread/create", response_model=ThreadResponse)
async def create_thread():
    """
    创建一个新的 Thread
    """
    global thread_id
    
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    try:
        thread = client.beta.threads.create()
        thread_id = thread.id
        
        return {"thread_id": thread.id}
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        print(f"An error occurred creating thread: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/thread/{thread_id}/messages", response_model=MessagesResponse)
async def get_thread_messages(thread_id: str):
    """
    获取指定 Thread 的所有消息
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    try:
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        
        message_list = []
        for msg in messages.data:
            content = ""
            if msg.content and len(msg.content) > 0:
                content = msg.content[0].text.value
            
            message_list.append(MessageResponse(
                message_id=msg.id,
                role=msg.role,
                content=content,
                created_at=msg.created_at
            ))
        
        return {"messages": message_list}
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        print(f"An error occurred getting thread messages: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.delete("/assistant/{assistant_id}")
async def delete_assistant(assistant_id: str):
    """
    删除指定的 Assistant
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    try:
        client.beta.assistants.delete(assistant_id)
        return {"message": f"Assistant {assistant_id} deleted successfully"}
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        print(f"An error occurred deleting assistant: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.delete("/thread/{thread_id}")
async def delete_thread(thread_id: str):
    """
    删除指定的 Thread
    """
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured on the server.")
    
    try:
        client.beta.threads.delete(thread_id)
        return {"message": f"Thread {thread_id} deleted successfully"}
    except Exception as e:
        tb = traceback.format_exc()
        error_detail = str(e)
        print(f"An error occurred deleting thread: {error_detail}\nTraceback:\n{tb}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.post("/dev/reset")
async def reset_dev_environment():
    """
    重置开发环境，删除本地保存的 ID 文件，下次启动时会创建新的 Assistant 和 Thread
    """
    global assistant_id, thread_id
    
    try:
        # 删除本地文件
        if os.path.exists(DEV_IDS_FILE):
            os.remove(DEV_IDS_FILE)
            print(f"Deleted {DEV_IDS_FILE}")
        
        # 重置全局变量
        assistant_id = None
        thread_id = None
        
        return {
            "message": "Development environment reset successfully. New Assistant and Thread will be created on next request.",
            "deleted_file": DEV_IDS_FILE
        }
    except Exception as e:
        error_detail = str(e)
        print(f"An error occurred resetting dev environment: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/dev/status")
async def get_dev_status():
    """
    获取开发环境状态，显示当前使用的 Assistant 和 Thread ID
    """
    return {
        "assistant_id": assistant_id,
        "thread_id": thread_id,
        "ids_file": DEV_IDS_FILE,
        "file_exists": os.path.exists(DEV_IDS_FILE)
    }

@router.get("/dev/search-status")
async def get_search_status():
    """
    获取搜索服务状态，用于调试和监控
    """
    if hasattr(create_search_tool, 'search_stats'):
        stats = create_search_tool.search_stats
    else:
        stats = {
            'duckduckgo_failures': 0,
            'google_custom_search_failures': 0,
            'serpapi_failures': 0,
            'last_duckduckgo_failure': 0,
            'last_google_custom_search_failure': 0,
            'last_serpapi_failure': 0,
            'duckduckgo_successes': 0,
            'google_custom_search_successes': 0,
            'serpapi_successes': 0
        }
    
    import time
    current_time = time.time()
    
    # 获取 API key 验证状态
    valid_api_keys = getattr(create_search_tool, 'valid_api_keys', {})
    
    return {
        "search_configuration": {
            "enable_duckduckgo": settings.ENABLE_DUCKDUCKGO_SEARCH,
            "enable_google_custom_search": settings.ENABLE_GOOGLE_CUSTOM_SEARCH,
            "enable_serpapi_fallback": settings.ENABLE_SERPAPI_FALLBACK,
            "google_custom_search_configured": bool(settings.GOOGLE_CUSTOM_SEARCH_API_KEY and settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID),
            "serpapi_configured": bool(settings.SERPAPI_API_KEY),
            "duckduckgo_timeout": settings.DUCKDUCKGO_TIMEOUT
        },
        "api_key_validation": {
            "google_custom_search_valid": valid_api_keys.get('google_custom_search', False),
            "serpapi_valid": valid_api_keys.get('serpapi', False)
        },
        "search_statistics": {
            "duckduckgo_failures": stats['duckduckgo_failures'],
            "google_custom_search_failures": stats['google_custom_search_failures'],
            "serpapi_failures": stats['serpapi_failures'],
            "duckduckgo_successes": stats['duckduckgo_successes'],
            "google_custom_search_successes": stats['google_custom_search_successes'],
            "serpapi_successes": stats['serpapi_successes'],
            "last_duckduckgo_failure_ago": f"{current_time - stats['last_duckduckgo_failure']:.1f}s" if stats['last_duckduckgo_failure'] > 0 else "Never",
            "last_google_custom_search_failure_ago": f"{current_time - stats['last_google_custom_search_failure']:.1f}s" if stats['last_google_custom_search_failure'] > 0 else "Never",
            "last_serpapi_failure_ago": f"{current_time - stats['last_serpapi_failure']:.1f}s" if stats['last_serpapi_failure'] > 0 else "Never"
        },
        "search_health": {
            "duckduckgo_healthy": stats['duckduckgo_failures'] < 3,
            "google_custom_search_healthy": stats['google_custom_search_failures'] < 2 and valid_api_keys.get('google_custom_search', False),
            "serpapi_healthy": stats['serpapi_failures'] < 3 and valid_api_keys.get('serpapi', False),
            "skip_duckduckgo": stats['duckduckgo_failures'] > 3 and (current_time - stats['last_duckduckgo_failure']) < 300,
            "skip_google_custom_search": stats['google_custom_search_failures'] > 2 and (current_time - stats['last_google_custom_search_failure']) < 600
        },
        "recommendations": {
            "primary_search": "DuckDuckGo" if stats['duckduckgo_failures'] < 3 else "Google Custom Search" if valid_api_keys.get('google_custom_search', False) else "SerpAPI",
            "check_google_custom_search_key": not valid_api_keys.get('google_custom_search', False) and bool(settings.GOOGLE_CUSTOM_SEARCH_API_KEY and settings.GOOGLE_CUSTOM_SEARCH_ENGINE_ID),
            "rate_limit_warning": stats['duckduckgo_failures'] > 0
        }
    }
    


    