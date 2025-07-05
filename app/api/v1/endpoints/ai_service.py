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
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
import logging

from app.core.config import settings
from app.services.r2_service import r2_service # Import the R2 service instance

router = APIRouter()

# Debug: Print environment variables
print(f"=== Environment Variables Debug ===")
print(f"OPENAI_API_KEY: {settings.OPENAI_API_KEY[:10]}..." if settings.OPENAI_API_KEY else "OPENAI_API_KEY: None")
print(f"OPENAI_BASE_PATH: {settings.OPENAI_BASE_PATH}")
print(f"CLOUDFLARE_R2_ACCOUNT_ID: {settings.CLOUDFLARE_R2_ACCOUNT_ID}")
print(f"CLOUDFLARE_R2_ACCESS_KEY_ID: {settings.CLOUDFLARE_R2_ACCESS_KEY_ID}")
print(f"CLOUDFLARE_R2_SECRET_ACCESS_KEY: {settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY[:10]}..." if settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY else "CLOUDFLARE_R2_SECRET_ACCESS_KEY: None")
print(f"CLOUDFLARE_R2_BUCKET_NAME: {settings.CLOUDFLARE_R2_BUCKET_NAME}")
print(f"R2_PUBLIC_DOMAIN: {settings.R2_PUBLIC_DOMAIN}")
print(f"===================================")

# Initialize the OpenAI client
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_PATH,
)

# 本地文件路径，用于保存 assistant 和 thread 的 ID
DEV_IDS_FILE = "dev_assistant_ids.json"

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

# === LangGraph ReAct Agent (with web search) ===
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_PATH,
    model_name="gpt-4o-mini",  # 使用 4.1 mini 模型
    temperature=0.7,
)
search_api = SerpAPIWrapper(serpapi_api_key=settings.SERPAPI_API_KEY)
search_tool = Tool(
    name="search",
    func=search_api.run,
    description="Searches the web using SerpAPI"
)
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
    history: list

# 全局只初始化一次 LLM、agent、graph、tools
llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    openai_api_base=settings.OPENAI_BASE_PATH,
    model_name=getattr(settings, "OPENAI_MODEL_NAME", None) or "gpt-4o-mini",
    temperature=0.7,
)
react_agent = create_react_agent(llm, tools)
graph = StateGraph(dict)
graph.add_node("agent", react_agent)
graph.set_entry_point("agent")
app_graph = graph.compile()

# 全局内存历史（单用户/无隔离）
chat_history = []

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
    
    try:
        # 使用 Chat API 进行对话
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
async def generate_agent_chat_response(request: AgentChatRequest):
    """
    用 LangGraph ReAct Agent 实现的对话接口，后端自动保存多轮历史（单用户版）
    """
    try:
        global chat_history
        messages = chat_history + [{"role": "user", "content": request.prompt}]
        state = {"messages": messages}
        result = app_graph.invoke(state)
        new_history = result["messages"]
        chat_history = new_history
        ai_reply = next((m.content for m in reversed(new_history) if getattr(m, "type", None) == "ai"), "")
        history_dict = [
            {"role": getattr(m, "type", None), "content": m.content}
            for m in new_history
        ]
        logger.info(f"[agent_chat] ai_reply: {ai_reply}")
        logger.info(f"[agent_chat] history_dict: {history_dict}")
        return {"content": ai_reply, "history": history_dict}
    except Exception as e:
        tb = traceback.format_exc()
        print(f"LangGraph agent error: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))

# Assistant API 管理端点

class CreateAssistantRequest(BaseModel):
    name: str = "English Learning Assistant"
    instructions: str = "You are a helpful English learning assistant. Help users improve their English skills through conversation, grammar correction, and vocabulary building."
    model: str = "gpt-4o-mini"

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
    


    