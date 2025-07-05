from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Service Project"

    # OpenAI Credentials
    # 对于 Microsoft Azure OpenAI，设置你的 Azure API 密钥
    OPENAI_API_KEY: Optional[str] = None
    # 对于 Microsoft Azure OpenAI，格式为：
    # https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name
    OPENAI_BASE_PATH: Optional[str] = None
    OPENAI_MODEL_NAME: Optional[str] = None

    # Cloudflare R2 Credentials
    CLOUDFLARE_R2_ACCOUNT_ID: Optional[str] = None
    CLOUDFLARE_R2_ACCESS_KEY_ID: Optional[str] = None
    CLOUDFLARE_R2_SECRET_ACCESS_KEY: Optional[str] = None
    CLOUDFLARE_R2_BUCKET_NAME: Optional[str] = None
    R2_PUBLIC_DOMAIN: Optional[str] = None # e.g., "pub-your-hash.r2.dev"

    # Search API Credentials
    SERPAPI_API_KEY: Optional[str] = None  # 用于 SerpAPI 搜索（备选）
    GOOGLE_CUSTOM_SEARCH_API_KEY: Optional[str] = None  # Google Custom Search API key
    GOOGLE_CUSTOM_SEARCH_ENGINE_ID: Optional[str] = None  # Google Custom Search Engine ID
    
    # Search Configuration
    ENABLE_DUCKDUCKGO_SEARCH: bool = False  # 是否启用 DuckDuckGo 搜索
    DUCKDUCKGO_TIMEOUT: int = 10  # DuckDuckGo 搜索超时时间（秒）
    ENABLE_SERPAPI_FALLBACK: bool = True  # 是否启用 SerpAPI 降级
    ENABLE_GOOGLE_CUSTOM_SEARCH: bool = True  # 是否启用 Google Custom Search

    class Config:
        case_sensitive = True
        # This tells pydantic-settings to load variables from a .env file
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings() 