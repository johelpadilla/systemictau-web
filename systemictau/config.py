from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_url: str = "http://localhost:8000"
    kafka_broker: str = "kafka://localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    jwt_secret: str = "super_secret_jwt_key_v5"
    gemini_api_key: str = "dummy_key"
    
    # Agent & Tool configuration
    google_api_key: str = Field(default="DUMMY_GEMINI_KEY", env="GOOGLE_API_KEY")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
