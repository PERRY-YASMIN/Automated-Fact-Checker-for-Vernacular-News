from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str | None = None
    KAFKA_BOOTSTRAP_SERVERS: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()