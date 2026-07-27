from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    spec_copilot_model: str = "gemini-2.5-flash"
    spec_copilot_embedding_model: str = "gemini-embedding-001"
    spec_copilot_embedding_dimensions: int = 768
    opensearch_url: str = "http://localhost:9200"
    allowed_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]
