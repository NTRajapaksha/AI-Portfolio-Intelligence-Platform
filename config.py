"""
Configuration and environment setup
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration"""
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY", None)
    
    # Feature Flags
    ENABLE_SENTIMENT: bool = os.getenv("ENABLE_SENTIMENT", "true").lower() == "true"
    ENABLE_LLM_ORCHESTRATION: bool = os.getenv("ENABLE_LLM_ORCHESTRATION", "false").lower() == "true"
    
    # Model Settings
    # UPDATED: Using the specific version ID to fix 404 error
    LLM_MODEL: str = "gemini-2.5-flash" 
    LLM_TEMPERATURE: float = 0.3
    
    # Analysis Settings
    DEFAULT_PERIOD: str = "2y"
    DEFAULT_FORECAST_DAYS: int = 60
    MAX_NEWS_ARTICLES: int = 10
    
    # Rate Limiting
    MAX_API_CALLS_PER_MINUTE: int = 10
    
    # Paths
    ASSETS_DIR: str = "assets"
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.GOOGLE_API_KEY:
            # Check if secrets are injected (Codespaces specific)
            pass
        
        if self.ENABLE_SENTIMENT and not self.NEWS_API_KEY:
            print("⚠️  Warning: Sentiment analysis enabled but NEWS_API_KEY not set. Disabling sentiment.")
            self.ENABLE_SENTIMENT = False
        
        # Create assets directory
        os.makedirs(self.ASSETS_DIR, exist_ok=True)

config = Config()