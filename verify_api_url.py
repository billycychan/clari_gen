
import os
import logging
from fastapi.testclient import TestClient

# Set env var before importing config that might read it (though from_env reads at runtime, module level code might not)
os.environ["API_URL"] = "http://test-url-verification.com/v1"
if "LOG_FILE" in os.environ:
    del os.environ["LOG_FILE"]

from core.clari_gen.config import Config
from apps.api.main import app

def test_config_loading():
    print("Testing Config class directly...")
    config = Config()
    assert config.app.api_url == "http://test-url-verification.com/v1", \
        f"Expected http://test-url-verification.com/v1, got {config.app.api_url}"
    print("Config class verified.")

def test_app_startup():
    print("Testing App startup logging...")
    # Capture logs
    logger = logging.getLogger("apps.api.main")
    logger.setLevel(logging.INFO)
    
    with TestClient(app) as client:
        # Just triggering startup is enough
        pass
    
    print("App startup verified (if no errors). Please check logs for 'Initializing with API URL'.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_config_loading()
    test_app_startup()
