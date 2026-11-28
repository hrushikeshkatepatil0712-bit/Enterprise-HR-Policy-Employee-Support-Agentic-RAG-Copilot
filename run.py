import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the project root
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8080,
        reload=True
    )