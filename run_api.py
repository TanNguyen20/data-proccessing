import os

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", 8000))

    # Run the FastAPI server
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port
    )
