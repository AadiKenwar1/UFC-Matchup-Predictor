
import sys
import os
from pathlib import Path

# Add src/ to Python path so imports work
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )