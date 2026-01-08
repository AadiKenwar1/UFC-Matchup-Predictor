import sys
from pathlib import Path

# Add src/ to Python path so all imports work
src_dir = Path(__file__).parent.parent  # From src/backend/ up to src/
sys.path.insert(0, str(src_dir))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)