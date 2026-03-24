from __future__ import annotations

import asyncio
from pathlib import Path

from fluxnet_shuttle import listall


# =========================
# Path settings
# =========================
SNAPSHOT_DIR = Path("./fluxnet_snapshot")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Main function
# =========================
async def main():
    print("Generating FLUXNET Shuttle snapshot...")
    snapshot_path = await listall(output_dir=str(SNAPSHOT_DIR))
    print("Snapshot created successfully:")
    print(snapshot_path)


# =========================
# Run
# =========================
if __name__ == "__main__":
    asyncio.run(main())