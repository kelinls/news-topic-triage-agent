"""Send the bundled article to the AgentX API and display the real response."""

import argparse
import json
from pathlib import Path

from .evaluate import request

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "graphics.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8088/api/triage")
    parser.add_argument("--file", type=Path, default=EXAMPLE)
    args = parser.parse_args()
    result = request(args.api, args.file.read_text(encoding="utf-8"), True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
