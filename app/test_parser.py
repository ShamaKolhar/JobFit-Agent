"""Quick manual test: parse a sample JD file and print the structured result."""

import json
from pathlib import Path

from jd_parser import parse_jd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JD_PATH = DATA_DIR / "sample_jds" / "jd_1.txt"
OUTPUT_PATH = DATA_DIR / "parsed" / "jd_1.json"


def main() -> None:
    jd_text = JD_PATH.read_text(encoding="utf-8")
    result = parse_jd(jd_text)
    output = json.dumps(result.model_dump(), indent=2)

    print(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
