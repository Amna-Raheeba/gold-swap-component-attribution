"""Regenerate data/systems.json from the system definitions."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_swap.systems import systems_to_json


def main():
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "systems.json",
    )
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(systems_to_json(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
