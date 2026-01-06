import os
import sys

# Ensure riichienv is in path
sys.path.insert(0, os.path.abspath("riichienv/src"))

from riichienv.visualizer.viewer import MetadataInjector

# Mock Log (Simulate a simple flow)
mock_log = [
    {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "1z",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "tehais": [
            ["1m", "1m", "1p", "2p", "3p", "4p", "5p", "6p", "1s", "2s", "3s", "4s", "5s"],
            ["1s", "1s", "1s", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "4s", "5s"],
            ["1m", "1m", "1m", "2m", "2m", "2m", "3m", "3m", "3m", "4m", "4m", "4m", "5m"],
            ["1p", "1p", "1p", "2p", "2p", "2p", "3p", "3p", "3p", "4p", "4p", "4p", "5p"],
        ],
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
    },
    {"type": "tsumo", "actor": 0, "pai": "1z"},
    {"type": "dahai", "actor": 0, "pai": "1z", "tsumogiri": True, "reach": True},
    {"type": "tsumo", "actor": 0, "pai": "6s"},
    # Tsumo: fallback to hand[-1]
    {"type": "hora", "actor": 0, "target": 0, "ura_markers": ["1z"]},
    # Next round
    {"type": "end_kyoku"},
    {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "1z",
        "kyoku": 2,
        "honba": 0,
        "kyotaku": 0,
        "tehais": [
            ["1m", "1m", "1p", "2p", "3p", "4p", "5p", "6p", "1s", "2s", "3s", "4s", "5s"],
            ["1s"] * 13,
            ["1s"] * 13,
            ["1s"] * 13,
        ],
        "oya": 0,
        "scores": [25000] * 4,
    },
    {"type": "dahai", "actor": 3, "pai": "6s", "tsumogiri": False},
    # Ron: fallback to last_tile ("6s")
    {"type": "hora", "actor": 0, "target": 3, "ura_markers": []},
    {"type": "end_kyoku"},
]

try:
    print("Testing MetadataInjector directly...")
    injector = MetadataInjector(mock_log)
    enriched = injector.process()

    # Start looking for injected meta
    found_waits = False
    found_score = False
    found_results = False

    for ev in enriched:
        if "meta" in ev:
            if "waits" in ev["meta"]:
                print(f"Found Waits: {ev['meta']['waits']}")
                found_waits = True
            if "score" in ev["meta"]:
                print(f"Found Score in Hora: {ev['meta']['score']}")
                found_score = True
            if "results" in ev["meta"]:
                print(f"Found Results in EndKyoku: {ev['meta']['results']}")
                found_results = True

    if found_waits and found_results:
        print("VERIFICATION PASSED: Metadata injected and accumulated.")
    else:
        print("VERIFICATION FAILED: Missing metadata.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
