import base64
import copy
import gzip
import hashlib
import json
import os
import traceback
import uuid
from typing import Any

from IPython.display import HTML

from riichienv import AgariCalculator, Conditions, Meld, MeldType, Wind
from riichienv import convert as cvt


def _get_viewer_js_compressed_base64() -> tuple[str, str]:
    """Returns the Gzipped JS content as a base64 string and its MD5 hash."""
    p_gz = os.path.join(os.path.dirname(__file__), "assets", "viewer.js.gz")

    # helper for processing
    def process_data(data: bytes) -> tuple[str, str]:
        b64 = base64.b64encode(data).decode("utf-8")
        h = hashlib.md5(data).hexdigest()
        return b64, h

    if os.path.exists(p_gz):
        with open(p_gz, "rb") as f:
            return process_data(f.read())

    # Fallback to compressing on the fly if .gz is missing but .js exists
    p = os.path.join(os.path.dirname(__file__), "assets", "viewer.js")
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = f.read()
            compressed = gzip.compress(data)
            return process_data(compressed)

    return "", ""


class MetadataInjector:
    def __init__(self, events: list[dict[str, Any]]):
        self.events = copy.deepcopy(events)
        self.hands: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
        self.melds: dict[int, list[Meld]] = {0: [], 1: [], 2: [], 3: []}  # List of Meld objects
        self.dora_markers: list[int] = []
        self.round_wind = 0  # 0: East, 1: South, etc.
        self.bakaze_map = {"E": 0, "S": 1, "W": 2, "N": 3}
        self.oya = 0
        self.riichi_declared = [False] * 4
        self.tile_counts = {}  # To track unique IDs for string tiles
        self.kyoku_results = []
        self.last_tile: str | None = None
        self.last_tid: int | None = None

        # State for Conditions
        self.ippatsu_eligible = [False] * 4
        self.is_rinshan = False
        self.is_chankan = False
        self.is_haitei = False
        self.is_first_round_of_kyoku = True
        self.any_melds_in_kyoku = False
        self.turn_count = 0
        self.kyoku_num = 0
        self.honba = 0

    def _get_tid(self, tile_str: str) -> int:
        """Get a unique 136-ID for a tile string to maintain valid state."""
        base_tid = cvt.mjai_to_tid(tile_str)
        cnt = self.tile_counts.get(base_tid, 0)
        self.tile_counts[base_tid] = cnt + 1
        return base_tid + cnt

    def _get_matching_tid(self, hand: list[int], tile_str: str) -> int:
        """Find a tile in hand that matches the tile_str type."""
        target_base = cvt.mjai_to_tid(tile_str)
        # Check for exact match first (e.g. Red vs Non-Red)
        # cvt.mpai_to_tid handles '5mr' -> 16, '5m' -> 17.
        # But ID 17 is just one of 17,18,19.

        # Helper to check if ID is red
        def is_red(t):
            return t in [16, 52, 88]

        target_is_red = is_red(target_base)

        candidates = []
        for t in hand:
            if t // 4 == target_base // 4:
                # Type match
                candidates.append(t)

        # Filter by Redness
        better = [t for t in candidates if is_red(t) == target_is_red]
        if better:
            return better[0]
        if candidates:
            return candidates[0]
        # Should not happen if log is valid
        return target_base

    def process(self) -> list[dict[str, Any]]:  # noqa: PLR0915
        for ev in self.events:
            etype = ev["type"]
            actor = ev.get("actor")

            # Initialize meta
            if "meta" not in ev:
                ev["meta"] = {}

            if etype == "start_kyoku":
                self.tile_counts = {}  # Reset for new kyoku
                self.dora_markers = [self._get_tid(ev["dora_marker"])]
                self.round_wind = self.bakaze_map.get(ev.get("bakaze", "E"), 0)
                self.oya = ev.get("oya", 0)
                self.kyoku_num = ev.get("kyoku", 1)  # Default to 1
                self.honba = ev.get("honba", 0)
                self.riichi_declared = [False] * 4
                self.hands = {0: [], 1: [], 2: [], 3: []}
                self.melds = {0: [], 1: [], 2: [], 3: []}
                self.kyoku_results = []
                self.kyoku_results = []
                self.ippatsu_eligible = [False] * 4
                self.just_reached = [False] * 4  # Track declaration discard
                self.is_rinshan = False
                self.is_chankan = False
                self.is_haitei = False
                self.is_first_round_of_kyoku = True
                self.any_melds_in_kyoku = False
                self.turn_count = 0

                for pid, tehai in enumerate(ev["tehais"]):
                    # tehai is list of strings
                    self.hands[pid] = sorted([self._get_tid(t) for t in tehai])

            elif etype == "tsumo":
                assert actor is not None
                self.last_tile = ev.get("pai")
                tile = self._get_tid(ev["pai"])
                self.hands[actor].append(tile)
                self.is_rinshan = False  # Reset unless set by Kan
                self.is_chankan = False

            elif etype == "dahai":
                assert actor is not None
                self.last_tile = ev.get("pai")
                pai = ev["pai"]

                # Identify which tile was discarded
                tid = self._get_matching_tid(self.hands[actor], pai)
                self.last_tid = tid  # Track the exact TID for calls
                if tid in self.hands[actor]:
                    self.hands[actor].remove(tid)

                # CHECK WAITS (Tenpai)
                # Calculate waits for this player after discard
                waits = self._calculate_waits(actor)
                if waits:
                    ev["meta"]["waits"] = waits

                if ev.get("reach"):
                    self.riichi_declared[actor] = True
                    self.ippatsu_eligible[actor] = True

                # If anyone discards, first round might be over
                if actor == 3:  # End of round
                    self.is_first_round_of_kyoku = False

                # Any discard clears ippatsu if it's not the riichi-er's first discard
                if self.just_reached[actor]:
                    self.just_reached[actor] = False
                else:
                    self.ippatsu_eligible[actor] = False

                self.turn_count += 1

            elif etype == "reach":
                assert actor is not None
                # Set Riichi declared flag immediately on announcement
                self.riichi_declared[actor] = True
                self.ippatsu_eligible[actor] = True
                self.just_reached[actor] = True

            elif etype in ["pon", "chi", "daiminkan"]:
                assert actor is not None
                # Remove consumed tiles from hand
                consumed = ev["consumed"]  # list of strings
                consumed_tids = []

                for c_str in consumed:
                    tid = self._get_matching_tid(self.hands[actor], c_str)

                    if tid in self.hands[actor]:
                        self.hands[actor].remove(tid)
                        consumed_tids.append(tid)
                    else:
                        # Fallback if not found (should generally not happen in valid log)
                        consumed_tids.append(self._get_tid(c_str))

                # Create Meld object
                stolen_str = ev["pai"]
                stolen_tid = 0
                if self.last_tid is not None:
                    stolen_tid = self.last_tid
                else:
                    # Fallback if logic mismatch (should not happen in valid log)
                    stolen_tid = self._get_tid(stolen_str)

                m_tiles = sorted(consumed_tids + [stolen_tid])

                m_type = MeldType.Peng
                if etype == "chi":
                    m_type = MeldType.Chi
                if etype == "daiminkan":
                    m_type = MeldType.Gang
                    self.is_rinshan = True

                self.melds[actor].append(Meld(m_type, m_tiles, True, actor))
                self.any_melds_in_kyoku = True
                # Clear all ippatsu on any call
                self.ippatsu_eligible = [False] * 4

            elif etype == "kakan":
                assert actor is not None
                self.last_tile = ev.get("pai")
                pai = ev["pai"]
                tid = self._get_matching_tid(self.hands[actor], pai)
                if tid in self.hands[actor]:
                    self.hands[actor].remove(tid)

                # Replace Pon with AddGang
                found_idx = -1
                for midx, m in enumerate(self.melds[actor]):
                    if m.meld_type == MeldType.Peng and m.tiles[0] // 4 == tid // 4:
                        found_idx = midx
                        break
                if found_idx != -1:
                    old_m = self.melds[actor].pop(found_idx)
                    new_tiles = sorted(old_m.tiles + [tid])
                    self.melds[actor].append(Meld(MeldType.Addgang, new_tiles, True, actor))
                    self.is_rinshan = True
                    self.is_chankan = True  # Eligible for Chankan

                self.any_melds_in_kyoku = True
                self.ippatsu_eligible = [False] * 4

            elif etype == "ankan":
                assert actor is not None
                consumed = ev["consumed"]
                m_tiles = []
                for c_str in consumed:
                    tid = self._get_matching_tid(self.hands[actor], c_str)
                    if tid in self.hands[actor]:
                        self.hands[actor].remove(tid)
                        m_tiles.append(tid)
                    else:
                        m_tiles.append(self._get_tid(c_str))

                m_tiles.sort()
                self.melds[actor].append(Meld(MeldType.Angang, m_tiles, False, actor))
                self.is_rinshan = True
                self.ippatsu_eligible = [False] * 4

            elif etype == "dora":
                self.dora_markers.append(self._get_tid(ev["dora_marker"]))

            elif etype == "hora":
                actor = ev.get("actor")
                target = ev.get("target")

                if actor is None or target is None:
                    continue

                is_tsumo = actor == target
                pai_str = ev.get("pai") or self.last_tile
                pai_tid = 0

                if is_tsumo:
                    # Tsumo: The winning tile should be the last one in hand
                    if self.hands[actor]:
                        pai_tid = self.hands[actor][-1]
                    elif pai_str:
                        pai_tid = self._get_tid(pai_str)
                # Ron: Use last_tid from dahai if available
                elif self.last_tid is not None:
                    pai_tid = self.last_tid
                elif pai_str:
                    pai_tid = self._get_tid(pai_str)

                # Double Riichi check
                is_double_riichi = False
                if self.riichi_declared[actor] and self.is_first_round_of_kyoku and not self.any_melds_in_kyoku:
                    is_double_riichi = True

                # Helpers for Wind conversion
                def get_wind(idx: int) -> Wind:
                    idx = idx % 4
                    if idx == 0:
                        return Wind.East
                    if idx == 1:
                        return Wind.South
                    if idx == 2:
                        return Wind.West
                    return Wind.North

                cond = Conditions(
                    tsumo=is_tsumo,
                    riichi=self.riichi_declared[actor],
                    double_riichi=is_double_riichi,
                    ippatsu=self.ippatsu_eligible[actor],
                    rinshan=self.is_rinshan if is_tsumo else False,
                    chankan=self.is_chankan if not is_tsumo else False,
                    player_wind=get_wind(actor - self.oya),
                    round_wind=get_wind(self.round_wind),
                    # haitei/houtei based on wall (if we had wall count)
                    # double_riichi if first round and no calls
                )

                # Ura markers
                ura_in = []
                if "ura_markers" in ev:
                    ura_in = [self._get_tid(u) for u in ev["ura_markers"]]

                calc = AgariCalculator(self.hands[actor], self.melds[actor])
                res = calc.calc(pai_tid, dora_indicators=self.dora_markers, conditions=cond, ura_indicators=ura_in)

                if res.agari:
                    pt_str = ""
                    if is_tsumo:
                        if actor == self.oya:
                            pt_str = f"{res.tsumo_agari_ko} all"
                        else:
                            pt_str = f"{res.tsumo_agari_ko}/{res.tsumo_agari_oya}"
                    else:
                        pt_str = str(res.ron_agari)

                    score_data = {"han": res.han, "fu": res.fu, "points": pt_str, "yaku": res.yaku}
                    ev["meta"]["score"] = score_data

                    self.kyoku_results.append({"actor": actor, "target": target, "score": score_data})

            elif etype == "end_kyoku":
                if self.kyoku_results:
                    ev["meta"]["results"] = self.kyoku_results

        return self.events

    def _calculate_waits(self, pid: int) -> list[str]:
        """Use AgariCalculator API to find waits."""
        hand = self.hands[pid]
        melds = self.melds[pid]

        calc = AgariCalculator(hand, melds)

        # get_waits returns list of u32 (0-33)
        wait_tids = calc.get_waits()

        waits = []
        for t34 in wait_tids:
            # Convert 34-tile ID to MJAI string (use *4 to get base 136 ID)
            waits.append(cvt.tid_to_mjai(t34 * 4))

        return waits


class Replay:
    def __init__(
        self, log: list[dict[str, Any]], step: int | None = None, perspective: int | None = None, freeze: bool = False
    ):
        self.log = log
        self.step = step
        self.perspective = perspective
        self.freeze = freeze

    @classmethod
    def from_jsonl(
        cls, path: str, step: int | None = None, perspective: int | None = None, freeze: bool = False
    ) -> HTML:
        with open(path, encoding="utf-8") as f:
            events = [json.loads(line) for line in f]
        return cls(events, step=step, perspective=perspective, freeze=freeze).show()

    @classmethod
    def from_list(
        cls, events: list[dict[str, Any]], step: int | None = None, perspective: int | None = None, freeze: bool = False
    ) -> HTML:
        return cls(events, step=step, perspective=perspective, freeze=freeze).show()

    def show(self) -> HTML:
        """
        Generates the HTML/JS viewer for the replay Log.
        Injects metadata (waits, scores) before rendering.
        """
        # Inject Metadata (Waits, Scores)
        try:
            injector = MetadataInjector(self.log)
            enriched_log = injector.process()
        except Exception as e:
            # Fallback if injection fails
            traceback.print_exc()
            print(f"Warning: Metadata injection failed: {e}")
            enriched_log = self.log

        unique_id = f"riichienv-viewer-{uuid.uuid4()}"
        log_json = json.dumps(enriched_log)
        viewer_js_b64, viewer_js_hash = _get_viewer_js_compressed_base64()

        if not viewer_js_b64:
            # Fallback if no assets found
            return HTML(f'<div id="{unique_id}">Error: Viewer assets not found.</div>')

        html_content = f"""
        <div id="{unique_id}" style="width: 100%; min-height: 600px; border: 1px solid #ddd; box-sizing: border-box;">
             <div style="padding: 20px; text-align: center; font-family: sans-serif; color: #666;">
                Loading RiichiEnv Replay...
             </div>
        </div>
        <script>
        (function() {{
            const expectedHash = "{viewer_js_hash}";

            const runViewer = (jsCode) => {{
                try {{
                    if (jsCode) {{
                        const script = document.createElement('script');
                        script.text = jsCode;
                        document.head.appendChild(script);
                        // Store the hash after loading new code
                        window.RiichiEnvViewerHash = expectedHash;
                    }}

                    const logData = {log_json};
                    const initialStep = {self.step if self.step is not None else "undefined"};
                    const perspective = {self.perspective if self.perspective is not None else "undefined"};
                    const freeze = {"true" if self.freeze else "false"};
                    if (window.RiichiEnvViewer) {{
                        new window.RiichiEnvViewer("{unique_id}", logData, initialStep, perspective, freeze);
                    }} else {{
                        throw new Error("RiichiEnvViewer global not found after injection");
                    }}
                }} catch (e) {{
                    console.error("RiichiEnv Viewer Error:", e);
                    document.getElementById("{unique_id}").innerHTML = "Error: " + e.message;
                }}
            }};

            // Check if global exists AND matches expected hash
            if (window.RiichiEnvViewer && window.RiichiEnvViewerHash === expectedHash) {{
                runViewer("");
            }} else {{
                // Decompress and load New Code
                const b64Data = "{viewer_js_b64}";
                const compressed = Uint8Array.from(atob(b64Data), c => c.charCodeAt(0));

                if (window.DecompressionStream) {{
                    const ds = new DecompressionStream('gzip');
                    const decompressedStream = new Response(compressed).body.pipeThrough(ds);
                    new Response(decompressedStream).text().then(runViewer).catch(e => {{
                        document.getElementById("{unique_id}").innerHTML = "Error decompressing: " + e.message;
                    }});
                }} else {{
                    const err = "Error: Browser too old (DecompressionStream missing).";
                    document.getElementById("{unique_id}").innerHTML = err;
                }}
            }}
        }})();
        </script>
        """
        return HTML(html_content)


def show_replay(log: list[dict[str, Any]]) -> HTML:
    """
    Displays a replay viewer for the given MJAI log.
    Start using Replay.from_list(log).show() instead.
    """
    return Replay.from_list(log)


def main() -> None:
    with open("example_before_injection.jsonl") as f:
        log = [json.loads(line) for line in f]
    injector = MetadataInjector(log)
    enriched_log = injector.process()
    with open("example_after_injection.jsonl", "w") as f:
        for line in enriched_log:
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
