# DREV replay validation data

This manifest adds the repository-tracked `126_204_0_mjai.jsonl` replay to the
default DREV validation corpus. Its 12 complete rounds contain both tedashi and
tsumogiri discards, ten riichi declarations, calls, an ankan, wins, and draws.
It is used to stress public-history reconstruction and exact DREV invariants.

This fixture has no separately recorded upstream provenance, so it is not used
as an external score/yaku authority. The attributed fixtures in
`../external_replay/manifest.json` remain the cross-source differential oracle.
