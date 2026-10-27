/** Shared panel materials for table overlays and dialogs. */
export const PANEL_CSS = `
:where(.scene-3d, .replay-dock, .re-modal-overlay) {
    --panel-bg: #30333b;
    --panel-raised: #42454e;
    --panel-inset: #23262d;
    --panel-border: #777982;
    --panel-divider: #4b4e58;
    --panel-text: #f2f4fa;
    --panel-muted: #c5c8d0;
    --panel-accent: #65b6dc;
    --panel-radius: 8px;
    --panel-shadow: 0 3px 0 #17191f, 0 5px 12px #0005, inset 0 1px #ffffff20;
    --panel-font: -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
}
.re-modal-overlay { background: #101217b8; font-family: var(--panel-font); }
.re-modal-content {
    background: var(--panel-bg); color: var(--panel-text);
    border: 1px solid var(--panel-border); border-radius: var(--panel-radius);
    box-shadow: var(--panel-shadow); font-family: var(--panel-font);
}
.re-result-dialog { box-sizing: border-box; min-width: 0; width: 660px; padding: 16px; font-size: 14px; }
.re-result-dialog .re-modal-title { font-size: 18px; line-height: 1.4; border-color: var(--panel-divider); margin: 0 0 12px; padding-bottom: 10px; }
.re-result-hand { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 0; margin-bottom: 10px; padding: 10px; background: var(--panel-inset); border-radius: 6px; }
.re-result-dora { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; padding: 6px 10px; background: var(--panel-inset); border-radius: 6px; font-size: 12px; color: var(--panel-muted); }
.re-result-dialog .re-yaku-list { columns: 2; column-gap: 40px; list-style: none; padding: 0; margin: 15px 0; font: bold 1.8em/1.8 "Times New Roman", Times, serif; }
.re-result-dialog .re-yaku-list li { margin-bottom: 5px; border-bottom: 1px dotted var(--panel-border); break-inside: avoid; overflow-wrap: anywhere; }
.re-result-stats { display: flex; justify-content: space-between; margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--panel-divider); font-weight: 600; color: var(--panel-muted); }
.re-result-dialog .re-score-display { margin-top: 10px; padding: 9px; border: 1px solid #776535; border-radius: 6px; background: #363326; color: #f8d55b; font-size: 20px; font-variant-numeric: tabular-nums; }
.re-result-dialog .limit-banner { margin: 8px auto; padding: 4px 12px; font-size: 14px; letter-spacing: 0; background: #443d2c; color: #ffe28b; border: 1px solid #97804c; box-shadow: none; animation: none; }
.re-result-dialog .limit-yakuman { background: #f8d55b; color: #30250b; }
`;
