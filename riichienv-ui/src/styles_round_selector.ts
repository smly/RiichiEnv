export const ROUND_SELECTOR_CSS = `
.re-modal-overlay.re-round-selector {
    padding: 10px;
    font: 11px/1.3 var(--panel-font);
}
.re-round-selector .round-browser {
    position: relative; width: 60%; min-width: min(420px, calc(100% - 48px)); max-width: 820px; max-height: 94%; padding: 0;
    container-type: inline-size; container-name: round-browser;
    display: flex; flex-direction: column; overflow: hidden; box-sizing: border-box;
    background: var(--panel-bg); color: var(--panel-text);
    border: 1px solid var(--panel-border); border-radius: var(--panel-radius); box-shadow: var(--panel-shadow);
}
.round-browser-tools { display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; padding: 3px 6px 3px 10px; background: var(--panel-bg); }
.round-browser-title { font-size: 12px; font-weight: 600; color: var(--panel-muted); }
.round-browser-close {
    width: 30px; height: 30px; padding: 0; border: 1px solid var(--panel-border); border-radius: 5px;
    background: var(--panel-raised); color: var(--panel-text); font: 22px/1 var(--panel-font); cursor: pointer;
}
.round-browser-close:hover { background: #535761; }
.round-browser button:focus-visible { outline: 2px solid var(--panel-accent); outline-offset: -2px; }
.round-browser-scroll { overflow-y: auto; overflow-x: hidden; min-height: 0; overscroll-behavior: contain; scrollbar-width: thin; scrollbar-color: var(--panel-border) var(--panel-inset); }
.round-browser-table { width: 100%; table-layout: fixed; border-spacing: 0; border-collapse: separate; font-size: 11px; }
.round-browser-caption { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
.round-browser-table th, .round-browser-table td { padding: 4px 5px; border-bottom: 1px solid var(--panel-divider); text-align: left; vertical-align: middle; }
.round-browser-table thead th {
    position: sticky; top: 0; z-index: 3; background: var(--panel-raised); color: var(--panel-muted);
    font-size: 11px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.round-browser-table thead th:first-child { width: 70px; }
.round-browser-row { cursor: pointer; }
.round-browser-row td, .round-browser-row th { background: var(--panel-bg); }
.round-browser-row:nth-child(even) td, .round-browser-row:nth-child(even) th { background: #343740; }
.round-browser-row:hover td, .round-browser-row:hover th, .round-browser-row:focus-within td, .round-browser-row:focus-within th { background: #4b4f59; }
.round-browser-row.is-current td, .round-browser-row.is-current th { background: #414650; }
.round-browser-row.is-current .round-browser-round { box-shadow: inset 3px 0 var(--panel-accent); }
.round-browser-table .round-browser-round { padding: 0; }
.round-browser .re-round-choice { width: 100%; min-height: 42px; padding: 4px 7px; text-align: left; font: inherit; }
.round-browser .re-round-choice strong { display: block; color: var(--panel-text); font-size: 12px; white-space: nowrap; }
.round-browser .re-round-choice > span:not(.round-badge) { display: block; margin-top: 2px; color: var(--panel-muted); font-size: 10px; }
.round-badge { display: inline-flex; align-items: center; max-width: 100%; border-radius: 3px; padding: 1px 3px; font-size: 9px; font-weight: 600; line-height: 1.3; overflow-wrap: anywhere; }
.round-badge.is-win { color: #a7f4e0; background: #184e4b; }
.round-badge.is-loss { color: #ffc2c2; background: #57313d; }
.round-badge.is-riichi { flex: 0 0 12px; justify-content: center; width: 12px; height: 14px; padding: 0; }
.round-badge.is-riichi::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: #ff777b; }
.round-badge.is-limit { color: #ffe28b; background: #443d2c; box-shadow: inset 0 0 0 1px #97804c; }
.round-badge.is-yakuman { color: #30250b; background: #ffe28b; }
.re-round-choice .round-badge { margin-top: 3px; }
.round-badge.is-draw { color: #e0e2e8; background: #4b4e58; }
.round-player { min-width: 0; font-variant-numeric: tabular-nums; }
.round-player-name { display: none; }
.round-player-status { min-height: 14px; display: flex; gap: 2px; align-items: center; flex-wrap: wrap; margin-bottom: 2px; }
.round-seat { display: inline-grid; place-items: center; flex-shrink: 0; width: 13px; height: 13px; font-size: 10px; color: var(--panel-muted); border: 1px solid var(--panel-border); border-radius: 3px; }
.round-seat.is-dealer { color: #ffc5bf; border-color: #aa666a; }
.round-delta { font-size: 13px; font-weight: 650; line-height: 1.25; color: #b9bdc7; }
.round-delta.is-positive { color: #9df0d6; }
.round-delta.is-negative { color: #ffabb3; }
.round-score-path { margin-top: 1px; color: var(--panel-muted); font-size: 10px; white-space: nowrap; }
.round-browser-empty { padding: 24px; text-align: center; color: var(--panel-muted); }
.round-browser [hidden] { display: none; }
@container round-browser (max-width: 700px) {
    .round-score-path { display: none; }
    .round-browser-table thead th:first-child { width: 60px; }
}
@container round-browser (max-width: 410px) {
    .round-browser-table, .round-browser-table tbody { display: block; }
    .round-browser-table thead { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
    .round-browser-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border-bottom: 1px solid var(--panel-border); }
    .round-browser-table .round-browser-round { grid-column: 1 / -1; border-bottom: 1px solid var(--panel-divider); }
    .round-browser .re-round-choice { display: flex; flex-wrap: wrap; align-items: center; gap: 3px 6px; min-height: 26px; padding: 3px 6px; }
    .round-browser .re-round-choice > span:not(.round-badge), .re-round-choice .round-badge { margin: 0; }
    .round-browser-table .round-player { display: grid; grid-template-columns: minmax(0, auto) minmax(0, 1fr); gap: 2px 5px; padding: 4px 6px; border-bottom: 0; }
    .round-player-name { display: block; grid-column: 1; grid-row: 1; margin-bottom: 0; color: var(--panel-text); font-size: 10px; font-weight: 600; overflow-wrap: anywhere; }
    .round-player-status { grid-column: 1 / -1; grid-row: 2; min-height: 0; margin: 0; }
    .round-delta { grid-column: 2; grid-row: 1; align-self: start; font-size: 12px; white-space: nowrap; }
    .round-score-path { display: none; }
}
@media (prefers-reduced-motion: reduce) { .re-modal-overlay.re-round-selector { animation: none; } }
`;
