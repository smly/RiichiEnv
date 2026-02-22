import { COLORS } from './constants';

export const VIEWER_3D_CSS = `
    /* ========================================
       Scene Structure
       ======================================== */
    .scene-3d {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        overflow: hidden;
    }

    /* Layer 1: CSS 3D Perspective Container */
    .table-perspective {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        perspective: 1500px;
        perspective-origin: 50% 40%;
    }

    .table-surface {
        position: absolute;
        width: 750px;
        height: 750px;
        left: 50%;
        top: 40%;
        transform: translate(-50%, -50%) rotateX(35deg);
        transform-style: preserve-3d;
        background: #1e2844;
        border-radius: 6px;
        box-shadow: 0 15px 0 0 #121830, 0 15px 40px rgba(0,0,0,0.7);
    }

    .table-inner {
        position: absolute;
        top: 18px; left: 18px; right: 18px; bottom: 18px;
        background: ${COLORS.boardBackground};
        border: 2px solid #4a5690;
        border-radius: 3px;
    }

    /* Layer 2: Hand Layer (flat, at bottom) */
    .hand-layer-3d {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 120px;
        display: flex;
        justify-content: center;
        align-items: flex-end;
        padding: 6px 40px;
        box-sizing: border-box;
        background: linear-gradient(to top, rgba(0,0,0,0.6) 0%, transparent 100%);
        z-index: 20;
    }

    /* Layer 3: UI Overlay */
    .ui-overlay-3d {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 30;
    }

    /* ========================================
       Table Elements
       ======================================== */

    /* River (discards) on table */
    .river-3d {
        position: absolute;
        display: flex;
        flex-direction: column;
        gap: 1px;
        transform-style: preserve-3d;
    }
    .river-row-3d {
        display: flex;
        gap: 1px;
    }
    .table-tile {
        width: 26px;
        height: 36px;
        flex-shrink: 0;
        position: relative;
    }
    .table-tile-rotated {
        width: 36px;
        height: 36px;
        flex-shrink: 0;
        position: relative;
    }
    .table-tile-rotated .tile-layer {
        transform: rotate(90deg) scale(0.88);
        transform-origin: center center;
    }
    .table-tile-tsumogiri {
        opacity: 0.6;
    }

    /* Opponent hand on table edge */
    .opp-hand-3d {
        position: absolute;
        display: flex;
        gap: 1px;
        transform-style: preserve-3d;
    }
    .opp-tile {
        width: 22px;
        height: 30px;
        flex-shrink: 0;
    }

    /* Opponent melds on table */
    .opp-meld-3d {
        position: absolute;
        display: flex;
        gap: 1px;
        transform-style: preserve-3d;
    }
    .meld-tile-table {
        width: 20px;
        height: 28px;
        flex-shrink: 0;
    }
    .meld-tile-table-rotated {
        width: 28px;
        height: 28px;
        flex-shrink: 0;
    }
    .meld-tile-table-rotated .tile-layer {
        transform: rotate(90deg) scale(0.85);
        transform-origin: center center;
    }

    /* Center info on table */
    .center-info-3d {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 180px;
        height: 180px;
        background: ${COLORS.centerInfoBackground};
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 5;
        cursor: pointer;
        transform-style: preserve-3d;
    }
    .center-info-3d:hover {
        background-color: ${COLORS.highlightBoard} !important;
    }
    .dora-tile-3d {
        width: 18px;
        height: 25px;
    }

    /* Riichi sticks on table */
    .riichi-stick-3d {
        position: absolute;
        width: 80px;
        height: 6px;
        background: white;
        border-radius: 3px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        z-index: 6;
        transform-style: preserve-3d;
    }
    .riichi-stick-3d .dot {
        width: 4px;
        height: 4px;
        background: #d00;
        border-radius: 50%;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }

    /* ========================================
       Hand Layer (own hand at bottom)
       ======================================== */
    .own-hand-area-3d {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        max-width: 900px;
        width: 100%;
    }
    .own-tiles-3d {
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
        flex-grow: 1;
    }
    .own-tile-3d {
        width: 50px;
        height: 70px;
        position: relative;
    }
    .own-melds-3d {
        display: flex;
        flex-direction: row-reverse;
        gap: 3px;
        align-items: flex-end;
    }
    .own-meld-group-3d {
        display: flex;
        align-items: flex-end;
        margin-left: 5px;
    }
    .meld-tile-own {
        width: 40px;
        height: 56px;
        display: flex;
        align-items: flex-end;
        justify-content: center;
    }
    .meld-tile-own-rotated {
        width: 56px;
        height: 56px;
        position: relative;
    }
    .meld-tile-own-rotated .tile-layer {
        transform: rotate(90deg);
        transform-origin: center center;
    }

    /* ========================================
       UI Overlay
       ======================================== */
    .score-panel-3d {
        position: absolute;
        pointer-events: auto;
        background: rgba(0,0,0,0.7);
        padding: 6px 12px;
        border-radius: 6px;
        text-align: center;
        font-family: monospace;
        font-size: 13px;
        color: white;
        cursor: pointer;
        transition: background-color 0.2s;
        min-width: 70px;
        white-space: nowrap;
    }
    .score-panel-3d:hover {
        background: rgba(60,60,100,0.8);
    }
    .score-panel-3d.active-vp {
        border: 2px solid #aaa;
        background: rgba(0,0,0,0.85);
    }
    .score-panel-3d .score-value {
        font-size: 15px;
        font-weight: bold;
        color: #ffdd00;
    }
    .score-panel-3d .wind-label {
        font-size: 11px;
        color: #aaa;
    }

    .call-overlay-3d {
        position: absolute;
        top: 35%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2.8em;
        font-weight: bold;
        color: white;
        text-shadow: 0 0 5px #ff0000, 0 0 10px #000;
        padding: 8px 25px;
        background: rgba(0,0,0,0.6);
        border-radius: 10px;
        border: 2px solid white;
        z-index: 50;
        pointer-events: none;
        animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    @keyframes popIn {
        0% { transform: translate(-50%, -50%) scale(0.5); opacity: 0; }
        100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
    }

    /* ========================================
       Animations
       ======================================== */
    @keyframes tsumo-enter-3d {
        0% { opacity: 0; transform: translateY(-40px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .tsumo-anim-3d {
        animation: tsumo-enter-3d 0.2s ease-out forwards;
    }

    /* ========================================
       Wait indicator
       ======================================== */
    .wait-indicator-3d {
        position: absolute;
        display: flex;
        gap: 3px;
        align-items: center;
        background: rgba(0,0,0,0.8);
        color: #fff;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        pointer-events: none;
        z-index: 40;
    }
    .wait-tile-3d {
        width: 20px;
        height: 28px;
    }
`;
