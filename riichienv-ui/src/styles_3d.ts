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
        background: linear-gradient(135deg, #6b4226 0%, #5c3a21 30%, #4e2f1a 60%, #5a3720 100%);
        border-radius: 8px;
        box-shadow:
            0 18px 0 0 #3d2514,
            0 18px 50px rgba(0,0,0,0.7),
            inset 0 0 20px rgba(0,0,0,0.3);
        border: 2px solid #7a5234;
    }

    .table-inner {
        position: absolute;
        top: 76px; left: 76px; right: 76px; bottom: 76px;
        background: ${COLORS.boardBackground};
        border: 3px solid #3d2514;
        border-radius: 4px;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.3);
        transform-style: preserve-3d;
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
        transform-style: preserve-3d;
        align-items: flex-end;
    }
    .table-tile {
        width: 26px;
        height: 36px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .table-tile-rotated {
        width: 36px;
        height: 26px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .table-tile-rotated .tile-3d-top {
        overflow: visible;
    }
    .table-tile-rotated .tile-layer {
        position: absolute;
        width: 26px;
        height: 36px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) rotate(90deg);
        transform-origin: center center;
    }
    /* tsumogiri darkening is applied via overlay in renderer */

    /* ---- 3D Tile Box (table surface tiles) ---- */
    .tile-3d-top {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        border-radius: 3px;
        overflow: hidden;
        background: #f0ead6;
    }
    /* Remove inner border-radius and box-shadow inside 3D top face
       to prevent black gaps between top face and side faces */
    .tile-3d-top .tile-bg {
        border-radius: 0;
        box-shadow: none;
    }
    .tile-3d-front {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(to bottom, #e4dec8 70%, #c8a030 70%);
        transform-origin: bottom center;
        transform: rotateX(-90deg);
        border-radius: 0 0 2px 2px;
    }
    .tile-3d-right {
        position: absolute;
        top: 0;
        left: 100%;
        height: 100%;
        background: linear-gradient(to right, #b08828 30%, #dcd6c0 30%);
        transform-origin: left center;
        transform: rotateY(-90deg);
        border-radius: 0 2px 2px 0;
    }
    .tile-3d-back {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(to bottom, #c8a030 30%, #e4dec8 30%);
        transform-origin: top center;
        transform: rotateX(90deg);
        border-radius: 2px 2px 0 0;
    }
    .tile-3d-left {
        position: absolute;
        top: 0;
        right: 100%;
        height: 100%;
        background: linear-gradient(to right, #dcd6c0 70%, #b08828 70%);
        transform-origin: right center;
        transform: rotateY(90deg);
        border-radius: 2px 0 0 2px;
    }

    /* Opponent hand + melds area on table edge */
    .opp-hand-3d {
        position: absolute;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        width: 680px;
        transform-style: preserve-3d;
    }
    .opp-tiles-inner {
        display: flex;
        gap: 1px;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .opp-melds-inner {
        display: flex;
        flex-direction: row-reverse;
        gap: 3px;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .opp-meld-group {
        display: flex;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .opp-tile {
        width: 30px;
        height: 42px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .opp-tile-rotated {
        width: 42px;
        height: 30px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .opp-tile-rotated .tile-3d-top {
        overflow: visible;
    }
    .opp-tile-rotated .tile-layer {
        position: absolute;
        width: 30px;
        height: 42px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) rotate(90deg);
        transform-origin: center center;
    }
    .meld-tile-table {
        width: 20px;
        height: 28px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .meld-tile-table-rotated {
        width: 28px;
        height: 20px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .meld-tile-table-rotated .tile-3d-top {
        overflow: visible;
    }
    .meld-tile-table-rotated .tile-layer {
        position: absolute;
        width: 20px;
        height: 28px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) rotate(90deg);
        transform-origin: center center;
    }

    /* Center info on table */
    .center-info-3d {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 250px;
        height: 250px;
        background: ${COLORS.centerInfoBackground};
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.4);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 5;
        pointer-events: none;
        transition: background-color 0.15s;
    }
    .center-info-3d.hover {
        background-color: ${COLORS.highlightBoard} !important;
    }
    .center-click-zone {
        position: absolute;
        left: 50%;
        top: 34%;
        transform: translate(-50%, -50%);
        width: 24%;
        height: 40%;
        cursor: pointer;
        pointer-events: auto;
        z-index: 20;
    }
    .dora-tile-3d {
        width: 28px;
        height: 39px;
        position: relative;
        transform-style: preserve-3d;
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
        border-radius: 5px;
        overflow: hidden;
        background: #f0ead6;
        box-shadow:
            0 3px 0 0 #c8c0a8,
            1px 3px 0 0 #bfb7a3,
            1px 4px 6px rgba(0,0,0,0.3);
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
        position: relative;
        border-radius: 4px;
        overflow: hidden;
        background: #f0ead6;
        box-shadow:
            0 2px 0 0 #c8c0a8,
            1px 2px 0 0 #bfb7a3,
            1px 3px 5px rgba(0,0,0,0.25);
    }
    .meld-tile-own-rotated {
        width: 56px;
        height: 40px;
        position: relative;
        border-radius: 4px;
        background: #f0ead6;
        box-shadow:
            0 2px 0 0 #c8c0a8,
            1px 2px 0 0 #bfb7a3,
            1px 3px 5px rgba(0,0,0,0.25);
    }
    .meld-tile-own-rotated .tile-layer {
        position: absolute;
        width: 40px;
        height: 56px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) rotate(90deg);
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
        font-family: sans-serif;
        color: white;
        cursor: pointer;
        transition: background-color 0.2s;
        white-space: nowrap;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .score-panel-3d:hover {
        background: rgba(60,60,100,0.8);
    }
    .score-panel-3d.active-vp {
        border: 2px solid #aaa;
        background: rgba(0,0,0,0.85);
    }
    .score-panel-3d .player-name {
        font-size: 14px;
        font-weight: bold;
        color: #ffffff;
        margin-top: 2px;
        text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }
    .score-panel-3d .avatar-3d {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        overflow: hidden;
        background: rgba(255,255,255,0.1);
        margin: 0 auto;
        flex-shrink: 0;
    }
    .score-panel-3d .avatar-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .floating-score-3d {
        position: absolute;
        pointer-events: auto;
        font-family: monospace;
        font-size: 18px;
        font-weight: bold;
        color: #ffdd00;
        text-shadow: 0 1px 3px rgba(0,0,0,0.9), 0 0 6px rgba(0,0,0,0.5);
        cursor: pointer;
        white-space: nowrap;
        z-index: 10;
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
