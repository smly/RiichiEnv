import { COLORS } from './constants';

export const VIEWER_CSS = `
    .mahjong-board {
        position: relative;
        width: 100%;
        aspect-ratio: 1/1;
        max-width: 900px;
        margin: 0 auto;
        background-color: ${COLORS.boardBackground};
        border-radius: 12px;
        overflow: hidden;
        font-size: 14px;
        color: white;
        font-family: sans-serif;
        box-sizing: border-box;
        user-select: none;
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
    }
    .mahjong-board svg { width: 100%; height: 100%; display: block; }
    .mahjong-board .center-info svg { width: auto; height: 100%; }
    
    .tile-layer {
        position: relative;
        width: 100%; 
        height: 100%;
    }
    .tile-bg, .tile-fg {
        position: absolute;
        top: 0; 
        left: 0;
        width: 100%;
        height: 100%;
    }
    .tile-bg { 
        z-index: 1; 
        border-radius: 4px;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .tile-fg { 
        z-index: 2; 
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .tile-fg svg {
        width: 85% !important;
        height: 85% !important;
    }
    
    @keyframes blink-yellow {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    .active-player-bar {
        height: 4px;
        width: 100%;
        background-color: #ffd700;
        margin-top: 4px;
        border-radius: 2px;
        animation: blink-yellow 1s infinite;
        box-shadow: 0 0 5px #ffd700;
    }
    
    .river-container {
        display: flex;
        flex-direction: column;
        gap: 2px;
        width: 214px; /* Fixed width: 6 * 34px + 5 * 2px = 214px */
        height: auto;
        min-height: 142px; /* Fixed min-height: 3 * 46px + 2 * 2px = 142px */
        justify-content: start;
        align-content: start;
    }
    .river-row {
        display: flex;
        gap: 2px;
    }
    
    .tile-rotated {
        transform: rotate(90deg) scale(0.9);
        transform-origin: center center;
    }

    .center-info {
        transition: background-color 0.2s;
    }
    .center-info:hover {
        background-color: ${COLORS.highlightBoard} !important;
    }

    @keyframes tsumo-enter {
        0% {
            opacity: 0;
            transform: translateY(-50px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .tsumo-anim {
        animation: tsumo-enter 0.2s ease-out forwards;
    }

    @keyframes dahai-enter {
        0% {
            opacity: 1;
            transform: translate(var(--dx), var(--dy));
        }
        100% {
            transform: translate(0, var(--target-y, 0));
        }
    }

    .dahai-anim {
        animation: dahai-enter 0.2s ease-out forwards;
    }

    @keyframes sort-slide {
        0% {
            transform: translateX(var(--sort-dx));
        }
        100% {
            transform: translateX(0);
        }
    }

    .sort-anim {
        /* Delay 200ms, Duration 200ms. Fill mode both/backwards ensures start state is held during delay */
        animation: sort-slide 0.2s ease-out 0.2s backwards;
    }

    .dahai-arrow {
        position: absolute;
        bottom: -8px; 
        left: 50%;
        transform: translateX(-50%);
        width: 0; 
        height: 0; 
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 6px solid white; /* Pointing UP at the tile */
        z-index: 101;
        animation: fade-in 0.2s ease-out 0.2s backwards;
    }

    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .player-info-box {
        background: rgba(0,0,0,0.6);
        padding: 8px;
        border-radius: 6px;
        color: white;
        text-align: center;
        min-width: 80px;
        z-index: 20;
        margin-left: 20px;
        transition: background-color 0.2s;
        cursor: pointer;
    }
    .player-info-box:hover {
        background-color: #444 !important;
    }
    .active-viewpoint {
        border: 2px solid #aaa;
        background: rgba(0,0,0,0.8);
    }

    .call-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3em;
        font-weight: bold;
        color: white;
        text-shadow: 0 0 5px #ff0000, 0 0 10px #000;
        padding: 10px 30px;
        background: rgba(0,0,0,0.6);
        border-radius: 10px;
        border: 2px solid white;
        z-index: 100;
        pointer-events: none;
        animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .icon-btn {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        color: white;
        border: 0px solid #666;
        border-radius: 8px;
        cursor: pointer;
        user-select: none;
        transition: all 0.2s;
        font-size: 20px;
    }
    .icon-btn:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #999;
    }
    .icon-btn:active {
        transform: translateY(1px);
        background: rgba(255, 255, 255, 0.2);
    }
    .active-btn {
        background: ${COLORS.modalBackground} !important;
        border-color: ${COLORS.highlightButton} !important;
        box-shadow: 0 0 5px ${COLORS.highlightButton};
    }

    .debug-panel {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 180px;
        background: rgba(0, 0, 0, 0.85);
        color: #0f0;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 13px;
        padding: 15px;
        overflow-y: auto;
        z-index: 200;
        display: none; /* Controlled by JS */
        box-sizing: border-box;
        border-bottom: 1px solid #333;
    }

    .re-modal-overlay {
        position: absolute; /* Changed from fixed to absolute */
        top: 0; left: 0;
        width: 100%;
        height: 100%;
        box-sizing: border-box;
        background: rgba(0,0,0,0.6);
        z-index: 2000; /* Increased z-index */
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s;
    }
    .re-modal-title { 
        font-size: 1.5em; 
        font-weight: bold; 
        margin-bottom: 10px; 
        border-bottom: 1px solid #777; 
        padding-bottom: 5px; 
    }
    .re-modal-content {
        background: ${COLORS.modalBackground};
        color: #fff;
        padding: 20px;
        border-radius: 8px;
        max-width: 95%;
        width: fit-content;
        max-height: 90%;
        min-width: 400px;
        overflow-y: auto;
        box-shadow: 0 4px 12px rgba(0,0,0,0.8);
        border: 1px solid ${COLORS.tableBorder};
        text-align: left;
    }
    .re-yaku-list { margin: 10px 0; padding-left: 20px; columns: 2; }
    .re-score-display { font-size: 1.2em; text-align: center; margin-top: 15px; font-weight: bold; background: #333; padding: 5px; border-radius: 4px;}

    .re-kyoku-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        background-color: transparent !important;
        color: white !important;
    }
    .re-kyoku-table th, .re-kyoku-table td {
        border: 1px solid ${COLORS.tableBorder};
        padding: 8px;
        text-align: center;
        background-color: ${COLORS.tableHeaderBackground} !important;
        color: white !important;
    }
    .re-kyoku-table th {
        background-color: ${COLORS.tableHeaderBackground} !important;
        position: sticky;
        top: 0;
        color: white !important;
    }
    .re-kyoku-row:hover td {
        background-color: ${COLORS.highlightBoard} !important;
        cursor: pointer;
    }

    .limit-banner {
        width: fit-content;
        margin: 10px auto;
        padding: 5px 15px;
        border-radius: 4px;
        font-weight: bold;
        text-align: center;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .limit-mangan { background-color: #d32f2f; } /* Red */
    .limit-haneman { background-color: #c2185b; } /* Pink */
    .limit-baiman { background-color: #7b1fa2; } /* Purple */
    .limit-sanbaiman { background-color: #512da8; } /* Deep Purple */
    .limit-yakuman { 
        background: linear-gradient(135deg, #ffd700 0%, #b8860b 100%);
        color: #000;
        font-size: 1.4em;
        padding: 8px 20px;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
        border: 1px solid #fff;
        animation: shine 2s infinite;
    }
    @keyframes shine {
        0% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
        100% { filter: brightness(1); }
    }
`;
