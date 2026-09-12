import { COLORS } from './constants';
import { TILE_BACK_COLOR, TILE_FACE_COLOR } from './renderers/tile_material';

export const VIEWER_3D_CSS = `
    /* ========================================
       Scene Structure
       ======================================== */
    .scene-3d {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        overflow: hidden;
        contain: layout style;
        --tile-ivory: #f6f2e8;
        --tile-amber: ${TILE_BACK_COLOR};
        font-family: -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        background: #081f1b;
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
        background: linear-gradient(135deg, #363a40, #1c1f24 30%, #101216 70%, #292d33);
        border-radius: 8px;
        box-shadow:
            0 18px 0 0 #090a0c,
            0 18px 50px rgba(0,0,0,0.7),
            inset 0 2px 0 #686d74, inset 0 -5px 0 #090a0d;
        border: 2px solid #51565d;
    }

    /* One raised rim for the whole table; tile geometry stays unchanged. */
    .table-surface::before {
        content: '';
        position: absolute;
        inset: 64px;
        border: 12px solid #25282d;
        border-radius: 7px;
        box-shadow: 0 6px 0 #0a0c0f, inset 0 2px 0 #686d74, 0 -2px 0 #454a51;
        pointer-events: none;
    }

    .table-inner {
        position: absolute;
        top: 76px; left: 76px; right: 76px; bottom: 76px;
        background:
            radial-gradient(ellipse at 50% 38%, #24665f 0%, #184e47 48%, #103a34 100%);
        border: 3px solid #10161d;
        border-radius: 4px;
        box-shadow: inset 0 0 0 2px #47786d, inset 0 0 60px #061f1980;
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
        padding: 6px 40px 16px;
        box-sizing: border-box;
        background: linear-gradient(to top, rgba(8,31,27,0.85) 0%, transparent 100%);
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
        height: 39px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .table-tile-rotated {
        width: 39px;
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
        height: 39px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%) rotate(90deg);
        transform-origin: center center;
    }
    /* tsumogiri darkening is applied via overlay in renderer */

    /* A cached body bitmap replaces the many individual curved facets. */
    .tile-3d-surface { position: absolute; box-sizing: border-box; }
    .tile-body-image { left: 0; top: 0; transform-origin: 0 0; pointer-events: none; max-width: none; }
    .tile-3d-top { backface-visibility: hidden; }
    .scene-3d .tile-back .tile-sprite { background-image: none; background-color: var(--tile-amber); }
    /* A 3D face uses only the flat printed area of the existing PNG cell.
       Crop away the atlas's illustrated amber lip and gray frame: those are
       appropriate for flat hand images, but would duplicate the real body. */
    .tile-3d-top > .tile-layer { border-radius: inherit; }
    .scene-3d .tile-3d-top .tile-bg {
        overflow: hidden;
        border-radius: inherit;
        background: ${TILE_FACE_COLOR};
        box-shadow: none;
    }
    .tile-3d-top .tile-sprite {
        position: absolute;
        width: 107.142857%; /* 120 / 112 */
        height: 118.518519%; /* 192 / 162 */
        left: -3.571429%; /* 4 / 112 */
        top: -14.814815%; /* 24 / 162 */
        border-radius: 0;
    }
    /* Opponent hand + melds area on table edge */
    .opp-hand-3d {
        position: absolute;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 16px;
        transform-style: preserve-3d;
    }
    .opp-tiles-inner {
        display: flex;
        flex-shrink: 0;
        justify-content: flex-start;
        gap: 1px;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .opp-tiles-inner > .tile-face-down {
        transform: rotateX(90deg);
        transform-origin: center top;
    }
    .opp-melds-inner {
        display: flex;
        flex-direction: row-reverse;
        flex-shrink: 0;
        gap: 3px;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .opp-hand-3d > .opp-melds-inner { margin-left: auto; }
    .opp-meld-group {
        display: flex;
        align-items: flex-end;
        transform-style: preserve-3d;
    }
    .kakan-pair-3d {
        display: flex;
        flex-direction: column;
        gap: 1px;
        transform-style: preserve-3d;
    }
    .opp-tile {
        width: 30px;
        height: 45px;
        flex-shrink: 0;
        position: relative;
        transform-style: preserve-3d;
    }
    .opp-tile-rotated {
        width: 45px;
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
        height: 45px;
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
        transform: translate(-50%, -50%) translateZ(1px);
        width: 250px;
        height: 250px;
        background: linear-gradient(135deg, #62646e, #42454e 45%, #343740);
        border: 3px solid #23262e;
        box-sizing: border-box;
        border-radius: 12px;
        box-shadow: 0 5px 0 #14171e, 0 8px 12px #07172480, inset 0 0 0 2px #858792;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 5;
        pointer-events: none;
    }
    .center-info-3d::before, .center-info-3d::after {
        content: '';
        position: absolute;
        inset: 24px;
        clip-path: polygon(17% 0, 83% 0, 100% 17%, 100% 83%, 83% 100%, 17% 100%, 0 83%, 0 17%);
        background: #777982;
    }
    .center-info-3d::after {
        inset: 26px;
        background: linear-gradient(145deg, #30333b, #23262d);
    }
    .center-info-3d.hover {
        box-shadow: 0 0 0 2px #65b6dc, 0 5px 0 #14171e;
    }
    .center-rail {
        position: absolute;
        width: 126px;
        height: 12px;
        border: 1px solid #333640;
        border-radius: 5px;
        background: linear-gradient(#92949e, #60636d 35%, #4b4e58 70%, #727580);
        box-shadow: 0 2px 2px #171a2180;
        z-index: 1;
    }
    .center-rail-0 { bottom: 6px; left: 50%; transform: translateX(-50%); }
    .center-rail-1 { right: 6px; top: 50%; width: 12px; height: 126px; transform: translateY(-50%); background: linear-gradient(90deg, #727580, #4b4e58 30%, #60636d 65%, #92949e); }
    .center-rail-2 { top: 6px; left: 50%; transform: translateX(-50%); }
    .center-rail-3 { left: 6px; top: 50%; width: 12px; height: 126px; transform: translateY(-50%); background: linear-gradient(90deg, #92949e, #60636d 35%, #4b4e58 70%, #727580); }
    .center-click-zone {
        position: absolute;
        left: 50%;
        top: 42%;
        transform: translate(-50%, -50%);
        width: 12%;
        height: 16%;
        background: transparent;
        border: 0;
        border-radius: 12px;
        cursor: pointer;
        pointer-events: auto;
        z-index: 20;
    }
    .dora-tile-3d {
        width: 28px;
        height: 42px;
        position: relative;
        transform-style: preserve-3d;
    }

    /* Riichi sticks on table */
    .riichi-stick-3d {
        position: absolute;
        width: 108px;
        height: 9px;
        background: white;
        border-radius: 3px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
        z-index: 6;
        transform-style: preserve-3d;
    }
    .riichi-stick-3d .dot {
        width: 5px;
        height: 5px;
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
        gap: 24px;
        width: 100%;
        margin-left: 0;
    }
    .own-tiles-3d {
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
        gap: 2px;
        flex-shrink: 0;
    }
    /* Keep table projection while painting above the hand gradient (z:20). */
    .own-meld-layer {
        z-index: 25;
        pointer-events: none;
    }
    .own-meld-layer .table-surface,
    .own-meld-layer .table-inner {
        background: none;
        border-color: transparent;
        box-shadow: none;
    }
    .own-meld-layer .table-surface::before,
    .own-meld-layer .table-inner::before,
    .own-meld-layer .table-inner::after { display: none; }
    .own-melds-3d {
        position: absolute;
        pointer-events: auto;
        right: 16%;
        /* Project the tile bases onto the flat hand's bottom baseline. */
        bottom: 6.3%;
        transform-style: preserve-3d;
    }
    .own-melds-inner {
        justify-content: flex-start;
    }
    .own-tile-3d {
        width: 60px;
        height: 90px;
        position: relative;
        border-radius: 5px;
        overflow: hidden;
        background: var(--tile-ivory);
        flex-shrink: 0;
        box-shadow: 0 2px 4px #06172466;
        /* Show the complete atlas cell, fitted to the slightly shorter face. */
        box-sizing: border-box;
    }

    /* ========================================
       UI Overlay
       ======================================== */
    .player-panel-3d {
        position: absolute;
        pointer-events: auto;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        width: 116px;
        height: 92px;
        padding: 0;
        color: #edf3f6;
        font: inherit;
        text-align: center;
        background: transparent;
        border: 0;
        border-radius: 3px;
        cursor: pointer;
    }
    .player-panel-3d:disabled { cursor: default; opacity: 1; }
    .player-panel-3d .avatar-3d {
        flex: 0 0 64px;
        width: 64px;
        height: 64px;
        border-radius: 3px;
        overflow: hidden;
        background: #14191f;
        border: 3px solid #07090c;
        box-shadow: 0 1px 3px #0008;
        box-sizing: border-box;
    }
    .player-panel-3d:hover:not(:disabled) .avatar-3d { border-color: #394b58; }
    .player-panel-3d.is-viewpoint .avatar-3d { box-shadow: 0 0 0 1px #65b6dc, 0 2px 4px #0008; }
    .player-panel-3d .avatar-img { display: block; width: 100%; height: 100%; object-fit: cover; }
    .player-panel-3d .player-name {
        font-family: inherit;
        font-weight: 700;
        width: 100%;
        height: 24px;
        box-sizing: border-box;
        padding: 0 6px;
        font-size: 13px;
        line-height: 24px;
        color: #ffffff;
        background: #070b10e8;
        border-radius: 2px;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .kita-display-3d {
        position: absolute;
        right: 0;
        bottom: calc(100% + 45px);
        display: flex;
        align-items: center;
        gap: 8px;
        transform-style: preserve-3d;
        pointer-events: none;
    }
    .kita-display-3d.own-kita-3d { right: 16%; bottom: 20%; }
    .kita-count-3d {
        font: 700 22px/1 var(--panel-font, sans-serif);
        color: #fff;
        text-shadow: 0 1px 2px #000;
        transform: translateZ(24px);
    }

    .floating-score-3d {
        position: absolute;
        pointer-events: auto;
        font-family: inherit;
        font-variant-numeric: tabular-nums;
        font-size: 23px;
        font-weight: 700;
        color: #f8d55b;
        line-height: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 94px;
        height: 28px;
        padding: 0 4px;
        box-sizing: border-box;
        text-shadow: 0 1px 1px #11141b;
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
        padding: 5px 16px;
        background: rgba(0,0,0,0.6);
        border-radius: 8px;
        border: 2px solid white;
        z-index: 50;
        pointer-events: none;
        animation: popIn3d 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .call-overlay-3d.call-chii {
        background: ${COLORS.callChiiBg};
        color: white;
        text-shadow: 0 0 5px rgba(0,80,0,0.8), 0 0 10px #000;
        border-color: rgba(255,255,255,0.7);
    }
    .call-overlay-3d.call-pon {
        background: ${COLORS.callPonBg};
        color: white;
        text-shadow: 0 0 5px rgba(0,40,120,0.8), 0 0 10px #000;
        border-color: rgba(255,255,255,0.7);
    }
    .call-overlay-3d.call-kan {
        background: ${COLORS.callKanBg};
        color: white;
        text-shadow: 0 0 5px rgba(60,20,100,0.8), 0 0 10px #000;
        border-color: rgba(255,255,255,0.7);
    }
    .call-overlay-3d.call-reach {
        background: ${COLORS.callReachBg};
        color: white;
        text-shadow: 0 0 5px rgba(140,60,0,0.8), 0 0 10px #000;
        border-color: rgba(255,255,255,0.7);
    }
    .call-overlay-3d.call-hora {
        background: ${COLORS.callHoraBg};
        color: white;
        text-shadow: 0 0 5px rgba(120,0,0,0.8), 0 0 10px #000;
        border-color: rgba(255,255,255,0.7);
    }

    @keyframes popIn3d {
        0% { scale: 0.5; opacity: 0; }
        100% { scale: 1; opacity: 1; }
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
        gap: 5px;
        align-items: center;
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        box-shadow: var(--panel-shadow);
        color: #fff;
        padding: 6px 10px;
        border-radius: var(--panel-radius);
        font-family: var(--panel-font);
        font-size: 16px;
        font-weight: 600;
        white-space: nowrap;
        pointer-events: none;
        z-index: 40;
    }
    .wait-tile-3d {
        width: 28px;
        height: 42px;
        flex-shrink: 0;
    }

    /* Subtle woven surface and an inset playing boundary; no texture downloads. */
    .table-inner::before {
        content: '';
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: repeating-linear-gradient(0deg, #ffffff03 0 1px, transparent 1px 3px),
            repeating-linear-gradient(90deg, #00000006 0 1px, transparent 1px 3px);
    }
    .table-inner::after {
        content: '';
        position: absolute;
        inset: 95px;
        border: 2px solid #0a342b70;
        box-shadow: 0 0 0 1px #69a58e26;
        border-radius: 16px;
        pointer-events: none;
    }
    .scene-3d .tile-bg {
        background: linear-gradient(150deg, #fffdf7, #eee9dc);
        box-shadow: inset 0 0 0 1px #9b998f80, inset 0 2px 1px #fff;
        border-radius: 3px;
    }
    /* The PNG face includes the bevel; CSS supplies the box sides and shadows. */
    .scene-3d .tile-face-down .tile-bg {
        background: linear-gradient(100deg, #e8a843, #cf882b);
        box-shadow: inset 0 2px 0 #f5d195, inset 0 0 0 1px #986326;
    }
    .table-tile, .table-tile-rotated, .opp-tile, .opp-tile-rotated {
        box-shadow: 1px 3px 4px #03172466;
        border-radius: 3px;
    }
    .center-display {
        position: relative;
        display: flex;
        flex-direction: column;
        gap: 1px;
        align-items: center;
        justify-content: center;
        width: 104px;
        height: 92px;
        color: #65b6dc;
        background: #0c151b;
        border: 2px solid #424851;
        border-radius: 4px;
        box-shadow: inset 0 2px 5px #000b;
        z-index: 1;
    }
    .center-turn-indicator {
        position: absolute;
        z-index: 2;
        width: 126px;
        height: 8px;
        clip-path: polygon(9% 0, 91% 0, 100% 100%, 0 100%);
        background: linear-gradient(#fff0a3, #f8d55b 40%, #d9a52c);
        pointer-events: none;
        animation: center-turn-blink 1.2s ease-in-out infinite;
    }
    @keyframes center-turn-blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.28; }
    }
    .center-display strong { font-size: 22px; font-weight: 700; line-height: 1.3; }
    .wall-remaining { font-size: 44px; font-weight: 700; line-height: 1; color: #65b6dc; font-variant-numeric: tabular-nums; }
    .center-wind {
        position: absolute;
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        font-size: 27px;
        font-weight: 600;
        line-height: 1;
        color: #f2f4fa;
        border-radius: 3px;
        z-index: 1;
    }
    .center-wind::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, #4e525c, #323640);
        clip-path: polygon(0 0, 60% 0, 100% 40%, 100% 100%, 0 100%);
        border-radius: inherit;
        z-index: -1;
    }
    .center-wind-0 { bottom: 4px; left: 4px; }
    .center-wind-1 { bottom: 4px; right: 4px; transform: rotate(-90deg); }
    .center-wind-2 { top: 4px; right: 4px; transform: rotate(180deg); }
    .center-wind-3 { top: 4px; left: 4px; transform: rotate(90deg); }
    .center-wind.dealer { color: #23171c; }
    .center-wind.dealer::before { background: linear-gradient(135deg, #ed696c, #b52f3f); }
    .dora-panel-3d {
        position: absolute;
        top: 20px;
        left: 24px;
        padding: 8px;
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: var(--panel-radius);
        box-shadow: var(--panel-shadow);
        color: var(--panel-text);
    }
    .dora-tiles { display: flex; gap: 2px; }
    .dora-marker { width: 36px; height: 54px; border-radius: 3px; box-shadow: 0 2px #ded6c5, 1px 4px #b97924, 1px 6px 4px #06172499; }
    .table-counters { display: flex; justify-content: space-around; gap: 12px; margin-top: 8px; color: #f4f6f8; font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .table-counter { display: inline-flex; align-items: center; gap: 3px; white-space: nowrap; }
    .table-counter svg { display: block; width: 32px; height: 32px; flex-shrink: 0; }
    .scene-3d button:focus-visible, .replay-dock :focus-visible {
        outline: 2px solid #65b6dc;
        outline-offset: 3px;
    }
    .viewer-settings-button {
        position: absolute;
        top: 20px;
        right: 24px;
        width: 56px;
        height: 56px;
        display: grid;
        place-items: center;
        padding: 0;
        pointer-events: auto;
        cursor: pointer;
        color: #f2f4fa;
        background: #50545e;
        border: 1px solid #a2a5ae;
        border-radius: var(--panel-radius);
        box-shadow: var(--panel-shadow), inset 0 0 0 1px #ffffff18;
    }
    .viewer-settings-button:hover { color: #fff; background: #626772; border-color: #d3d6df; }
    .viewer-settings-button svg { width: 30px; height: 30px; }
    /* Size against the actual viewer, including narrow embeds on a wide page. */
    .re-settings-overlay { padding: 8px; background: #101217b8; container-type: size; container-name: viewer-settings; }
    .settings-dialog {
        --settings-inset: clamp(12px, 3cqw, 20px);
        --settings-gap: clamp(3px, 1cqh, 6px);
        width: min(460px, 92cqw, max(300px, 80cqh));
        max-width: 100%;
        max-height: min(640px, 94cqh, 100%);
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-sizing: border-box;
        color: var(--panel-text);
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: var(--panel-radius);
        box-shadow: var(--panel-shadow);
        font: clamp(12px, 2.5cqh, 13px)/1.4 -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN", "Yu Gothic", sans-serif;
        text-align: left;
    }
    .settings-header { display: flex; align-items: center; justify-content: space-between; flex: 0 0 auto; padding: var(--settings-gap) var(--settings-inset); border-bottom: 1px solid var(--panel-divider); }
    .settings-header h2 { margin: 0; font-size: 1.4em; font-weight: 700; color: #fff; }
    .settings-dialog button { font: inherit; cursor: pointer; }
    .settings-close { width: 28px; height: 28px; padding: 0; color: var(--panel-muted); background: transparent; border: 1px solid var(--panel-border); border-radius: 5px; font-size: 21px !important; line-height: 1 !important; }
    .settings-close:hover { background: var(--panel-raised); color: white; }
    .settings-body { min-height: 0; overflow-y: auto; overscroll-behavior: contain; padding: 0 var(--settings-inset); scrollbar-width: thin; scrollbar-gutter: stable; }
    .settings-row { position: relative; min-width: 0; margin: 0; padding: var(--settings-gap) 0 var(--settings-gap) 44%; border: 0; border-bottom: 1px solid var(--panel-divider); }
    .settings-row legend { position: absolute; top: 50%; left: 0; transform: translateY(-50%); padding: 0; font-weight: 600; }
    .settings-language { box-sizing: border-box; width: 100%; min-width: 0; height: 34px; padding: 4px 8px; font: inherit; color: var(--panel-text); background: var(--panel-inset); border: 1px solid var(--panel-border); border-radius: 6px; cursor: pointer; }
    .settings-options { display: flex; gap: 2px; padding: 2px; background: var(--panel-inset); border: 1px solid var(--panel-border); border-radius: 6px; }
    .settings-options label { position: relative; flex: 1; min-width: 0; cursor: pointer; }
    .settings-options input { position: absolute; opacity: 0; width: 1px; height: 1px; }
    .settings-options span { display: grid; place-items: center; box-sizing: border-box; min-height: 28px; padding: 3px 6px; text-align: center; white-space: nowrap; border-radius: 3px; color: var(--panel-muted); }
    .settings-options input:checked + span { color: #17191f; background: #65b6dc; font-weight: 600; }
    .settings-options input:focus-visible + span { outline: 2px solid #b5e5ff; outline-offset: 2px; }
    .settings-load { min-width: 0; margin: var(--settings-gap) 0 0; padding: 0 0 var(--settings-gap); border: 0; }
    .settings-load legend { padding: 0; margin-bottom: 4px; font-weight: 600; }
    .settings-input-label { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; color: var(--panel-muted); font-size: 12px; }
    .settings-input-label[hidden] { display: none; }
    .settings-input-label input { box-sizing: border-box; width: 100%; min-width: 0; height: 32px; padding: 4px 8px; background: var(--panel-inset); color: var(--panel-text); border: 1px solid var(--panel-border); border-radius: 5px; font: inherit; font-size: 13px; }
    .settings-input-label input::placeholder { color: #979ca8; }
    .settings-input-label input[type=file] { padding: 2px; font-size: 12px; }
    .settings-input-label input::file-selector-button { padding: 3px 8px; margin-right: 8px; border: 0; border-radius: 3px; background: var(--panel-raised); color: var(--panel-text); font: inherit; cursor: pointer; }
    .settings-load-button { display: block; margin: 0; min-height: 28px; padding: 4px 16px; border: 1px solid var(--panel-border); border-radius: 5px; background: var(--panel-inset); color: #a6aab5; }
    .settings-dialog button:disabled { opacity: 0.55; cursor: default; }
    .settings-load-button:not(:disabled) { background: #65b6dc; color: #17191f; cursor: pointer; }
    .settings-load-status { margin: 4px 0 8px; color: var(--panel-muted); overflow-wrap: anywhere; }
    .settings-load-status.is-error { color: #ffb0b7; }
    .settings-footer { display: flex; justify-content: flex-end; gap: 8px; flex: 0 0 auto; padding: var(--settings-gap) var(--settings-inset); border-top: 1px solid var(--panel-divider); }
    .settings-done { min-height: 28px; padding: 4px 20px; border: 1px solid #65b6dc; border-radius: 5px; background: #65b6dc; color: #17191f; font-weight: 600 !important; }
    .settings-done:hover { background: #91cff0; }
    .settings-dialog button:focus-visible, .settings-input-label input:focus-visible, .settings-language:focus-visible { outline: 2px solid #91cff0; outline-offset: 3px; }
    @container viewer-settings (max-width: 280px) {
        .settings-row { padding-left: 0; }
        .settings-row legend { position: static; transform: none; padding-top: var(--settings-gap); margin-bottom: 4px; }
    }
    @container viewer-settings (max-height: 440px) {
        .settings-dialog { --settings-gap: 2px; max-height: 100%; }
        .settings-load legend { margin-bottom: 2px; }
        .settings-input-label { margin-top: 4px; }
        .settings-input-label > span { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
    }
    @media (pointer: coarse) {
        .settings-language { height: 40px; }
        .settings-options span, .settings-close, .settings-done, .settings-load-button { min-height: 34px; }
        .settings-close { min-width: 34px; }
        .settings-input-label input { height: 34px; font-size: 16px; }
    }
    /* The scaled board is a separate stacking context from the floating dock.
       Raise that context while a result modal is open, including its backdrop. */
    .viewer-stage:has(.re-modal-overlay) { z-index: 50; }

    .replay-dock {
        box-sizing: border-box;
        width: 248px;
        max-width: 100%;
        margin: 0 auto;
        padding: 6px;
        color: var(--panel-text);
        background: var(--panel-bg);
        border: 1px solid var(--panel-border);
        border-radius: var(--panel-radius);
        font: 12px -apple-system, BlinkMacSystemFont, "Hiragino Kaku Gothic ProN", sans-serif;
    }
    .replay-dock.is-floating {
        position: absolute;
        z-index: 40;
        margin: 0;
        transform-origin: top left;
        box-shadow: var(--panel-shadow);
    }
    .replay-actions { display: grid; grid-template-columns: 1fr 1fr; align-items: center; gap: 4px 8px; }
    .replay-playback { grid-column: 1 / -1; }
    .replay-group { display: flex; align-items: center; justify-content: center; gap: 4px; min-width: 0; }
    .replay-group-label { color: var(--panel-muted); margin-right: 2px; font-size: 11px; }
    .replay-dock .icon-btn { width: 32px; height: 32px; padding: 4px; border-radius: 4px; flex-shrink: 0; color: var(--panel-text); }
    .replay-dock .icon-btn:hover:not(:disabled) { background: var(--panel-raised); color: var(--panel-text); }
    .replay-dock .icon-btn.active-btn { background: var(--panel-raised) !important; color: #a5dcf4 !important; box-shadow: inset 0 0 0 1px #65b6dc; }
    .replay-dock .icon-btn svg { width: 20px !important; height: 20px !important; }
    .replay-dock .replay-play { background: #65b6dc; color: #17191f; width: 44px; }
    .replay-dock .replay-play:hover:not(:disabled) { background: #91cff0; color: #17191f; }
    .replay-dock .replay-play.active-btn { background: #91cff0 !important; color: #17191f !important; box-shadow: none; }
    .replay-dock .icon-btn:disabled { opacity: 0.3; cursor: default; }
    @media (prefers-reduced-motion: reduce) {
        .scene-3d *, .replay-dock * { animation: none !important; transition: none !important; }
    }
`;
