import { GameState } from './game_state';
import { Renderer } from './renderer';
import { MjaiEvent } from './types';

export class Viewer {
    gameState: GameState;
    renderer: Renderer;
    container: HTMLElement;
    log: MjaiEvent[];

    debugPanel!: HTMLElement;
    // slider!: HTMLInputElement; // Removed

    constructor(containerId: string, log: MjaiEvent[]) {
        const el = document.getElementById(containerId);
        if (!el) throw new Error(`Container #${containerId} not found`);
        this.container = el;
        this.log = log;

        // Setup DOM Structure: 2-Column Flex (Main Area + Right Sidebar)
        // Reset Body / HTML to prevent default margins causing scrollbars
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.overflow = 'hidden';
        document.body.style.height = '100vh';
        document.body.style.width = '100vw';
        document.documentElement.style.margin = '0';
        document.documentElement.style.padding = '0';
        document.documentElement.style.overflow = 'hidden';
        document.documentElement.style.height = '100vh';
        document.documentElement.style.width = '100vw';

        this.container.innerHTML = '';
        Object.assign(this.container.style, {
            display: 'flex',
            flexDirection: 'row',
            width: '100%',
            height: '100vh',
            maxHeight: '768px', // Force Max Height
            backgroundColor: '#000',
            overflow: 'hidden',
            margin: '0',
            padding: '0'
        });

        // Scrollable Main Content Area
        const scrollContainer = document.createElement('div');
        Object.assign(scrollContainer.style, {
            flex: '1',
            height: '100%',
            position: 'relative',
            overflow: 'hidden', // Disable scrolling entirely
            backgroundColor: '#000',
            display: 'flex',       // Center child
            alignItems: 'flex-start', // Top Align
            justifyContent: 'center'
        });
        this.container.appendChild(scrollContainer);

        // 1. Board Wrapper (Main Area)
        const boardWrapper = document.createElement('div');
        boardWrapper.id = this.container.id + '-board-wrapper';
        Object.assign(boardWrapper.style, {
            width: '100%',
            height: '100%', // Full size
            position: 'relative',
            backgroundColor: '#000',
            overflow: 'hidden'
        });
        scrollContainer.appendChild(boardWrapper);

        // The View Area - Fixed Base Size 900x900
        const viewArea = document.createElement('div');
        viewArea.id = `${containerId}-board`;
        Object.assign(viewArea.style, {
            width: '900px',
            height: '900px',
            position: 'absolute',
            top: '0',        // Top Align
            left: '50%',     // Horizontal Center
            transform: 'translateX(-50%)', // Initial horizontal center only
            transformOrigin: 'top center', // Scale from Top
            backgroundColor: '#2d5a27', // Board background
            boxShadow: '0 0 20px rgba(0,0,0,0.5)'
        });
        boardWrapper.appendChild(viewArea);        // 2. Right Sidebar (Controls) - Fixed Width
        const rightSidebar = document.createElement('div');
        Object.assign(rightSidebar.style, {
            width: '80px', // Fixed width for icons
            backgroundColor: '#111',
            borderLeft: '1px solid #333',
            display: 'flex',
            flexDirection: 'column',
            gap: '15px',
            padding: '20px 10px',
            paddingTop: '50px',
            alignItems: 'center',
            flexShrink: '0',
            zIndex: '500',
            height: '100%', // Full height
            boxSizing: 'border-box', // Ensure padding doesn't overflow height
            overflow: 'hidden'       // Prevent any scrollbars
        });
        this.container.appendChild(rightSidebar);

        this.debugPanel = document.createElement('div');
        this.debugPanel.className = 'debug-panel';
        // Debug Panel in boardWrapper so it scrolls with board
        Object.assign(this.debugPanel.style, {
            position: 'absolute',
            top: '0',
            left: '0',
            width: '100%',
            zIndex: '1000'
        });
        boardWrapper.appendChild(this.debugPanel);

        // Helper to create buttons
        const createBtn = (text: string, onClick: () => void) => {
            const btn = document.createElement('div');
            btn.className = 'icon-btn';
            btn.textContent = text;
            btn.onclick = onClick;
            return btn;
        };

        console.log("[Viewer] Initializing GameState with log length:", log.length);
        this.gameState = new GameState(log);
        console.log("[Viewer] GameState initialized. Current event index:", this.gameState.current.eventIndex);

        console.log("[Viewer] Initializing Renderer");
        this.renderer = new Renderer(viewArea);

        // Show/Hide Log
        const logBtn = createBtn('📜', () => {
            const display = this.debugPanel.style.display;
            if (display === 'none' || !display) {
                this.debugPanel.style.display = 'block';
                logBtn.classList.add('active-btn');
            } else {
                this.debugPanel.style.display = 'none';
                logBtn.classList.remove('active-btn');
            }
        });
        rightSidebar.appendChild(logBtn);

        // Prev Kyoku
        rightSidebar.appendChild(createBtn('⏮️', () => {
            if (this.gameState.jumpToPrevKyoku()) this.update();
        }));

        // Next Kyoku
        rightSidebar.appendChild(createBtn('⏭️', () => {
            if (this.gameState.jumpToNextKyoku()) this.update();
        }));

        // Prev Step
        rightSidebar.appendChild(createBtn('◀️', () => {
            if (this.gameState.stepBackward()) this.update();
        }));

        // Next Step
        rightSidebar.appendChild(createBtn('▶️', () => {
            if (this.gameState.stepForward()) this.update();
        }));

        // Auto Play
        let autoPlayTimer: number | null = null;
        const autoBtn = createBtn('⏯️', () => {
            if (autoPlayTimer) {
                clearInterval(autoPlayTimer);
                autoPlayTimer = null;
                autoBtn.classList.remove('active-btn');
            } else {
                autoBtn.classList.add('active-btn');
                autoPlayTimer = window.setInterval(() => {
                    if (!this.gameState.stepForward()) {
                        if (autoPlayTimer) clearInterval(autoPlayTimer);
                        autoPlayTimer = null;
                        autoBtn.classList.remove('active-btn');
                    } else {
                        this.update();
                    }
                }, 200); // 200ms per step
            }
        });
        rightSidebar.appendChild(autoBtn);

        // Robust Responsive Scaling (Contain Strategy)
        // With MAX_WIDTH = 768px and MAX_HEIGHT = 768px constraint
        const MAX_WIDTH = 768;
        const MAX_HEIGHT = 768;

        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                if (width === 0 || height === 0) continue;

                const availableWidth = Math.min(width, MAX_WIDTH);
                const availableHeight = Math.min(height, MAX_HEIGHT);

                // Scale to fit smaller dimension
                // We want to fit 900x900 into availableWidth x availableHeight
                const scale = Math.min(availableWidth / 900, availableHeight / 900);

                viewArea.style.transform = `translateX(-50%) scale(${scale})`;
            }
        });
        resizeObserver.observe(boardWrapper);

        // Initial call
        setTimeout(() => {
            // Manually trigger resize observer for initial layout
            boardWrapper.getBoundingClientRect(); // accessing layout properties can flush layout
            resizeObserver.observe(boardWrapper); // Re-observe to trigger
            resizeObserver.disconnect(); // Disconnect to avoid duplicate triggers
            resizeObserver.observe(boardWrapper); // Re-observe for future changes
        }, 0);

        // Handle Viewpoint Change from Renderer (Click on Player Info)
        this.renderer.onViewpointChange = (pIdx: number) => {
            if (this.renderer.viewpoint !== pIdx) {
                this.renderer.viewpoint = pIdx;
                this.update();
            }
        };

        console.log("[Viewer] Calling first update()");
        this.update();

        // Handle Center Click -> Show Round Selector
        this.renderer.onCenterClick = () => {
            this.showRoundSelector();
        };

        // Mouse Wheel Navigation (Turn by Turn)
        // Mouse Wheel Navigation (Turn by Turn)
        let lastWheelTime = 0;
        const WHEEL_THROTTLE_MS = 100; // Limit updates to 10/sec

        boardWrapper.addEventListener('wheel', (e) => {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            const now = Date.now();
            if (now - lastWheelTime < WHEEL_THROTTLE_MS) return;
            lastWheelTime = now;

            console.log("[Viewer] Wheel event detected", e.deltaY);

            // Use current viewpoint for turn navigation
            const vp = this.renderer.viewpoint;

            if (e.deltaY > 0) {
                // Scroll Down -> Next Turn (Next Tsumo for Viewpoint)
                console.log("[Viewer] Jumping to next turn");
                if (this.gameState.jumpToNextTurn(vp)) this.update();
            } else {
                // Scroll Up -> Prev Turn (Prev Tsumo for Viewpoint)
                console.log("[Viewer] Jumping to prev turn");
                if (this.gameState.jumpToPrevTurn(vp)) this.update();
            }
        }, { passive: false });
    }

    showRoundSelector() {
        // Create Modal Overlay
        const overlay = document.createElement('div');
        overlay.className = 're-modal-overlay';
        overlay.onclick = () => {
            overlay.remove();
        };

        const content = document.createElement('div');
        content.className = 're-modal-content';
        content.onclick = (e) => e.stopPropagation();

        const title = document.createElement('h3');
        title.textContent = 'Jump to Round';
        title.className = 're-modal-title';
        title.style.marginTop = '0';
        content.appendChild(title);

        const table = document.createElement('table');
        table.className = 're-kyoku-table';

        // Header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr>
                <th>Round</th>
                <th>Honba</th>
                <th>P0 Score</th>
                <th>P1 Score</th>
                <th>P2 Score</th>
                <th>P3 Score</th>
            </tr>
        `;
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');
        const checkpoints = this.gameState.getKyokuCheckpoints();

        checkpoints.forEach((cp) => {
            const tr = document.createElement('tr');
            tr.className = 're-kyoku-row';
            tr.onclick = () => {
                this.gameState.jumpTo(cp.index + 1);
                this.update();
                overlay.remove();
            };

            const scores = cp.scores || [0, 0, 0, 0];
            tr.innerHTML = `
                <td>${this.renderer.formatRound(cp.round)}</td>
                <td>${cp.honba}</td>
                <td>${scores[0]}</td>
                <td>${scores[1]}</td>
                <td>${scores[2]}</td>
                <td>${scores[3]}</td>
            `;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        content.appendChild(table);

        overlay.appendChild(content);

        // Append to CONTAINER (Not viewArea) to ensure it covers the whole widget
        // and isn't affected by board scaling/transform
        this.container.appendChild(overlay);
    }

    update() {
        console.log("[Viewer] update() called");
        try {
            console.log("[Viewer] Invoking renderer.render...");
            this.renderer.render(this.gameState.current, this.debugPanel);
            console.log("[Viewer] renderer.render completed");
        } catch (e: any) {
            console.error("Render Error:", e);
            if (this.debugPanel) {
                this.debugPanel.style.display = 'block';
                this.debugPanel.innerHTML += `<div style="color: red; border-top: 1px solid #555; margin-top: 10px; padding-top: 10px;">
                    <strong>Render Error:</strong><br>
                    ${e.message}<br>
                    <pre>${e.stack}</pre>
                </div>`;
            }
        }
    }
}

// Global Export for usage in HTML
// @ts-ignore
window.RiichiEnvViewer = Viewer;
