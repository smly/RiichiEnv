export type { Locale } from './i18n/index';

import { LiveViewer } from './live_viewer';
import { RiichiViewer } from './riichi_viewer';
import { Viewer } from './viewer';
import { Viewer3D } from './viewer_3d';

export { Viewer, Viewer3D, LiveViewer, RiichiViewer };
export type {
    KyokuInfo,
    KyokuKeyEvent,
    KyokuPlayerAction,
    KyokuResult,
    KyokuSummary,
    KyokuWinner,
    PlayerConfig,
    ViewerEventMap,
    ViewerOptions,
    ViewerPosition,
} from './types';

if (typeof window !== 'undefined') {
    (window as any).RiichiEnvViewer = Viewer;
    (window as any).RiichiEnv3DViewer = Viewer3D;
    (window as any).RiichiEnvLiveViewer = LiveViewer;
    (window as any).RiichiViewer = RiichiViewer;
}
