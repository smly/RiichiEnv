import { EventJournal, parseJournalEvent } from './event_journal';
import { LiveViewer } from './live_viewer';
import { RiichiViewer } from './riichi_viewer';
import { Viewer } from './viewer';
import { Viewer3D } from './viewer_3d';
import { initWasm, isWasmReady } from './wasm/loader';

export { EventJournal, parseJournalEvent, Viewer, Viewer3D, LiveViewer, RiichiViewer, initWasm, isWasmReady };
export type {
    AppendOnlyEventSink,
    CompletedKyokuSpan,
    EventCursor,
    JournalDelta,
    JournalDeltaWire,
    KyokuKey,
} from './event_journal';
export type {
    KyokuInfo,
    KyokuKeyEvent,
    KyokuPlayerAction,
    KyokuResult,
    KyokuSummary,
    KyokuWinner,
    MjaiEvent,
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
    (window as any).RiichiEnvEventJournal = EventJournal;
}
