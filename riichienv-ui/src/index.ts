import { Viewer } from './viewer';
import { LiveViewer } from './live_viewer';

export { Viewer, LiveViewer };

(window as any).RiichiEnvViewer = Viewer;
(window as any).RiichiEnvLiveViewer = LiveViewer;
