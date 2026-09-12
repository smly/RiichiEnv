// Keep the analysis API's existing English yaku names stable.
export { yakuEn as YAKU_MAP } from './i18n/yaku';

export const COLORS = {
    boardBackground: '#184e47',
    modalBackground: '#0f1744',
    centerInfoBackground: '#13151b',

    highlightBoard: '#0f1744',
    highlightButton: '#2a3c85',
    activeButtonBg: '#0f1744',

    tableBorder: '#2a3c85',
    text: '#ffffff',
    activePlayerBar: '#ffd700',

    riverContainer: '#fff',

    callChiiBg: 'rgba(34, 139, 34, 0.85)',
    callPonBg: 'rgba(30, 80, 180, 0.85)',
    callKanBg: 'rgba(120, 50, 180, 0.85)',
    callReachBg: 'rgba(220, 120, 20, 0.90)',
    callHoraBg: 'rgba(200, 30, 30, 0.90)',
    callDefaultBg: 'rgba(0, 0, 0, 0.6)',
};

export const CALL_TYPES: { [key: string]: { label: string; cssClass?: string } } = {
    chi: { label: 'Chii', cssClass: 'call-chii' },
    pon: { label: 'Pon', cssClass: 'call-pon' },
    kan: { label: 'Kan', cssClass: 'call-kan' },
    ankan: { label: 'Kan', cssClass: 'call-kan' },
    daiminkan: { label: 'Kan', cssClass: 'call-kan' },
    kakan: { label: 'Kan', cssClass: 'call-kan' },
    reach: { label: 'Riichi', cssClass: 'call-reach' },
    kita: { label: 'Pei', cssClass: 'call-kan' },
};
