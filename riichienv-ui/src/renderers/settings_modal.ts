import { I18n, isLocale, type Locale, locales } from '../i18n/index';
import { ReplayLoadError } from '../replay_loader';
import { DEFAULT_DISPLAY_OPTIONS, type DisplayOptions } from './board_presentation';

let instance = 0;

/** Settings apply immediately and preserve the current replay position. */
export function createSettingsModal(
    onClose: () => void,
    i18n = new I18n(),
    onLanguageChange?: (locale: Locale) => void,
    onLoadUrl?: (url: string) => Promise<void>,
    onLoadFile?: (file: File) => Promise<void>,
    displayOptions: Readonly<DisplayOptions> = DEFAULT_DISPLAY_OPTIONS,
    onDisplayOptionsChange?: (options: DisplayOptions) => void,
): HTMLElement {
    const id = `viewer-settings-${++instance}`;
    const overlay = document.createElement('div');
    overlay.className = 're-modal-overlay re-settings-overlay';
    overlay.innerHTML = `
        <section class="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="${id}-title">
            <header class="settings-header">
                <h2 id="${id}-title" data-i18n="Settings">設定</h2>
                <button type="button" class="settings-close" data-i18n-label="Close settings" aria-label="設定を閉じる">×</button>
            </header>
            <div class="settings-body">
                <fieldset class="settings-row">
                    <legend data-i18n="Language">表示言語</legend>
                    <select class="settings-language" name="${id}-language" data-i18n-label="Language">
                        ${Object.entries(locales)
                            .map(
                                ([locale, data]) =>
                                    `<option value="${locale}" lang="${locale}" ${locale === i18n.locale ? 'selected' : ''}>${data.name}</option>`,
                            )
                            .join('')}
                    </select>
                </fieldset>
                <fieldset class="settings-row">
                    <legend data-i18n="Opponent hands">相手の手牌表示</legend>
                    <div class="settings-options">
                        <label><input type="radio" name="${id}-opponents" value="show" checked><span data-i18n="Show">表示</span></label>
                        <label><input type="radio" name="${id}-opponents" value="hide"><span data-i18n="Hide">非表示</span></label>
                    </div>
                </fieldset>
                <fieldset class="settings-row">
                    <legend data-i18n="Wait tiles">待ち牌表示</legend>
                    <div class="settings-options">
                        <label><input type="radio" name="${id}-waits" value="show" checked><span data-i18n="Show">表示</span></label>
                        <label><input type="radio" name="${id}-waits" value="hide"><span data-i18n="Hide">非表示</span></label>
                    </div>
                </fieldset>
                <fieldset class="settings-load">
                    <legend data-i18n="Load replay">牌譜のロード</legend>
                    <div class="settings-options settings-source">
                        <label><input type="radio" name="${id}-source" value="url" checked><span data-i18n="URL">URL指定</span></label>
                        <label><input type="radio" name="${id}-source" value="file"><span data-i18n="Choose file">ファイル選択</span></label>
                    </div>
                    <label class="settings-input-label" data-source="url">
                        <span data-i18n="Replay URL">牌譜のURL</span>
                        <input type="url" placeholder="https://…" autocomplete="off" spellcheck="false">
                    </label>
                    <label class="settings-input-label" data-source="file" hidden>
                        <span data-i18n="Replay file">牌譜ファイル</span>
                        <input type="file" accept=".jsonl,.jsonl.gz">
                    </label>
                </fieldset>
                <p class="settings-load-status" role="status" aria-live="polite" hidden></p>
            </div>
            <footer class="settings-footer">
                <button type="button" class="settings-load-button" disabled data-i18n="Load">読み込む</button>
                <button type="button" class="settings-done" data-i18n="Close">閉じる</button>
            </footer>
        </section>`;

    i18n.apply(overlay);
    let currentDisplayOptions = { ...displayOptions };
    for (const [group, key] of [
        ['opponents', 'showOpponentHands'],
        ['waits', 'showWaits'],
    ] as const) {
        overlay.querySelectorAll<HTMLInputElement>(`input[name="${id}-${group}"]`).forEach((radio) => {
            radio.checked = (radio.value === 'show') === currentDisplayOptions[key];
            radio.onchange = () => {
                if (!radio.checked) return;
                currentDisplayOptions = { ...currentDisplayOptions, [key]: radio.value === 'show' };
                onDisplayOptionsChange?.(currentDisplayOptions);
            };
        });
    }
    const url = overlay.querySelector<HTMLInputElement>('input[type=url]')!;
    const file = overlay.querySelector<HTMLInputElement>('input[type=file]')!;
    const sources = overlay.querySelectorAll<HTMLInputElement>('.settings-source input');
    const load = overlay.querySelector<HTMLButtonElement>('.settings-load-button')!;
    const status = overlay.querySelector<HTMLElement>('.settings-load-status')!;
    let loading = false;
    const refreshLoad = () => {
        const source = overlay.querySelector<HTMLInputElement>('.settings-source input:checked')!.value;
        const available = source === 'file' ? !!onLoadFile && !!file.files?.length : !!onLoadUrl && !!url.value.trim();
        load.disabled = loading || !available;
        url.disabled = file.disabled = loading;
        sources.forEach((radio) => {
            radio.disabled = loading;
        });
    };
    const inputChanged = () => {
        status.hidden = true;
        refreshLoad();
    };
    url.oninput = inputChanged;
    file.onchange = inputChanged;
    load.onclick = async () => {
        if (load.disabled) return;
        const source = overlay.querySelector<HTMLInputElement>('.settings-source input:checked')!.value;
        const selectedFile = file.files?.[0];
        const performLoad =
            source === 'file'
                ? selectedFile && onLoadFile && (() => onLoadFile(selectedFile))
                : onLoadUrl && (() => onLoadUrl(url.value));
        if (!performLoad) return;
        loading = true;
        refreshLoad();
        status.hidden = false;
        status.dataset.i18n = 'Loading replay…';
        status.classList.remove('is-error');
        overlay.querySelector('.settings-dialog')?.setAttribute('aria-busy', 'true');
        i18n.apply(overlay);
        try {
            await performLoad();
            if (overlay.isConnected) onClose();
        } catch (error) {
            if (!overlay.isConnected) return;
            status.dataset.i18n =
                error instanceof ReplayLoadError ? error.message : 'Replay loading failed. Please try again.';
            status.classList.add('is-error');
            i18n.apply(overlay);
        } finally {
            loading = false;
            overlay.querySelector('.settings-dialog')?.removeAttribute('aria-busy');
            refreshLoad();
        }
    };
    url.onkeydown = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            load.click();
        }
    };
    const language = overlay.querySelector<HTMLSelectElement>('.settings-language')!;
    language.onchange = () => {
        if (!isLocale(language.value)) return;
        if (onLanguageChange) onLanguageChange(language.value);
        else i18n.locale = language.value;
        i18n.apply(overlay);
    };
    overlay.querySelectorAll<HTMLButtonElement>('.settings-close, .settings-done').forEach((button) => {
        button.onclick = onClose;
    });
    overlay.onclick = (event) => {
        if (event.target === overlay) onClose();
    };
    overlay.onwheel = (event) => event.stopPropagation();
    sources.forEach((radio) => {
        radio.onchange = () => {
            overlay.querySelectorAll<HTMLElement>('[data-source]').forEach((pane) => {
                pane.hidden = pane.dataset.source !== radio.value;
            });
            inputChanged();
        };
    });
    overlay.onkeydown = (event) => {
        event.stopPropagation();
        if (event.key === 'Escape') {
            event.preventDefault();
            onClose();
        } else if (event.key === 'Tab') {
            const focusable = [
                ...overlay.querySelectorAll<HTMLElement>(
                    'button:not(:disabled), input:not(:disabled), select:not(:disabled)',
                ),
            ].filter(
                (element) =>
                    element.getClientRects().length > 0 &&
                    (!(element instanceof HTMLInputElement) || element.type !== 'radio' || element.checked),
            );
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first?.focus();
            }
        }
    };
    return overlay;
}
