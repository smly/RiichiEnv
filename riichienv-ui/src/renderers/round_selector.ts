import { I18n } from '../i18n/index';
import type { KyokuSummary } from '../types';

function node<K extends keyof HTMLElementTagNameMap>(
    tag: K,
    className: string,
    text?: string,
): HTMLElementTagNameMap[K] {
    const element = document.createElement(tag);
    element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
}
function badge(text: string, kind: string) {
    return node('span', `round-badge ${kind}`, text);
}

export function createRoundSelector(options: {
    i18n?: I18n;
    summaries: KyokuSummary[];
    names: string[];
    playerCount: number;
    currentIndex: number;
    onSelect: (index: number) => void;
    onClose: () => void;
}): HTMLDivElement {
    const { summaries, names, playerCount: pc, currentIndex, onSelect, onClose } = options;
    const i18n = options.i18n ?? new I18n();
    const number = (value: number) => value.toLocaleString(i18n.locale);
    const content = node('div', 're-modal-content round-browser');
    content.setAttribute('role', 'dialog');
    content.setAttribute('aria-modal', 'true');
    content.setAttribute('aria-label', i18n.text('Jump to round'));
    const close = node('button', 'round-browser-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', i18n.text('Close round selector'));
    close.onclick = onClose;
    const tools = node('div', 'round-browser-tools');
    tools.append(node('span', 'round-browser-title', i18n.text('Jump to round')), close);
    content.append(tools);
    const scroller = node('div', 'round-browser-scroll');
    const table = node('table', 'round-browser-table');
    table.setAttribute('role', 'table');
    const caption = node('caption', 'round-browser-caption', i18n.text('Round summary'));
    table.append(caption);
    const head = node('thead', '');
    head.setAttribute('role', 'rowgroup');
    const headRow = node('tr', '');
    headRow.setAttribute('role', 'row');
    for (const label of [
        i18n.text('Round / Honba'),
        ...Array.from({ length: pc }, (_, p) => names[p] || `Player${p}`),
    ]) {
        const th = node('th', '', label);
        th.scope = 'col';
        th.setAttribute('role', 'columnheader');
        th.title = label;
        headRow.append(th);
    }
    head.append(headRow);
    table.append(head);
    const body = node('tbody', '');
    body.setAttribute('role', 'rowgroup');
    summaries.forEach((summary, index) => {
        const row = node('tr', `round-browser-row${index === currentIndex ? ' is-current' : ''}`);
        row.setAttribute('role', 'row');
        const round = i18n.round(summary.round, pc);
        const dealer = summary.dealer ?? summary.round % pc;
        const seat = (p: number) => i18n.wind((p - dealer + pc) % pc, true);
        const roundCell = node('th', 'round-browser-round');
        roundCell.scope = 'row';
        roundCell.setAttribute('role', 'rowheader');
        const select = node('button', 're-round-choice');
        select.type = 'button';
        select.setAttribute('aria-label', i18n.text('Jump to {round}, {honba} honba', { round, honba: summary.honba }));
        if (index === currentIndex) select.setAttribute('aria-current', 'true');
        select.append(
            node('strong', '', round),
            node('span', '', i18n.text('{count} honba', { count: summary.honba })),
        );
        roundCell.append(select);
        row.append(roundCell);
        row.onclick = () => onSelect(index);
        if (summary.result?.type === 'ryukyoku') {
            const draw = badge(i18n.text('Ryukyoku'), 'is-draw');
            draw.title = i18n.reason(summary.result.reason);
            select.append(draw);
        } else if (!summary.result) {
            select.append(node('span', '', i18n.text(summary.completed ? 'No result' : 'In progress')));
        }
        summary.playerActions.forEach((action, p) => {
            const cell = node('td', `round-player${action.hora ? ' is-winner' : action.houjuu ? ' is-deal-in' : ''}`);
            cell.setAttribute('role', 'cell');
            const playerName = node('span', 'round-player-name', names[p] || `Player${p}`);
            playerName.setAttribute('aria-hidden', 'true');
            cell.append(playerName);
            const status = node('div', 'round-player-status');
            status.append(node('span', `round-seat${p === dealer ? ' is-dealer' : ''}`, seat(p)));
            if (action.hora) status.append(badge(i18n.text(action.tsumo ? 'Tsumo' : 'Ron'), 'is-win'));
            const win = summary.result?.winners?.find((winner) => winner.actor === p);
            if (win?.limit) {
                const label = i18n.limit(win.limit);
                const limit = badge(label, `is-limit${win.limit.includes('yakuman') ? ' is-yakuman' : ''}`);
                limit.title = label;
                status.append(limit);
            }
            if (action.houjuu) status.append(badge(i18n.text('Deal-in'), 'is-loss'));
            if (action.riichi) {
                const riichi = badge('', 'is-riichi');
                riichi.setAttribute('role', 'img');
                riichi.setAttribute('aria-label', i18n.text('Riichi'));
                riichi.title = i18n.text('Riichi');
                status.append(riichi);
            }
            if (summary.result?.type === 'ryukyoku' && action.tenpai)
                status.append(badge(i18n.text('Tenpai'), 'is-win'));
            const known = summary.deltasKnown !== false;
            const delta = summary.deltas[p];
            const deltaText = known ? `${delta > 0 ? '+' : delta < 0 ? '−' : '±'}${number(Math.abs(delta))}` : '—';
            cell.append(
                status,
                node(
                    'div',
                    `round-delta${known && delta > 0 ? ' is-positive' : known && delta < 0 ? ' is-negative' : ''}`,
                    deltaText,
                ),
            );
            cell.append(
                node(
                    'div',
                    'round-score-path',
                    `${number(summary.startScores[p])} → ${known ? number(summary.endScores[p]) : '—'}`,
                ),
            );
            row.append(cell);
        });
        body.append(row);
    });
    table.append(body);
    scroller.append(table);
    if (summaries.length === 0) scroller.append(node('p', 'round-browser-empty', i18n.text('No rounds available')));
    content.append(scroller);
    return content;
}
