import { en } from './en';
import { ja, type Messages } from './ja';
import { yakuEn, yakuJa } from './yaku';
import { zhHans, zhHansYaku } from './zh-Hans';
import { zhHant, zhHantYaku } from './zh-Hant';

// A locale is added by supplying a complete message catalog and yaku catalog here.
// BCP 47 locale IDs keep Simplified and Traditional Chinese scripts distinct.
export const locales = {
    ja: { name: '日本語', messages: ja, yaku: yakuJa },
    en: { name: 'English', messages: en, yaku: yakuEn },
    'zh-Hans': { name: '简体中文', messages: zhHans, yaku: zhHansYaku },
    'zh-Hant': { name: '繁體中文', messages: zhHant, yaku: zhHantYaku },
} satisfies Record<string, { name: string; messages: Messages; yaku: Record<number, string> }>;
export type Locale = keyof typeof locales;
export const isLocale = (value: string): value is Locale => Object.keys(locales).includes(value);

const calls: Record<string, string> = {
    chi: 'Chii',
    pon: 'Pon',
    kan: 'Kan',
    ankan: 'Kan',
    daiminkan: 'Kan',
    kakan: 'Kan',
    reach: 'Riichi',
    kita: 'Pei',
    tsumo: 'Tsumo',
    ron: 'Ron',
    ryukyoku: 'Ryukyoku',
};
const limits: Record<string, string> = {
    mangan: 'Mangan',
    haneman: 'Haneman',
    baiman: 'Baiman',
    sanbaiman: 'Sanbaiman',
    yakuman: 'Yakuman',
    'double-yakuman': 'Double Yakuman',
};
const reasons: Record<string, string> = {
    fanpai: 'Exhaustive draw',
    exhaustive_draw: 'Exhaustive draw',
    kyushu_kyuhai: 'Nine terminals',
    suukansansen: 'Four kans',
    suucha_riichi: 'Four riichi',
    sanchaho: 'Triple ron',
    sufuurenta: 'Four winds',
    nagashimangan: 'Nagashi Mangan',
    Error: 'Error (Penalty)',
};

const lookup = (map: Record<string, string>, key: string): string =>
    Object.getOwnPropertyDescriptor(map, key)?.value ?? key;

/** Per-viewer locale state; no global language or game-state mutations. */
export class I18n {
    constructor(public locale: Locale = 'ja') {}

    text(key: string, values: Record<string, string | number> = {}): string {
        const catalog: Record<string, string> = locales[this.locale].messages;
        return lookup(catalog, key).replace(/\{(\w+)\}/g, (token, name) => String(values[name] ?? token));
    }
    apply(root: HTMLElement): void {
        root.querySelectorAll<HTMLElement>('[data-i18n]').forEach((el) => {
            el.textContent = this.text(el.dataset.i18n!);
        });
        root.querySelectorAll<HTMLElement>('[data-i18n-label]').forEach((el) => {
            const text = this.text(el.dataset.i18nLabel!);
            el.setAttribute('aria-label', text);
            if (el.hasAttribute('title')) el.title = text;
        });
    }
    wind(index: number, short = false): string {
        return this.text(`${short ? 'seat' : ''}${['East', 'South', 'West', 'North'][index] ?? 'East'}`);
    }
    round(round: number, playerCount = 4): string {
        return this.text('roundName', {
            wind: this.wind(Math.floor(round / playerCount)),
            number: (round % playerCount) + 1,
        });
    }
    call(type: string): string {
        return this.text(lookup(calls, type));
    }
    limit(limit: string): string {
        const multiple = /^(\d+)x-yakuman$/.exec(limit);
        return multiple ? this.text('{count}x Yakuman', { count: multiple[1] }) : this.text(lookup(limits, limit));
    }
    reason(reason = ''): string {
        return this.text(lookup(reasons, reason || 'Ryukyoku'));
    }
    yaku(id: number): string {
        return locales[this.locale].yaku[id] ?? this.text('Yaku {id}', { id });
    }
}
