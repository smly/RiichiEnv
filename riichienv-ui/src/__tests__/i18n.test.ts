import { describe, expect, it } from 'vitest';
import { YAKU_MAP } from '../constants';
import { I18n, isLocale, locales } from '../i18n/index';

describe('Viewer localization', () => {
    it('provides matching catalogs, placeholders and all supported yaku IDs', () => {
        const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]).sort();
        for (const locale of Object.values(locales)) {
            expect(Object.keys(locale.messages).sort()).toEqual(Object.keys(locales.ja.messages).sort());
            for (const key of Object.keys(locales.ja.messages) as (keyof typeof locales.ja.messages)[]) {
                expect(locale.messages[key].length).toBeGreaterThan(0);
                expect(placeholders(locale.messages[key])).toEqual(placeholders(locales.ja.messages[key]));
            }
            expect(Object.keys(locale.yaku).sort()).toEqual(Object.keys(YAKU_MAP).sort());
        }
    });
    it('keeps the language independent for each viewer', () => {
        const first = new I18n(),
            second = new I18n('en');
        first.locale = 'en';
        second.locale = 'ja';
        expect(first.text('Settings')).toBe('Settings');
        expect(second.text('Settings')).toBe('設定');
    });
    it.each([4, 3])('formats wind transitions for %i-player rounds', (pc) => {
        expect(new I18n().round(pc - 1, pc)).toBe(`東${pc}局`);
        expect(new I18n().round(pc, pc)).toBe('南1局');
        expect(new I18n('en').round(pc, pc)).toBe('South 1');
        expect(new I18n('en').wind(1, true)).toBe('S');
    });
    it('translates calls, limits, draw reasons, yaku and point units', () => {
        const ja = new I18n();
        expect(['chi', 'pon', 'ankan', 'kakan', 'reach', 'kita', 'tsumo', 'ron'].map((type) => ja.call(type))).toEqual([
            'チー',
            'ポン',
            'カン',
            'カン',
            'リーチ',
            '抜きドラ',
            'ツモ',
            'ロン',
        ]);
        expect(ja.limit('double-yakuman')).toBe('ダブル役満');
        expect(ja.limit('3x-yakuman')).toBe('3倍役満');
        expect(new I18n('en').limit('3x-yakuman')).toBe('3x Yakuman');
        expect(ja.reason('kyushu_kyuhai')).toBe('九種九牌');
        expect(ja.yaku(1)).toBe('門前清自摸和');
        expect(new I18n('en').yaku(1)).toBe('Menzen Tsumo');
        expect(ja.text('{count} Points', { count: 8000 })).toBe('8000 点');
    });
    it.each([
        ['zh-Hans', '东1局', '荣和', '门前清自摸和', '双倍役满', '国士无双十三面'],
        ['zh-Hant', '東1局', '榮和', '門前清自摸和', '雙倍役滿', '國士無雙十三面'],
    ] as const)('localizes Chinese mahjong terms in %s', (locale, round, ron, tsumo, limit, kokushi) => {
        const i18n = new I18n(locale);
        expect(i18n.round(0)).toBe(round);
        expect(i18n.round(3, 3)).toBe('南1局');
        expect(i18n.call('ron')).toBe(ron);
        expect(i18n.yaku(1)).toBe(tsumo);
        expect(i18n.yaku(49)).toBe(kokushi);
        expect(i18n.limit('double-yakuman')).toBe(limit);
        expect(i18n.text('{count} Han', { count: 6 })).toBe('6 番');
        expect(i18n.text('{count} Fu', { count: 20 })).toBe('20 符');
    });
    it('preserves external text and unknown IDs without accidental recursive replacement', () => {
        const i18n = new I18n();
        expect(i18n.reason('custom reason')).toBe('custom reason');
        expect(i18n.reason('constructor')).toBe('constructor');
        expect(i18n.yaku(999)).toBe('役 999');
        expect(i18n.text('Turn: {name}', { name: '{count} <player>' })).toBe('手番：{count} <player>');
        expect(isLocale('en')).toBe(true);
        expect(isLocale('zh-Hans')).toBe(true);
        expect(isLocale('zh-Hant')).toBe(true);
        expect(isLocale('zh')).toBe(false);
        expect(isLocale('constructor')).toBe(false);
    });
});
