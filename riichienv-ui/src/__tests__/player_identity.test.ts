import { describe, expect, it, vi } from 'vitest';
import { playerAvatar } from '../renderers/player_identity';

const svgFor = (name: string) => decodeURIComponent(playerAvatar(name).split(',')[1]);

describe('Player Beam avatars', () => {
    // Expected geometry comes from the upstream generateData implementation at
    // d0ff2582a8921b643a89de4a4912be28938a828b with our five-color palette.
    it.each([
        [
            'Alice',
            'translate(-4 8) rotate(88 18 18) scale(1.1)',
            'translate(0 4) rotate(-8 18 18)',
            'M13,20 a1,0.75 0 0,0 10,0',
        ],
        [
            '牌効率くん',
            'translate(8 -4) rotate(78 18 18) scale(1)',
            'translate(4 -6) rotate(-8 18 18)',
            'M15 19c2 1 4 1 6 0',
        ],
        [
            '👩🏽‍💻 player',
            'translate(-5 -5) rotate(269 18 18) scale(1.2)',
            'translate(-5 -6) rotate(-9 18 18)',
            'M15 21c2 1 4 1 6 0',
        ],
    ])('preserves upstream Beam geometry for %s', (name, shape, face, mouth) => {
        const svg = svgFor(name);
        expect(svg).toContain(`transform="${shape}"`);
        expect(svg).toContain(`transform="${face}"`);
        expect(svg).toContain(`d="${mouth}"`);
    });

    it('uses the requested palette with contrasting facial features', () => {
        expect(svgFor('Alice')).toContain('fill="#fa6632"');
        expect(svgFor('Alice')).toContain('fill="#0a996f"');
        expect(svgFor('Alice')).toContain('fill="#FFFFFF"');
        expect(svgFor('3')).toContain('stroke="#000000"');
        expect(svgFor('1')).toContain('fill="#fecd23"');
        expect(svgFor('1')).toContain('fill="#0a6789"');
        expect(svgFor('2')).toContain('fill="#cf0638"');
    });

    it('clips to a square and returns an embedded image without external resources', () => {
        expect(playerAvatar('Alice')).toMatch(/^data:image\/svg\+xml,/);
        const svg = svgFor('Alice');
        expect(svg).toContain('<rect width="36" height="36" fill="#FFFFFF"/></mask>');
        expect(svg).not.toMatch(/<script|<image|href=|onload=/);
    });

    it.each(['', ' ', '𠮷田', '<script>alert(1)</script>', '\ud800'])('handles arbitrary names safely: %s', (name) => {
        const svg = svgFor(name);
        expect(svg).toContain('<svg');
        expect(svg).not.toMatch(/NaN|undefined|<script/);
    });

    it('distinguishes players whose names start with the same character', () => {
        expect(playerAvatar('Alice')).not.toBe(playerAvatar('Anna'));
        expect(new Set(['0', '1', '2', '3'].map(playerAvatar)).size).toBe(4);
    });

    it('reuses generated images and stays deterministic after cache eviction', () => {
        const encode = vi.spyOn(globalThis, 'encodeURIComponent');
        try {
            const first = playerAvatar('cache verification');
            expect(encode).toHaveBeenCalledTimes(1);
            expect(playerAvatar('cache verification')).toBe(first);
            expect(encode).toHaveBeenCalledTimes(1);
            for (let i = 0; i < 256; i++) playerAvatar(`other player ${i}`);
            encode.mockClear();
            expect(playerAvatar('cache verification')).toBe(first);
            expect(encode).toHaveBeenCalledTimes(1);
        } finally {
            encode.mockRestore();
        }
    });
});
