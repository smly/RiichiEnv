import { describe, it, expect } from 'vitest';
import {
    createGameConfig4P,
    createGameConfig3P,
    createLayoutConfig4P,
    createLayoutConfig3P,
    GameConfig,
    LayoutConfig,
} from '../config';

describe('GameConfig', () => {
    describe('createGameConfig4P', () => {
        let config: GameConfig;

        it('should create a valid 4-player config', () => {
            config = createGameConfig4P();
            expect(config.playerCount).toBe(4);
        });

        it('should have 4 default scores', () => {
            config = createGameConfig4P();
            expect(config.defaultScores).toHaveLength(4);
            expect(config.defaultScores).toEqual([25000, 25000, 25000, 25000]);
        });

        it('should have 4 winds', () => {
            config = createGameConfig4P();
            expect(config.winds).toHaveLength(4);
            expect(config.winds).toEqual(['E', 'S', 'W', 'N']);
        });

        it('should have 4 wind char keys', () => {
            config = createGameConfig4P();
            expect(config.windCharKeys).toHaveLength(4);
        });

        it('should have correct initial wall remaining', () => {
            config = createGameConfig4P();
            expect(config.initialWallRemaining).toBe(70);
        });
    });

    describe('createGameConfig3P', () => {
        let config: GameConfig;

        it('should create a valid 3-player config', () => {
            config = createGameConfig3P();
            expect(config.playerCount).toBe(3);
        });

        it('should have 3 default scores', () => {
            config = createGameConfig3P();
            expect(config.defaultScores).toHaveLength(3);
            expect(config.defaultScores).toEqual([35000, 35000, 35000]);
        });

        it('should have 3 winds', () => {
            config = createGameConfig3P();
            expect(config.winds).toHaveLength(3);
            expect(config.winds).toEqual(['E', 'S', 'W']);
        });

        it('should have 3 wind char keys', () => {
            config = createGameConfig3P();
            expect(config.windCharKeys).toHaveLength(3);
        });

        it('should have correct initial wall remaining', () => {
            config = createGameConfig3P();
            expect(config.initialWallRemaining).toBe(55);
        });
    });

    describe('consistency', () => {
        it('should have matching array lengths for 4P', () => {
            const config = createGameConfig4P();
            expect(config.defaultScores.length).toBe(config.playerCount);
            expect(config.winds.length).toBe(config.playerCount);
            expect(config.windCharKeys.length).toBe(config.playerCount);
        });

        it('should have matching array lengths for 3P', () => {
            const config = createGameConfig3P();
            expect(config.defaultScores.length).toBe(config.playerCount);
            expect(config.winds.length).toBe(config.playerCount);
            expect(config.windCharKeys.length).toBe(config.playerCount);
        });
    });
});

describe('LayoutConfig', () => {
    describe('createLayoutConfig4P', () => {
        let layout: LayoutConfig;

        it('should have 4 player angles', () => {
            layout = createLayoutConfig4P();
            expect(layout.playerAngles).toHaveLength(4);
        });

        it('should have correct board size', () => {
            layout = createLayoutConfig4P();
            expect(layout.boardSize).toBe(800);
        });

        it('should have correct content dimensions', () => {
            layout = createLayoutConfig4P();
            expect(layout.contentWidth).toBe(970);
            expect(layout.contentHeight).toBe(900);
        });

        it('should have correct view area size', () => {
            layout = createLayoutConfig4P();
            expect(layout.viewAreaSize).toBe(880);
        });

        it('should have bottom player at 0 degrees', () => {
            layout = createLayoutConfig4P();
            expect(layout.playerAngles[0]).toBe(0);
        });
    });

    describe('createLayoutConfig3P', () => {
        let layout: LayoutConfig;

        it('should have 3 player angles', () => {
            layout = createLayoutConfig3P();
            expect(layout.playerAngles).toHaveLength(3);
        });

        it('should have bottom player at 0 degrees', () => {
            layout = createLayoutConfig3P();
            expect(layout.playerAngles[0]).toBe(0);
        });

        it('should have correct board size', () => {
            layout = createLayoutConfig3P();
            expect(layout.boardSize).toBe(800);
        });
    });

    describe('consistency', () => {
        it('4P layout angles match player count', () => {
            const config = createGameConfig4P();
            const layout = createLayoutConfig4P();
            expect(layout.playerAngles.length).toBe(config.playerCount);
        });

        it('3P layout angles match player count', () => {
            const config = createGameConfig3P();
            const layout = createLayoutConfig3P();
            expect(layout.playerAngles.length).toBe(config.playerCount);
        });
    });
});
