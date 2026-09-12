import { afterEach, describe, expect, it, vi } from 'vitest';
import {
    getTileBodySprite,
    projectTileBody,
    type TileBodyRequest,
    type TileBodyShape,
    type Vec,
} from '../renderers/tile_body_sprite';

const shape: TileBodyShape = {
    width: 30,
    height: 48,
    depth: (30 * 16.5) / 20.5,
    radius: 30 * 0.14,
    edge: 30 * 0.07,
    flipped: false,
};
const dot = (a: Vec, b: Vec) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const center: Vec = [shape.width / 2, shape.height / 2, shape.depth / 2];

// Build the table/seat/standing pose independently of DOMMatrix and the renderer.
function basis(seat: number, standing: boolean): [Vec, Vec, Vec] {
    const tilt = (48 * Math.PI) / 180;
    const angle = (seat * Math.PI) / 2;
    const rotate = ([x, y, z]: Vec): Vec => {
        if (standing) [y, z] = [-z, y];
        [x, y] = [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
        return [x, y * Math.cos(tilt) - z * Math.sin(tilt), y * Math.sin(tilt) + z * Math.cos(tilt)];
    };
    const columns = [rotate([1, 0, 0]), rotate([0, 1, 0]), rotate([0, 0, 1])];
    return [0, 1, 2].map((i) => columns.map((column) => column[i]) as Vec) as [Vec, Vec, Vec];
}

describe('tile body bitmap projection', () => {
    it.each(
        [0, 1, 2, 3].flatMap((seat) => [false, true].map((standing) => ({ seat, standing }))),
    )('matches perspective projection for seat $seat, standing=$standing', ({ seat, standing }) => {
        const [u, v, n] = basis(seat, standing);
        const eye = center.map((c, i) => c + u[i] * 250 - v[i] * 170 + n[i] * 1800) as Vec;
        const projection = projectTileBody(shape, u, v, n, eye);
        expect(projection.visible.length).toBeGreaterThan(0);
        expect(projection.width).toBeGreaterThan(0);
        expect(projection.height).toBeGreaterThan(0);
        // Every projected facet must lie inside the outer contour: a missing
        // far-rim corner would draw a diagonal across the printed artwork.
        for (let i = 0; i < projection.outline.length; i++) {
            const a = projection.outline[i];
            const b = projection.outline[(i + 1) % projection.outline.length];
            expect(a[0] - projection.left).toBeGreaterThanOrEqual(0.35);
            expect(a[1] - projection.top).toBeGreaterThanOrEqual(0.35);
            expect(projection.left + projection.width - a[0]).toBeGreaterThanOrEqual(0.35);
            expect(projection.top + projection.height - a[1]).toBeGreaterThanOrEqual(0.35);
            for (const p of projection.visible.flatMap((face) => face.xy)) {
                expect((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])).toBeGreaterThanOrEqual(-1e-9);
            }
        }
        for (const face of projection.visible) {
            expect(Math.hypot(...face.normal)).toBeCloseTo(1, 10);
            face.points.forEach((point, i) => {
                const relative = point.map((x, j) => x - center[j]) as Vec;
                const ahead = dot(projection.origin.map((value, j) => value - point[j]) as Vec, n);
                expect(ahead).toBeGreaterThan(0);
                const x = dot(relative, u),
                    y = dot(relative, v),
                    z = dot(relative, n);
                const [px, py] = face.xy[i];
                // Moving the supporting bitmap ahead of the tabletop must not
                // change the final screen projection of its silhouette.
                const dz = dot(projection.origin.map((value, j) => value - center[j]) as Vec, n);
                expect((1800 * px - 250 * dz) / (1800 - dz)).toBeCloseTo((1800 * x - 250 * z) / (1800 - z), 9);
                expect((1800 * py + 170 * dz) / (1800 - dz)).toBeCloseTo((1800 * y + 170 * z) / (1800 - z), 9);
                expect(px).toBeGreaterThan(projection.left);
                expect(px).toBeLessThan(projection.left + projection.width);
                expect(py).toBeGreaterThan(projection.top);
                expect(py).toBeLessThan(projection.top + projection.height);
            });
        }
    });

    it('keeps the printed-face opening transparent and closes the opposite face', () => {
        const front = projectTileBody(shape, [1, 0, 0], [0, 1, 0], [0, 0, 1], null);
        // The bitmap contains the rim, not a lid that could cover the live artwork.
        expect(front.visible.length).toBeGreaterThan(0);
        expect(front.visible.some((face) => face.normal[2] === 1)).toBe(false);
        const back = projectTileBody(shape, [-1, 0, 0], [0, 1, 0], [0, 0, -1], null);
        expect(back.visible.some((face) => face.normal[2] === -1 && face.cap)).toBe(true);
    });

    it.each([false, true])('keeps the amber cap in the thin back layer (flipped=$0)', (flipped) => {
        const [u, v, n] = basis(1, true);
        const { visible } = projectTileBody({ ...shape, flipped }, u, v, n, null);
        expect(visible.some((face) => face.cap)).toBe(true);
        expect(visible.some((face) => !face.cap)).toBe(true);
        for (const face of visible.filter((face) => face.cap)) {
            for (const point of face.points) {
                if (flipped) expect(point[2]).toBeGreaterThanOrEqual(shape.depth * 0.8 - 1e-9);
                else expect(point[2]).toBeLessThanOrEqual(shape.depth * 0.2 + 1e-9);
            }
        }
    });

    it('uses orthographic projection for the own meld row without changing its baseline', () => {
        const [u, v, n] = basis(0, false);
        const projection = projectTileBody(shape, u, v, n, null);
        for (const face of projection.visible) {
            face.points.forEach((point, i) => {
                const relative = point.map((x, j) => x - center[j]) as Vec;
                expect(face.xy[i][0]).toBeCloseTo(dot(relative, u), 10);
                expect(face.xy[i][1]).toBeCloseTo(dot(relative, v), 10);
            });
        }
    });

    it.each([
        0, 1, 2, 3,
    ])('keeps meld skins coplanar with artwork without moving their silhouettes (seat %i)', (seat) => {
        const [u, v, n] = basis(seat, false);
        const eye = center.map((c, i) => c + u[i] * 250 - v[i] * 170 + n[i] * 1800) as Vec;
        // Upright and sideways tiles share a top height despite different footprints.
        for (const [width, height] of [
            [30, 48],
            [48, 30],
        ]) {
            const meld = { ...shape, width, height, facePlane: true };
            const projection = projectTileBody(meld, u, v, n, eye);
            expect(projection.origin[2]).toBeGreaterThan(shape.depth);
            expect(projection.origin[2]).toBe(shape.depth + 0.02);
            for (const face of projection.visible) {
                face.points.forEach((point, i) => {
                    const imagePoint: Vec = [
                        projection.origin[0] + face.xy[i][0],
                        projection.origin[1] + face.xy[i][1],
                        projection.origin[2],
                    ];
                    // The skin's reprojected pixel stays on the exact same camera ray.
                    const t = (imagePoint[2] - eye[2]) / (point[2] - eye[2]);
                    expect(imagePoint[0]).toBeCloseTo(eye[0] + t * (point[0] - eye[0]), 9);
                    expect(imagePoint[1]).toBeCloseTo(eye[1] + t * (point[1] - eye[1]), 9);
                });
            }
        }
    });
});

describe('tile body bitmap cache', () => {
    afterEach(() => vi.unstubAllGlobals());

    function canvasStub() {
        const encode = vi.fn(() => 'data:image/png;base64,test');
        const context = {
            scale() {},
            translate() {},
            beginPath() {},
            moveTo() {},
            lineTo() {},
            closePath() {},
            fill() {},
            stroke() {},
            resetTransform() {},
            fillRect() {},
        };
        vi.stubGlobal('document', { createElement: () => ({ getContext: () => context, toDataURL: encode }) });
        return encode;
    }

    it('does not rasterize again on a cache hit; different tints get separate images', () => {
        const encode = canvasStub();
        const request: TileBodyRequest = { ...shape, width: 30.01, element: {} as HTMLElement, overlays: [] };
        const [u, v, n] = basis(0, false);
        const first = getTileBodySprite(request, u, v, n, null);
        expect(getTileBodySprite({ ...request }, u, v, n, null)).toBe(first);
        expect(encode).toHaveBeenCalledTimes(1);
        getTileBodySprite({ ...request, overlays: ['rgba(0, 0, 0, 0.35)'] }, u, v, n, null);
        expect(encode).toHaveBeenCalledTimes(2);
        const meld = getTileBodySprite({ ...request, facePlane: true }, u, v, n, null);
        expect(meld).not.toBe(first);
        expect(encode).toHaveBeenCalledTimes(3);
    });

    it('evicts old poses instead of retaining every generated PNG indefinitely', () => {
        const encode = canvasStub();
        const request: TileBodyRequest = { ...shape, width: 30.02, element: {} as HTMLElement, overlays: [] };
        const [u, v, n] = basis(0, false);
        const eye = (x: number): Vec => [x, 800, 1800];
        getTileBodySprite(request, u, v, n, eye(0));
        for (let i = 1; i <= 400; i++) getTileBodySprite(request, u, v, n, eye(i * 8));
        expect(encode).toHaveBeenCalledTimes(401);
        getTileBodySprite(request, u, v, n, eye(0));
        expect(encode).toHaveBeenCalledTimes(402);
    });
});
