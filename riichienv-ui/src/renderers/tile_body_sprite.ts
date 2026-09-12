import { TILE_BACK_FRACTION, TILE_BACK_RGB, TILE_FACE_RGB } from './tile_material';

/** Rasterize a rounded body once per pose; the live scene contains one image. */
export type Vec = [number, number, number];
export interface TileBodyShape {
    width: number;
    height: number;
    depth: number;
    radius: number;
    edge: number;
    flipped: boolean;
    /** Composite meld skins and artwork on the same plane in painter order. */
    facePlane?: boolean;
}
export interface TileBodyRequest extends TileBodyShape {
    element: HTMLElement;
    overlays: string[];
}
interface Sprite {
    url: string;
    left: number;
    top: number;
    width: number;
    height: number;
    origin: Vec;
}
interface Face {
    points: Vec[];
    normal: Vec;
    cap: boolean;
}
const sprites = new Map<string, Sprite>();
const meshes = new Map<string, Face[]>();
const MAX_SPRITES = 384;
const dot = (a: Vec, b: Vec) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec, b: Vec): Vec => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];
const unit = (a: Vec): Vec => {
    const l = Math.hypot(...a);
    return a.map((x) => x / l) as Vec;
};
const LIGHT = unit([-0.3, 0.4, 1]);
const point = (p: DOMPoint): Vec => [p.x, p.y, p.z];

type Point2 = [number, number];

/** The rounded body is convex. Trace its silhouette, never its facet seams. */
function silhouette(points: Point2[]): Point2[] {
    const sorted = points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const unique = sorted.filter((p, i) => !i || p[0] !== sorted[i - 1][0] || p[1] !== sorted[i - 1][1]);
    const turn = (a: Point2, b: Point2, c: Point2) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const half = (vertices: Point2[]) => {
        const hull: Point2[] = [];
        for (const p of vertices) {
            while (hull.length > 1 && turn(hull[hull.length - 2], hull[hull.length - 1], p) <= 0) hull.pop();
            hull.push(p);
        }
        return hull.slice(0, -1);
    };
    return [...half(unique), ...half([...unique].reverse())];
}

function mesh(request: TileBodyShape): Face[] {
    const { width: w, height: h, depth: d, radius: r, edge: e, flipped } = request;
    const key = [w, h, d, r, e, flipped].join(':');
    const cached = meshes.get(key);
    if (cached) return cached;
    const capZ = d * (flipped ? 1 - TILE_BACK_FRACTION : TILE_BACK_FRACTION);
    const levels = [...new Set([0, e * 0.25, e * 0.6, e, capZ, d - e, d - e * 0.6, d - e * 0.25, d])].sort(
        (a, b) => a - b,
    );
    const rings = levels.map((z) => {
        const distance = Math.min(e, z, d - z);
        const inset = e - Math.sqrt(Math.max(0, e * e - (e - distance) ** 2));
        const radius = r - inset;
        const ring: Vec[] = [];
        for (const [cx, cy, start] of [
            [w - r, h - r, 90],
            [w - r, r, 0],
            [r, r, -90],
            [r, h - r, -180],
        ]) {
            for (let i = 0; i <= 5; i++) {
                const a = ((start - i * 18) * Math.PI) / 180;
                ring.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a), z]);
            }
        }
        return ring;
    });
    const faces: Face[] = [];
    for (let j = 0; j < rings.length - 1; j++) {
        const lower = rings[j],
            upper = rings[j + 1];
        const z = (levels[j] + levels[j + 1]) / 2;
        for (let i = 0; i < lower.length; i++) {
            const n = (i + 1) % lower.length;
            const points = [lower[i], lower[n], upper[n], upper[i]];
            faces.push({
                points,
                normal: unit(cross(sub(upper[i], lower[i]), sub(lower[n], lower[i]))),
                cap: flipped ? z > capZ : z < capZ,
            });
        }
    }
    faces.push({ points: rings[0], normal: [0, 0, -1], cap: !flipped });
    // Leave the printed-face opening transparent. Its live CSS plane can lie
    // behind parts of the billboard, so painting a lid here would hide the art.
    meshes.set(key, faces);
    if (meshes.size > 32) meshes.delete(meshes.keys().next().value!);
    return faces;
}

/** Keep the bitmap above the tabletop. Melds share their artwork plane for
 * painter ordering; other tiles use a camera-facing plane ahead of the body.
 * Reprojection preserves the silhouette when moving the supporting plane. */
export function tileBodyImageOrigin(request: TileBodyShape, n: Vec): Vec {
    if (request.facePlane) return [request.width / 2, request.height / 2, request.depth + 0.02];
    const extent =
        (Math.abs(n[0]) * request.width + Math.abs(n[1]) * request.height + Math.abs(n[2]) * request.depth) / 2;
    const center: Vec = [request.width / 2, request.height / 2, request.depth / 2];
    return center.map((value, i) => value + n[i] * (extent + 0.1)) as Vec;
}

/** Project only the body skin; the printed-face opening stays transparent. */
export function projectTileBody(request: TileBodyShape, u: Vec, v: Vec, n: Vec, eye: Vec | null) {
    const c = tileBodyImageOrigin(request, n);
    const planeNormal: Vec = request.facePlane ? [0, 0, 1] : n;
    const project = (p: Vec): [number, number] => {
        let q = sub(p, c);
        if (eye) {
            const direction = sub(p, eye);
            const t = dot(sub(c, eye), planeNormal) / dot(direction, planeNormal);
            q = sub(eye.map((x, i) => x + t * direction[i]) as Vec, c);
        } else if (request.facePlane) {
            const t = -q[2] / n[2];
            q = q.map((value, i) => value + t * n[i]) as Vec;
        }
        return request.facePlane ? [q[0], q[1]] : [dot(q, u), dot(q, v)];
    };
    const faces = mesh(request);
    const visible = faces
        .filter((face) => dot(face.normal, eye ? sub(eye, face.points[0]) : n) > 0)
        .map((face) => ({ ...face, xy: face.points.map(project) }));
    // Include the far rim around the transparent artwork opening as well.
    const outline = silhouette(faces.flatMap((f) => f.points.map(project)));
    const coords = outline;
    const left = Math.min(...coords.map((p) => p[0])) - 0.5,
        top = Math.min(...coords.map((p) => p[1])) - 0.5;
    const width = Math.max(...coords.map((p) => p[0])) - left + 0.5,
        height = Math.max(...coords.map((p) => p[1])) - top + 0.5;
    return { visible, outline, left, top, width, height, origin: c };
}

export function getTileBodySprite(request: TileBodyRequest, u: Vec, v: Vec, n: Vec, eye: Vec | null): Sprite {
    // Subpixel pose quantization avoids cache churn from layout rounding.
    // At the default 1800px perspective, 4 local units change a rim by <0.1px.
    const key = [
        request.width,
        request.height,
        request.depth,
        request.radius,
        request.edge,
        request.flipped,
        !!request.facePlane,
        ...request.overlays,
        ...u.map((x) => Math.round(x * 10000)),
        ...v.map((x) => Math.round(x * 10000)),
        ...(eye ? eye.map((x) => Math.round(x / 4)) : ['ortho']),
    ].join(':');
    const cached = sprites.get(key);
    if (cached) {
        sprites.delete(key);
        sprites.set(key, cached);
        return cached;
    }
    const { visible, outline, left, top, width, height, origin } = projectTileBody(request, u, v, n, eye);
    const canvas = document.createElement('canvas');
    const resolution = 3;
    canvas.width = Math.ceil(width * resolution);
    canvas.height = Math.ceil(height * resolution);
    const ctx = canvas.getContext('2d')!;
    ctx.scale(resolution, resolution);
    ctx.translate(-left, -top);
    // A convex body has no overlapping visible facets. A matching subpixel
    // stroke seals antialias seams between facets inside this single bitmap.
    for (const face of visible) {
        const diffuse = Math.max(0, dot(face.normal, LIGHT));
        const light = face.cap ? 0.9 + 0.1 * diffuse : 0.82 + 0.18 * Math.min(1, diffuse / LIGHT[2]);
        const base = face.cap ? TILE_BACK_RGB : TILE_FACE_RGB;
        const color = `rgb(${base.map((x) => Math.round(x * light)).join(',')})`;
        ctx.beginPath();
        face.xy.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 0.2;
        ctx.stroke();
    }
    // Bake a fine outer contour into this same cached PNG. The face opening
    // stays transparent, and adjacent ivory bodies remain distinguishable.
    ctx.beginPath();
    outline.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.strokeStyle = 'rgba(58, 64, 70, 0.82)';
    ctx.lineWidth = 0.7;
    ctx.lineJoin = 'round';
    ctx.stroke();
    // Composite highlights into the cached bitmap: no per-tile CSS filters or
    // overlay facets are needed, and simultaneous tsumogiri/danger tints stack.
    ctx.resetTransform();
    ctx.globalCompositeOperation = 'source-atop';
    for (const color of request.overlays) {
        ctx.fillStyle = color;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    const sprite = { url: canvas.toDataURL('image/png'), left, top, width, height, origin };
    sprites.set(key, sprite);
    if (sprites.size > MAX_SPRITES) sprites.delete(sprites.keys().next().value!);
    return sprite;
}

/** Read transforms in one batch, then attach sprites without further layout reads. */
export function paintTileBodies(requests: TileBodyRequest[], scene: HTMLElement): void {
    const styles = new Map<HTMLElement, CSSStyleDeclaration>();
    const offsets = new Map<HTMLElement, [number, number]>();
    const matrices = new Map<HTMLElement, DOMMatrix>();
    const roots = new Map<HTMLElement, HTMLElement>();
    const style = (el: HTMLElement) => {
        let s = styles.get(el);
        if (!s) {
            s = getComputedStyle(el);
            styles.set(el, s);
        }
        return s;
    };
    const offset = (el: HTMLElement): [number, number] => {
        let p = offsets.get(el);
        if (p) return p;
        const parent = el.offsetParent as HTMLElement | null;
        const base = parent ? offset(parent) : [0, 0];
        p = [base[0] + el.offsetLeft + (parent?.clientLeft ?? 0), base[1] + el.offsetTop + (parent?.clientTop ?? 0)];
        offsets.set(el, p);
        return p;
    };
    const root = (el: HTMLElement): HTMLElement => {
        let r = roots.get(el);
        if (r) return r;
        const parent = el.parentElement!;
        r = parent === scene || style(parent).perspective !== 'none' ? parent : root(parent);
        roots.set(el, r);
        return r;
    };
    const matrix = (el: HTMLElement, r: HTMLElement): DOMMatrix => {
        if (el === r) return new DOMMatrix();
        let m = matrices.get(el);
        if (m) return m;
        const parent = el.parentElement!;
        const s = style(el);
        const [x, y] = offset(el),
            [px, py] = offset(parent);
        const [ox, oy, oz = 0] = s.transformOrigin.split(' ').map(parseFloat);
        const local = new DOMMatrix()
            .translate(x - px, y - py)
            .translate(ox, oy, oz)
            .multiply(s.transform === 'none' ? new DOMMatrix() : new DOMMatrix(s.transform))
            .translate(-ox, -oy, -oz);
        m = matrix(parent, r).multiply(local);
        matrices.set(el, m);
        return m;
    };
    const plans = requests.map((request) => {
        const r = root(request.element),
            inverse = matrix(request.element, r).inverse();
        const u = unit(point(inverse.transformPoint(new DOMPoint(1, 0, 0, 0))));
        const v = unit(point(inverse.transformPoint(new DOMPoint(0, 1, 0, 0))));
        const n = unit(cross(u, v));
        const s = style(r);
        let eye: Vec | null = null;
        if (s.perspective !== 'none') {
            const [x, y] = s.perspectiveOrigin.split(' ').map(parseFloat);
            eye = point(inverse.transformPoint(new DOMPoint(x, y, parseFloat(s.perspective))));
        }
        return { request, u, v, n, eye };
    });
    const meldOrder = new Map<HTMLElement, number>();
    for (const { request, u, v, n, eye } of plans) {
        if (request.facePlane && eye) {
            const center: Vec = [request.width / 2, request.height / 2, request.depth / 2];
            const order = -Math.round(Math.hypot(...sub(eye, center)) * 100);
            // Coplanar surfaces must paint whole tiles from far to near. Apply
            // the same ordering to nested added-kan pairs and separate melds.
            for (
                let el: HTMLElement | null = request.element;
                el && !el.classList.contains('opp-melds-inner');
                el = el.parentElement
            ) {
                meldOrder.set(el, Math.max(meldOrder.get(el) ?? -Infinity, order));
            }
        }
        const sprite = getTileBodySprite(request, u, v, n, eye);
        // Melds share one plane with their artwork, so a camera-facing
        // billboard cannot slice a neighbor. Each tile paints body then art.
        const [imageU, imageV, imageN]: Vec[] = request.facePlane
            ? [
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
              ]
            : [u, v, n];
        const transform = new DOMMatrix([
            imageU[0],
            imageU[1],
            imageU[2],
            0,
            imageV[0],
            imageV[1],
            imageV[2],
            0,
            imageN[0],
            imageN[1],
            imageN[2],
            0,
            sprite.origin[0],
            sprite.origin[1],
            sprite.origin[2],
            1,
        ]).translate(sprite.left, sprite.top);
        const image = document.createElement('img');
        image.className = 'tile-3d-surface tile-body-image';
        image.alt = '';
        image.draggable = false;
        image.src = sprite.url;
        Object.assign(image.style, {
            width: `${sprite.width}px`,
            height: `${sprite.height}px`,
            transform: transform.toString(),
        });
        request.element.prepend(image);
    }
    for (const [element, order] of meldOrder) element.style.zIndex = `${order}`;
}
