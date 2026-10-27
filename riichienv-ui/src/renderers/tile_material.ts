/** Orange sampled from the upper back lip of the bundled tile face atlas. */
export const TILE_BACK_RGB = [212, 122, 5] as const;
export const TILE_BACK_COLOR = `rgb(${TILE_BACK_RGB.join(',')})`;

/** Keep a visible orange band above the rounded lower edge. */
export const TILE_BACK_FRACTION = 0.2;

/** Flat ivory area of the PNG, shared with the adjoining rounded body rim. */
export const TILE_FACE_RGB = [228, 228, 228] as const;
export const TILE_FACE_COLOR = `rgb(${TILE_FACE_RGB.join(',')})`;
