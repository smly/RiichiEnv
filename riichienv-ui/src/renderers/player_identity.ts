/*!
 * Boring Avatars — Beam and utilities (adapted for standalone SVG images).
 * Source: https://github.com/boringdesigners/boring-avatars
 * Revision: d0ff2582a8921b643a89de4a4912be28938a828b
 *
 * MIT License
 *
 * Copyright (c) 2021 boringdesigners
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

const hashCode = (name: string): number => {
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        const character = name.charCodeAt(i);
        hash = (hash << 5) - hash + character;
        hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
};

const getDigit = (number: number, ntn: number): number => {
    return Math.floor((number / 10 ** ntn) % 10);
};

const getBoolean = (number: number, ntn: number): boolean => {
    return !(getDigit(number, ntn) % 2);
};

const getUnit = (number: number, range: number, index?: number): number => {
    const value = number % range;

    if (index && getDigit(number, index) % 2 === 0) {
        return -value;
    } else return value;
};

const getRandomColor = (number: number, colors: string[], range: number): string => {
    return colors[number % range];
};

const getContrast = (hexcolor: string): string => {
    // If a leading # is provided, remove it
    if (hexcolor.slice(0, 1) === '#') {
        hexcolor = hexcolor.slice(1);
    }

    // Convert to RGB value
    const r = parseInt(hexcolor.substr(0, 2), 16);
    const g = parseInt(hexcolor.substr(2, 2), 16);
    const b = parseInt(hexcolor.substr(4, 2), 16);

    // Get YIQ ratio
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;

    // Check contrast
    return yiq >= 128 ? '#000000' : '#FFFFFF';
};

const SIZE = 36;

function generateData(name: string, colors: string[]) {
    const numFromName = hashCode(name);
    const range = colors.length;
    const wrapperColor = getRandomColor(numFromName, colors, range);
    const preTranslateX = getUnit(numFromName, 10, 1);
    const wrapperTranslateX = preTranslateX < 5 ? preTranslateX + SIZE / 9 : preTranslateX;
    const preTranslateY = getUnit(numFromName, 10, 2);
    const wrapperTranslateY = preTranslateY < 5 ? preTranslateY + SIZE / 9 : preTranslateY;

    const data = {
        wrapperColor: wrapperColor,
        faceColor: getContrast(wrapperColor),
        backgroundColor: getRandomColor(numFromName + 13, colors, range),
        wrapperTranslateX: wrapperTranslateX,
        wrapperTranslateY: wrapperTranslateY,
        wrapperRotate: getUnit(numFromName, 360),
        wrapperScale: 1 + getUnit(numFromName, SIZE / 12) / 10,
        isMouthOpen: getBoolean(numFromName, 2),
        isCircle: getBoolean(numFromName, 1),
        eyeSpread: getUnit(numFromName, 5),
        mouthSpread: getUnit(numFromName, 3),
        faceRotate: getUnit(numFromName, 10, 3),
        faceTranslateX: wrapperTranslateX > SIZE / 6 ? wrapperTranslateX / 2 : getUnit(numFromName, 8, 1),
        faceTranslateY: wrapperTranslateY > SIZE / 6 ? wrapperTranslateY / 2 : getUnit(numFromName, 7, 2),
    };

    return data;
}

const COLORS = ['#cf0638', '#fa6632', '#fecd23', '#0a996f', '#0a6789'];
// Bound memory when a long-lived embedded viewer loads many different replays.
const avatarCache = new Map<string, string>();
const CACHE_LIMIT = 256;

/** Square Beam image seeded only by name, independent of language and viewpoint. */
export function playerAvatar(name: string): string {
    const cached = avatarCache.get(name);
    if (cached) return cached;

    const d = generateData(name, COLORS);
    // Names never enter the markup. A fixed mask ID is safe inside each img's SVG document.
    const mouth = d.isMouthOpen
        ? `<path d="M15 ${19 + d.mouthSpread}c2 1 4 1 6 0" stroke="${d.faceColor}" fill="none" stroke-linecap="round"/>`
        : `<path d="M13,${19 + d.mouthSpread} a1,0.75 0 0,0 10,0" fill="${d.faceColor}"/>`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 36 36" width="64" height="64" fill="none">
<mask id="beam" maskUnits="userSpaceOnUse" x="0" y="0" width="36" height="36"><rect width="36" height="36" fill="#FFFFFF"/></mask>
<g mask="url(#beam)">
<rect width="36" height="36" fill="${d.backgroundColor}"/>
<rect width="36" height="36" transform="translate(${d.wrapperTranslateX} ${d.wrapperTranslateY}) rotate(${d.wrapperRotate} 18 18) scale(${d.wrapperScale})" fill="${d.wrapperColor}" rx="${d.isCircle ? 36 : 6}"/>
<g transform="translate(${d.faceTranslateX} ${d.faceTranslateY}) rotate(${d.faceRotate} 18 18)">
${mouth}
<rect x="${14 - d.eyeSpread}" y="14" width="1.5" height="2" rx="1" stroke="none" fill="${d.faceColor}"/>
<rect x="${20 + d.eyeSpread}" y="14" width="1.5" height="2" rx="1" stroke="none" fill="${d.faceColor}"/>
</g></g></svg>`;
    const url = `data:image/svg+xml,${encodeURIComponent(svg)}`;
    if (avatarCache.size >= CACHE_LIMIT) {
        const oldest = avatarCache.keys().next().value;
        if (oldest !== undefined) avatarCache.delete(oldest);
    }
    avatarCache.set(name, url);
    return url;
}
