import type { MjaiEvent } from './types';

export class ReplayLoadError extends Error {}

/** Fetch without credentials; gzip may already have been decoded by HTTP. */
export async function loadReplayUrl(input: string, signal?: AbortSignal): Promise<MjaiEvent[]> {
    let url: URL;
    try {
        url = new URL(input.trim());
        if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) throw new Error();
    } catch {
        throw new ReplayLoadError('Enter a valid HTTP or HTTPS replay URL.');
    }
    let bytes: ArrayBuffer;
    try {
        const response = await fetch(url.href, { signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
        if (!response.ok) throw new Error();
        bytes = await response.arrayBuffer();
    } catch (error) {
        if (signal?.aborted) throw error;
        throw new ReplayLoadError('Could not fetch replay. Check the URL and server access.');
    }
    return decodeReplay(bytes);
}

/** Read local files entirely in the browser; no upload or network request. */
export async function loadReplayFile(file: File, signal?: AbortSignal): Promise<MjaiEvent[]> {
    if (signal?.aborted) throw signal.reason;
    if (!/\.jsonl(?:\.gz)?$/i.test(file.name)) {
        throw new ReplayLoadError('Choose a .jsonl or .jsonl.gz replay file.');
    }
    let bytes: ArrayBuffer;
    try {
        bytes = await file.arrayBuffer();
    } catch (error) {
        if (signal?.aborted) throw error;
        throw new ReplayLoadError('Could not read the replay file. Please select it again.');
    }
    if (signal?.aborted) throw signal.reason;
    const events = await decodeReplay(bytes);
    if (signal?.aborted) throw signal.reason;
    return events;
}

export async function decodeReplay(bytes: ArrayBuffer): Promise<MjaiEvent[]> {
    const header = new Uint8Array(bytes);
    let text: string;
    try {
        if (header[0] === 0x1f && header[1] === 0x8b) {
            const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
            text = await new Response(stream).text();
        } else {
            text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
        }
    } catch {
        throw new ReplayLoadError('Could not decompress or decode the replay.');
    }
    return parseReplay(text);
}

export function parseReplay(text: string): MjaiEvent[] {
    try {
        const events: MjaiEvent[] = text
            .replace(/^\uFEFF/, '')
            .split(/\r?\n/)
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
        if (
            !events.length ||
            events.some((event) => !event || typeof event !== 'object' || typeof event.type !== 'string')
        ) {
            throw new Error();
        }
        const rounds = events.filter((event) => event.type === 'start_kyoku');
        if (
            !rounds.length ||
            rounds.some(
                (round) =>
                    !Array.isArray(round.tehais) ||
                    ![3, 4].includes(round.tehais.length) ||
                    round.tehais.some(
                        (hand: unknown) => !Array.isArray(hand) || hand.some((tile) => typeof tile !== 'string'),
                    ) ||
                    !Array.isArray(round.scores) ||
                    round.scores.length !== round.tehais.length ||
                    round.scores.some((score: unknown) => typeof score !== 'number'),
            )
        )
            throw new Error();
        return events;
    } catch {
        throw new ReplayLoadError('This is not a valid MJAI JSONL replay.');
    }
}
