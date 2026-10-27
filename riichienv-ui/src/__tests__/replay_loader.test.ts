import { afterEach, describe, expect, it, vi } from 'vitest';
import { decodeReplay, loadReplayFile, loadReplayUrl, parseReplay } from '../replay_loader';

const events = [
    { type: 'start_game', names: ['A', 'B', 'C', 'D'] },
    { type: 'start_kyoku', tehais: [['1m'], ['2m'], ['3m'], ['4m']], scores: [25000, 25000, 25000, 25000] },
];
const jsonl = events.map((event) => JSON.stringify(event)).join('\n');
const url = 'https://logs.riichi.dev/replay.jsonl.gz';
afterEach(() => vi.unstubAllGlobals());

describe('URL replay loading', () => {
    it('parses JSONL with a BOM, CRLF and empty lines', () => {
        expect(parseReplay(`\uFEFF\n${jsonl.replace(/\n/g, '\r\n')}\r\n`)).toEqual(events);
    });
    it('decodes gzip using the browser stream API', async () => {
        const stream = new Blob([jsonl]).stream().pipeThrough(new CompressionStream('gzip'));
        const compressed = await new Response(stream).arrayBuffer();
        expect(await decodeReplay(compressed)).toEqual(events);
    });
    it('accepts already decompressed HTTP responses despite the gz suffix', async () => {
        const fetcher = vi.fn().mockResolvedValue(new Response(jsonl));
        vi.stubGlobal('fetch', fetcher);
        expect(await loadReplayUrl(url)).toEqual(events);
        expect(fetcher).toHaveBeenCalledWith(url, expect.objectContaining({ credentials: 'omit' }));
    });
    it.each([
        '',
        'file:///tmp/replay',
        'javascript:alert(1)',
        'https://user:pass@logs.riichi.dev/a',
    ])('rejects invalid URL %s', async (input) => {
        const fetcher = vi.fn();
        vi.stubGlobal('fetch', fetcher);
        await expect(loadReplayUrl(input)).rejects.toThrow('Enter a valid');
        expect(fetcher).not.toHaveBeenCalled();
    });
    it('reports HTTP failures', async () => {
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 404 })));
        await expect(loadReplayUrl(url)).rejects.toThrow('Could not fetch');
    });
    it('reports network failures', async () => {
        vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
        await expect(loadReplayUrl(url)).rejects.toThrow('Could not fetch');
    });
    it('propagates cancellation', async () => {
        const controller = new AbortController();
        controller.abort();
        vi.stubGlobal('fetch', vi.fn().mockRejectedValue(controller.signal.reason));
        await expect(loadReplayUrl(url, controller.signal)).rejects.toBe(controller.signal.reason);
    });
    it.each([
        '',
        '<html>Error</html>',
        '{broken',
        'null',
        '{"type":"start_game"}',
        '{"type":"start_kyoku"}',
    ])('rejects malformed or missing replay data: %s', (input) => {
        expect(() => parseReplay(input)).toThrow('valid MJAI JSONL');
    });
    it('reports corrupt gzip', async () => {
        await expect(decodeReplay(new Uint8Array([0x1f, 0x8b, 0]).buffer)).rejects.toThrow('decompress');
    });
});

describe('Local replay file loading', () => {
    it('loads JSONL locally without uploading it', async () => {
        const fetcher = vi.fn();
        vi.stubGlobal('fetch', fetcher);
        expect(await loadReplayFile(new File([jsonl], '牌譜.jsonl'))).toEqual(events);
        expect(fetcher).not.toHaveBeenCalled();
    });
    it('loads compressed JSONL with a case-insensitive extension', async () => {
        const compressed = await new Response(
            new Blob([jsonl]).stream().pipeThrough(new CompressionStream('gzip')),
        ).arrayBuffer();
        expect(await loadReplayFile(new File([compressed], 'replay.JSONL.GZ'))).toEqual(events);
    });
    it('loads a three-player replay', async () => {
        const sanma = [{ type: 'start_kyoku', tehais: [['1p'], ['2p'], ['3p']], scores: [35000, 35000, 35000] }];
        expect(await loadReplayFile(new File([JSON.stringify(sanma[0])], 'sanma.jsonl'))).toEqual(sanma);
    });
    it.each(['replay.json', 'replay.gz', 'replay.jsonl.zip'])('rejects unsupported extension %s', async (name) => {
        await expect(loadReplayFile(new File([jsonl], name))).rejects.toThrow('Choose a .jsonl or .jsonl.gz');
    });
    it('reports corrupt gzip and invalid JSONL', async () => {
        await expect(loadReplayFile(new File([new Uint8Array([0x1f, 0x8b, 0])], 'bad.jsonl.gz'))).rejects.toThrow(
            'decompress',
        );
        await expect(loadReplayFile(new File(['not a replay'], 'bad.jsonl'))).rejects.toThrow('valid MJAI JSONL');
    });
    it('reports file read failures', async () => {
        const file = new File([jsonl], 'replay.jsonl');
        vi.spyOn(file, 'arrayBuffer').mockRejectedValue(new Error('File unavailable'));
        await expect(loadReplayFile(file)).rejects.toThrow('Could not read the replay file');
    });
    it('does not read a file after cancellation', async () => {
        const file = new File([jsonl], 'replay.jsonl');
        const read = vi.spyOn(file, 'arrayBuffer');
        const controller = new AbortController();
        controller.abort();
        await expect(loadReplayFile(file, controller.signal)).rejects.toBe(controller.signal.reason);
        expect(read).not.toHaveBeenCalled();
    });
    it('discards a read that finishes after cancellation', async () => {
        const file = new File([jsonl], 'replay.jsonl');
        const controller = new AbortController();
        vi.spyOn(file, 'arrayBuffer').mockImplementation(async () => {
            controller.abort();
            return new TextEncoder().encode(jsonl).buffer;
        });
        await expect(loadReplayFile(file, controller.signal)).rejects.toBe(controller.signal.reason);
    });
});
