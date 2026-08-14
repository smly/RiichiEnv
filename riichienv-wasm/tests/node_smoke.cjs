'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');

const packageDirectory = process.argv[2];
if (!packageDirectory) throw new Error('usage: node node_smoke.cjs <wasm-package-directory>');

const wasm = require(path.join(path.resolve(packageDirectory), 'riichienv_wasm.js'));

function assertThrowsWith(fn, pattern) {
    let thrown;
    try {
        fn();
    } catch (error) {
        thrown = error;
    }
    assert.notEqual(thrown, undefined, `expected an exception matching ${pattern}`);
    assert.match(String(thrown), pattern);
}

const startGame = '{"type":"start_game","names":["A","B","C","D"]}';

const partial = new wasm.EventJournal();
partial.pushRawJson(startGame);
partial.pushRawJson('{"type":"end_game","wall":["PRIVATE"]}');
assert.equal(partial.complete, true);
assert.equal(partial.spectatorCursor, 1);
assert.deepEqual(partial.spectatorSafePrefix().events, [startGame]);
partial.free();

const framedPrivate = new wasm.EventJournal();
framedPrivate.pushRawJson(startGame);
framedPrivate.pushRawJson('{"type":"start_kyoku"}');
framedPrivate.pushRawJson('{"type":"end_kyoku"}');
framedPrivate.pushRawJson('{"type":"end_game","wall":["PRIVATE"]}');
assert.equal(framedPrivate.spectatorCursor, 3);
assert.doesNotMatch(framedPrivate.spectatorSafePrefix().events.join('\n'), /PRIVATE/);
framedPrivate.free();

const numeric = new wasm.EventJournal();
numeric.pushRawJson('{"type":"start_game","id":1.0,"seed":[2e0]}');
assert.equal(numeric.spectatorCursor, 0);
numeric.pushRawJson('{"type":"start_kyoku","bakaze":"E","kyoku":1.0,"honba":2e0}');
numeric.pushRawJson('{"type":"end_kyoku"}');
numeric.pushRawJson('{"type":"end_game"}');
assert.equal(numeric.spectatorCursor, numeric.revision);
assert.deepEqual(numeric.completedKyokuSpans()[0].key, { bakaze: 'E', kyoku: 1, honba: 2 });
assertThrowsWith(() => numeric.eventsSince('0'), /cursor must be a JavaScript number/);
numeric.free();

const unsafe = new wasm.EventJournal();
unsafe.pushRawJson('{"type":"start_game","id":9007199254740992}');
unsafe.pushRawJson('{"type":"start_kyoku"}');
unsafe.pushRawJson('{"type":"end_kyoku"}');
unsafe.pushRawJson('{"type":"end_game"}');
assert.equal(unsafe.spectatorCursor, 0);
unsafe.free();

const duplicate = new wasm.EventJournal();
assertThrowsWith(
    () => duplicate.pushRawJson('{"type":"start_game","future":{"secret":1,"secret":2}}'),
    /duplicate JSON object key/,
);
assert.equal(duplicate.revision, 0);
duplicate.free();

const defaults = new wasm.GameEngine('4p-red-single', null, null);
defaults.free();
assertThrowsWith(
    () => new wasm.GameEngine('4p-red-single', '42', false),
    /seed must be a JavaScript number/,
);

const engine = new wasm.GameEngine('4p-red-single', 42, false);
const decisions = engine.decisions();
assert.ok(decisions.length > 0);
const first = decisions[0];
const base = engine.baseFeatures(first.playerId);
const extended = engine.extendedFeatures(first.playerId);
const mask = engine.actionMask(first.playerId);
const maskV1 = engine.actionMaskV1(first.playerId);
assert.ok(base instanceof Float32Array);
assert.ok(extended instanceof Float32Array);
assert.ok(mask instanceof Uint8Array);
assert.ok(maskV1 instanceof Uint8Array);
assert.equal(first.actionMask.length, 82);
assert.equal(first.actionMaskV1.length, 164);
assert.equal(maskV1.length, 164);
assert.equal(base.length, first.baseShape[0] * first.baseShape[1]);
assert.equal(extended.length, first.extendedShape[0] * first.extendedShape[1]);
assertThrowsWith(() => engine.baseFeatures('0'), /playerId must be a JavaScript number/);

const outcome = engine.stepActionIds(
    decisions.map((decision) => ({
        playerId: decision.playerId,
        actionId: decision.legalActionIds[0],
    })),
);
assert.equal(outcome.error, null);
assert.ok(Array.isArray(outcome.events));
assert.ok(Array.isArray(outcome.nextDecisions));
engine.free();

const engineV1 = new wasm.GameEngine('4p-red-single', 42, false);
const decisionsV1 = engineV1.decisions();
const outcomeV1 = engineV1.stepActionIdsV1(
    decisionsV1.map((decision) => ({
        playerId: decision.playerId,
        actionId: decision.legalActionIdsV1[0],
    })),
);
assert.equal(outcomeV1.error, null);
engineV1.free();

const sanma = new wasm.GameEngine('3p-red-single', 42, false);
assert.equal(wasm.drevV2SchemaId(), 'riichienv.drev_v2.81ch.v1');
assert.deepEqual(wasm.drevV2OpponentSlotOrder(), ['relative_1', 'relative_2', 'relative_3']);
assert.equal(wasm.drevV2YakuEvidenceNames().length, 19);
assert.equal(wasm.drevV2ChannelNames().length, 81);
const sanmaDecision = sanma.decisions()[0];
const sanmaSp = sanma.spFeatures(sanmaDecision.playerId);
const sanmaDrev = sanma.drevFeatures(sanmaDecision.playerId);
const sanmaDrevV2 = sanma.drevV2Features(sanmaDecision.playerId);
const sanmaCombined = sanma.extendedWithSpFeatures(sanmaDecision.playerId);
const sanmaCombinedV2 = sanma.extendedWithSpDrevV2Features(sanmaDecision.playerId);
assert.ok(sanmaSp instanceof Float32Array);
assert.ok(sanmaDrev instanceof Float32Array);
assert.ok(sanmaDrevV2 instanceof Float32Array);
assert.ok(sanmaCombined instanceof Float32Array);
assert.ok(sanmaCombinedV2 instanceof Float32Array);
assert.deepEqual(sanmaDecision.spShape, [178, 27]);
assert.deepEqual(sanmaDecision.drevShape, [9, 27]);
assert.deepEqual(sanmaDecision.drevV2Shape, [81, 27]);
assert.deepEqual(sanmaDecision.extendedWithSpShape, [402, 27]);
assert.deepEqual(sanmaDecision.extendedWithSpDrevV2Shape, [474, 27]);
assert.equal(sanmaSp.length, 178 * 27);
assert.equal(sanmaDrev.length, 9 * 27);
assert.equal(sanmaDrevV2.length, 81 * 27);
assert.equal(sanmaCombined.length, 402 * 27);
assert.equal(sanmaCombinedV2.length, 474 * 27);
assert.equal(sanmaDecision.actionMask.length, 60);
assert.equal(sanmaDecision.actionMaskV1.length, 120);
sanma.free();

function runFullEastGame(mode, seed, numPlayers) {
    const full = new wasm.GameEngine(mode, seed, true);
    const winId = numPlayers === 4 ? 79 : 56;
    const passId = numPlayers === 4 ? 81 : 58;
    const discardLimit = numPlayers === 4 ? 34 : 27;
    let pending = full.decisions();
    let steps = 0;

    while (!full.snapshot().done) {
        steps += 1;
        assert.ok(steps < 20000, `${mode} exceeded the full-game step cap`);
        const inputs = pending.map((decision) => {
            const ids = decision.legalActionIds;
            let actionId;
            if (ids.includes(winId)) {
                actionId = winId;
            } else if (ids.includes(passId)) {
                actionId = passId;
            } else {
                actionId = ids.find((id) => id < discardLimit) ?? ids[0];
            }
            return { playerId: decision.playerId, actionId };
        });
        const outcome = full.stepActionIds(inputs);
        assert.equal(outcome.error, null);
        pending = outcome.nextDecisions;
    }

    assert.deepEqual(pending, []);
    const log = full.mjaiLog();
    assert.equal(JSON.parse(log.at(-1)).type, 'end_game');
    const journal = full.eventJournal();
    assert.equal(journal.complete, true);
    assert.equal(journal.revision, log.length);
    assert.ok(journal.completedKyokuSpans().length >= numPlayers);
    journal.free();
    full.free();
}

runFullEastGame('4p-red-east', 0x4e01, 4);
runFullEastGame('3p-red-east', 0x3e01, 3);
