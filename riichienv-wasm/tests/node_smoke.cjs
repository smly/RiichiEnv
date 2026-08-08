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
assert.ok(base instanceof Float32Array);
assert.ok(extended instanceof Float32Array);
assert.ok(mask instanceof Uint8Array);
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
