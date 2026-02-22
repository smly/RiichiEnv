/**
 * Async WASM module loader.
 *
 * Handles initialization of the riichienv-wasm module.
 * The WASM binary is inlined by esbuild's binary loader at build time,
 * so no separate .wasm file fetch is needed at runtime.
 */

// @ts-ignore - esbuild binary loader provides Uint8Array
import wasmBinary from './pkg/riichienv_wasm_bg.wasm';

type WasmModule = typeof import('./pkg/riichienv_wasm');

let wasmModule: WasmModule | null = null;
let initPromise: Promise<void> | null = null;
let initFailed = false;

/**
 * Initialize the WASM module. Safe to call multiple times.
 * Returns a promise that resolves when the module is ready.
 */
export async function initWasm(): Promise<void> {
    if (wasmModule) return;
    if (initFailed) return;
    if (initPromise) return initPromise;

    initPromise = (async () => {
        try {
            const mod = await import('./pkg/riichienv_wasm');
            // Pass the inlined WASM binary directly to avoid import.meta.url
            // issues in IIFE format. esbuild's --loader:.wasm=binary provides
            // the binary as a Uint8Array.
            await mod.default({ module_or_path: wasmBinary });
            wasmModule = mod;
        } catch (e) {
            initFailed = true;
            console.warn('[WASM] Initialization failed:', e);
            throw e; // Re-throw so callers' .catch() handlers fire
        }
    })();

    return initPromise;
}

/**
 * Get the loaded WASM module, or null if not yet initialized.
 */
export function getWasm(): WasmModule | null {
    return wasmModule;
}

/**
 * Check if the WASM module is ready for use.
 */
export function isWasmReady(): boolean {
    return wasmModule !== null;
}
