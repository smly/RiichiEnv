// esbuild's binary loader embeds WASM as bytes in both bundle formats.
declare module '*.wasm' {
    const bytes: Uint8Array;
    export default bytes;
}
