const fs = require('fs');
const zlib = require('zlib');
const path = require('path');

const inputFile = path.join(__dirname, '..', 'dist', 'viewer.js');
const outputFile = path.join(__dirname, '..', 'dist', 'viewer.js.gz');

if (!fs.existsSync(inputFile)) {
    console.error(`Input file not found: ${inputFile}`);
    process.exit(1);
}

const input = fs.readFileSync(inputFile);
zlib.gzip(input, (err, buffer) => {
    if (err) {
        console.error('Compression failed:', err);
        process.exit(1);
    }
    fs.writeFileSync(outputFile, buffer);
    console.log(`Compressed ${inputFile} (${(input.length / 1024).toFixed(1)} KB) to ${outputFile} (${(buffer.length / 1024).toFixed(1)} KB)`);
});
