// PNG (png.h). A 32-bit framebuffer is BGRA-ish "0x80RRGGBB", so the app hands over 3-byte RGB rows; this writes
// the signature, IHDR, one IDAT with deflate's stored blocks and IEND, each chunk with its CRC.
#include "png.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint32_t crc_table[256];
static int crc_ready;

static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
    crc_ready = 1;
}

static uint32_t crc32_of(const uint8_t* p, size_t n, uint32_t crc) {
    if (!crc_ready) crc_init();
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = crc_table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

static uint32_t adler32_of(const uint8_t* p, size_t n) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < n; i++) {
        a = (a + p[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

static void put32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24), p[1] = (uint8_t)(v >> 16), p[2] = (uint8_t)(v >> 8), p[3] = (uint8_t)v;
}

uint8_t* png_encode(const uint8_t* rgb, int w, int h, size_t* size) {
    if (!rgb || w <= 0 || h <= 0) return NULL;
    size_t stride = (size_t)w * 3 + 1;           // one filter byte in front of every row
    size_t raw_n = stride * (size_t)h;
    size_t blocks = (raw_n + 65534) / 65535;     // deflate's stored blocks hold 65535 bytes
    size_t z_n = 2 + 5 * blocks + raw_n + 4;     // zlib header, block headers, data, adler32
    size_t out_cap = 8 + 25 + 12 + z_n + 12;
    uint8_t* out = malloc(out_cap);
    if (!out) return NULL;
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    memcpy(out, sig, 8);
    uint8_t* p = out + 8;
    // IHDR: 8 bits, colour type 2 (RGB), no interlace
    put32(p, 13);
    const char* ihdr = "IHDR";
    memcpy(p + 4, ihdr, 4);
    put32(p + 8, (uint32_t)w);
    put32(p + 12, (uint32_t)h);
    p[16] = 8, p[17] = 2, p[18] = 0, p[19] = 0, p[20] = 0;
    put32(p + 21, crc32_of(p + 4, 17, 0));
    p += 25;
    // IDAT
    put32(p, (uint32_t)z_n);
    memcpy(p + 4, "IDAT", 4);
    uint8_t* z = p + 8;
    z[0] = 0x78, z[1] = 0x01;   // deflate, 32K window, no dictionary
    uint8_t* zp = z + 2;
    size_t left = raw_n;
    size_t row = 0, col = 0;    // where we are in the raw stream (filter byte, then the row's pixels)
    while (left) {
        size_t n = left > 65535 ? 65535 : left;
        zp[0] = (uint8_t)(left > 65535 ? 0x00 : 0x01);   // last block: BFINAL
        zp[1] = (uint8_t)(n & 0xFF), zp[2] = (uint8_t)(n >> 8);
        zp[3] = (uint8_t)(~n & 0xFF), zp[4] = (uint8_t)(~n >> 8);
        zp += 5;
        for (size_t i = 0; i < n; i++) {   // the filter byte first, then the row's bytes
            if (col == 0) {
                zp[i] = 0;
                col = 1;
                continue;
            }
            zp[i] = rgb[row * (size_t)w * 3 + col - 1];
            if (++col > (size_t)w * 3) col = 0, row++;
        }
        zp += n;
        left -= n;
    }
    {   // the adler over the raw stream: a zero filter byte, then the row's pixels
        uint32_t a = 1, b = 0;
        for (int y = 0; y < h; y++) {
            a = (a + 0) % 65521, b = (b + a) % 65521;
            const uint8_t* r = rgb + (size_t)y * (size_t)w * 3;
            for (int x = 0; x < w * 3; x++) {
                a = (a + r[x]) % 65521;
                b = (b + a) % 65521;
            }
        }
        put32(zp, (b << 16) | a);
    }
    zp += 4;
    size_t idat = (size_t)(zp - (p + 4));
    put32(p, (uint32_t)(z_n));
    put32(p + 4 + 4 + z_n, crc32_of(p + 4, idat, 0));   // CRC over "IDAT" + the zlib stream
    p += 8 + z_n + 4;
    put32(p, 0);
    memcpy(p + 4, "IEND", 4);
    put32(p + 8, crc32_of(p + 4, 4, 0));
    p += 12;
    *size = (size_t)(p - out);
    return out;
}

int png_write(const char* path, const uint8_t* rgb, int w, int h) {
    size_t n = 0;
    uint8_t* d = png_encode(rgb, w, h, &n);
    if (!d) return -1;
    int f = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (f < 0) {
        free(d);
        return -1;
    }
    size_t done = 0;
    while (done < n) {
        ssize_t r = write(f, d + done, n - done);
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(f);
    free(d);
    return done == n ? 0 : -1;
}
