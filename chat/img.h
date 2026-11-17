// Images, read by hand: PNG decoding, scaling and base64, beside png.c's encoder. There is no zlib and no libjpeg on
// the console, so this is the inflate the PNG chunks need and nothing more (8-bit, non-interlaced: what a model or an
// image backend sends, and what the API answers with). No console calls here, so tests/img_test.c runs on a PC.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int w, h;
    uint8_t* rgb;               // w * h * 3 bytes, row-major, no padding (NULL if the image is empty)
} Img;

// Decodes a PNG file. Colour types 0 (grey), 2 (RGB), 3 (palette), 4 (grey+alpha), 6 (RGBA), with or without a
// tRNS chunk; 16-bit samples lose their low byte; alpha is mixed over bg (0xRRGGBB). 0 = ok, else a negative code
// (see img_strerror).
int img_decode(const uint8_t* data, size_t n, Img* out, uint32_t bg);
// Decodes a base64 image (a data: URL's tail, or bare base64). 0 = ok.
int img_decode_b64(const char* s, size_t n, Img* out, uint32_t bg);
// Reads a PNG file. 0 = ok.
int img_load(const char* path, Img* out, uint32_t bg);
void img_free(Img* im);

// Fits src into max_w x max_h (never enlarging), averaging the pixels it drops. dst gets a new buffer; returns 0,
// or -1 when out of memory (dst is then empty).
int img_scale(const Img* src, Img* dst, int max_w, int max_h);

// base64, as data: URLs and JSON's b64_json carry it. The result is NUL-terminated (the caller frees it).
char* img_b64_encode(const uint8_t* p, size_t n, size_t* out_n);
// Why the last img_decode failed, for the log and the API's error message.
const char* img_strerror(int code);
