// PNG, written by hand: the app shows images the model draws and the API sends them, and there is no zlib on the
// console (nor in a package this size). Deflate's stored blocks cost size but nothing else, and an image the
// console renders is a few hundred kilobytes, not a photo.
#pragma once
#include <stddef.h>
#include <stdint.h>

// Encodes w x h pixels of rgb (3 bytes each, row-major, no padding) into a PNG file, in a buffer the caller frees
// (NULL on failure; *size gets its length). Filters: none.
uint8_t* png_encode(const uint8_t* rgb, int w, int h, size_t* size);

// The same, straight to a file. 0 = ok.
int png_write(const char* path, const uint8_t* rgb, int w, int h);
