// Host tests for chat/img.c (the PNG reader, the scaler and base64).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/img_test.c chat/img.c
//                chat/png.c -o build/img_test && build/img_test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "img.h"
#include "png.h"

static int fails, checks;
#define CHECK(c)                                                                                                      \
    do {                                                                                                              \
        checks++;                                                                                                     \
        if (!(c)) {                                                                                                   \
            fails++;                                                                                                  \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                                       \
        }                                                                                                             \
    } while (0)

// Fixtures from Python's zlib (a real deflate, so the dynamic-Huffman path is the one under test).
static const char* PNG_RGB7x5 =
    "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAFCAIAAAAG+GGPAAAAV0lEQVR42l3KoQ0CQRBA0Qec3YyeFSSn1iEm68DQAAWgKYJKqIQCOYIj+"
    "erlw5ELd568eLNTm7a/9puqproaaqqruh08RPbINXJEniJn5Hn5vhqdwfz1ASGlCZNlUNoDAAAAAElFTkSuQmCC";
static const char* PNG_PAL4x3 =
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAMAAACDKl70AAAADFBMVEUA/wAo4RRQwyh4pTyrUAlLAAAABHRSTlP/gAD/oaGUZgAAABFJ"
    "REFUeNpjYGBkYgZjBhANAACVABNrP7KVAAAAAElFTkSuQmCC";
static const char* PNG_RGBA3x2 =
    "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAYAAACddGYaAAAAFElEQVR42mM4kWLEAMT/QZgBmQMAmDYMMWMWJtUAAAAASUVORK5CYII=";
static const char* PNG_G16_2x1 =
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABEAAAAACB2fwVAAAADUlEQVR42mMQMll9FgADDAG/sefUTQAAAABJRU5ErkJggg==";

static void test_decode(void) {
    Img im;
    CHECK(img_decode_b64(PNG_RGB7x5, strlen(PNG_RGB7x5), &im, 0) == 0);
    CHECK(im.w == 7 && im.h == 5);
    int good = 1;   // every filter type is in this one (0..4, a row each)
    for (int y = 0; y < 5 && good; y++)
        for (int x = 0; x < 7; x++) {
            const uint8_t* p = im.rgb + (y * 7 + x) * 3;
            if (p[0] != x * 30 % 256 || p[1] != y * 50 % 256 || p[2] != x * y * 13 % 256) {
                good = 0;
                printf("  rgb (%d,%d) = %d,%d,%d\n", x, y, p[0], p[1], p[2]);
                break;
            }
        }
    CHECK(good);
    img_free(&im);

    // a palette with tRNS: index 1 is half transparent, index 2 fully
    CHECK(img_decode_b64(PNG_PAL4x3, strlen(PNG_PAL4x3), &im, 0) == 0);
    CHECK(im.w == 4 && im.h == 3);
    CHECK(im.rgb[0] == 0 && im.rgb[1] == 255 && im.rgb[2] == 0);           // index 0
    CHECK(im.rgb[3] == 20 && im.rgb[4] == 112 && im.rgb[5] == 10);         // index 1 over black
    CHECK(im.rgb[6] == 0 && im.rgb[7] == 0 && im.rgb[8] == 0);             // index 2, fully transparent
    img_free(&im);

    CHECK(img_decode_b64(PNG_RGBA3x2, strlen(PNG_RGBA3x2), &im, 0xFFFFFF) == 0);
    CHECK(im.w == 3 && im.h == 2);
    CHECK(im.rgb[0] == 255 && im.rgb[1] == 255 && im.rgb[2] == 255);       // alpha 0 over white
    CHECK(im.rgb[3] == 200 && im.rgb[4] == 100 && im.rgb[5] == 50);
    img_free(&im);

    CHECK(img_decode_b64(PNG_G16_2x1, strlen(PNG_G16_2x1), &im, 0) == 0);  // 16 bits: the high byte
    CHECK(im.w == 2 && im.h == 1);
    CHECK(im.rgb[0] == 0x12 && im.rgb[3] == 0xAB);
    img_free(&im);

    CHECK(img_decode_b64("bm90IGEgcG5n", 11, &im, 0) < 0);                 // "not a png"
    CHECK(im.rgb == NULL);
    size_t n = strlen(PNG_RGB7x5);
    // (cutting only the end of the deflate stream's checksum still decodes: this reader does not check it)
    CHECK(img_decode_b64(PNG_RGB7x5, n - 60, &im, 0) < 0);                 // cut off in the middle
    CHECK(img_decode_b64("!!!", 3, &im, 0) < 0);
}

static void test_encode_roundtrip(void) {
    uint8_t rgb[16 * 9 * 3];
    for (int i = 0; i < (int)sizeof(rgb); i++) rgb[i] = (uint8_t)(i * 7);
    size_t n = 0;
    uint8_t* png = png_encode(rgb, 16, 9, &n);
    CHECK(png && n > 8);
    Img im;
    CHECK(img_decode(png, n, &im, 0) == 0);
    CHECK(im.w == 16 && im.h == 9);
    CHECK(im.rgb && !memcmp(im.rgb, rgb, sizeof(rgb)));
    img_free(&im);
    free(png);
}

static void test_scale(void) {
    uint8_t rgb[8 * 4 * 3];
    memset(rgb, 200, sizeof(rgb));
    Img src = {8, 4, rgb}, dst;
    CHECK(img_scale(&src, &dst, 4, 100) == 0);
    CHECK(dst.w == 4 && dst.h == 2);
    CHECK(dst.rgb[0] == 200 && dst.rgb[3 * 4 * 2 - 1] == 200);
    img_free(&dst);
    CHECK(img_scale(&src, &dst, 100, 2) == 0);   // the height is the limit here
    CHECK(dst.w == 4 && dst.h == 2);
    img_free(&dst);
    CHECK(img_scale(&src, &dst, 100, 100) == 0); // never enlarged
    CHECK(dst.w == 8 && dst.h == 4);
    img_free(&dst);
}

static void test_base64(void) {
    const uint8_t in[] = {'M', 'a', 'n', 0, 0xFF, 0x10, 0x7F};
    size_t n = 0;
    char* s = img_b64_encode(in, sizeof(in), &n);
    CHECK(s != NULL);
    CHECK(n == 12);
    CHECK(!strcmp(s, "TWFuAP8Qfw=="));
    size_t back_n = 0;
    Img im;
    CHECK(img_decode_b64(s, n, &im, 0) < 0);   // valid base64, not an image
    CHECK(back_n == 0);
    free(s);
}

int main(void) {
    test_decode();
    test_encode_roundtrip();
    test_scale();
    test_base64();
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
