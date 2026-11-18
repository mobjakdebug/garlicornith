// Station AI: images read by hand (img.h). Inflate (RFC 1951) is the whole of it: a PNG's pixels are one zlib stream,
// and the console has no zlib. Written for the images this app meets — a model's drawing, an image backend's answer,
// a screenshot — so 8 bits a sample, not interlaced, and no CRC check (a broken file fails in the inflate anyway).
#include "img.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "png.h"   // png_encode: the tests and the API's file writing

// ---- inflate --------------------------------------------------------------------------------------------------
// puff.c's way: a 32-bit accumulator, the code tables as counts + symbols, and the bits read a byte at a time.

typedef struct {
    uint32_t buf;
    int cnt;
    const uint8_t* p;
    const uint8_t* end;
    int bad;                    // ran out, or a stream that does not decode
} Bits;

static int bits(Bits* b, int need) {
    while (b->cnt < need) {
        if (b->p >= b->end) {
            b->bad = 1;
            return 0;
        }
        b->buf |= (uint32_t)*b->p++ << b->cnt;
        b->cnt += 8;
    }
    int v = (int)(b->buf & ((1u << need) - 1));
    b->buf >>= need;
    b->cnt -= need;
    return v;
}

// The code tables of one block: how many codes of each length, and the symbols in canonical order.
typedef struct {
    short count[16];
    short symbol[288];
} Huff;

static int huff_build(Huff* h, const uint8_t* lengths, int n) {
    memset(h->count, 0, sizeof(h->count));
    for (int i = 0; i < n; i++) h->count[lengths[i]]++;
    if (h->count[0] == n) return 0;   // no codes: any symbol decodes to nothing
    int left = 1;                     // over-subscribed or incomplete tables: a broken stream
    for (int len = 1; len < 16; len++) {
        left <<= 1;
        left -= h->count[len];
        if (left < 0) return -1;
    }
    short offs[16];
    offs[1] = 0;
    for (int len = 1; len < 15; len++) offs[len + 1] = (short)(offs[len] + h->count[len]);
    for (int i = 0; i < n; i++)
        if (lengths[i]) h->symbol[offs[lengths[i]]++] = (short)i;
    return left;                      // > 0: an incomplete table (allowed for the distance codes)
}

static int huff_decode(Bits* b, const Huff* h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len < 16; len++) {
        code |= bits(b, 1);
        if (b->bad) return -1;
        int count = h->count[len];
        if (code - first < count) return h->symbol[index + (code - first)];
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    b->bad = 1;
    return -1;
}

static const short LBASE[29] = {3,  4,  5,  6,  7,  8,  9,  10,  11,  13,  15,  17,  19,  23, 27,
                                31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const short LEXT[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const short DBASE[30] = {1,   2,   3,   4,   5,   7,    9,    13,   17,   25,   33,   49,   65,    97,    129,
                                193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const short DEXT[30] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                               6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

typedef struct {
    uint8_t* out;
    size_t n, cap;
} Out;

static int out_byte(Out* o, uint8_t v) {
    if (o->n >= o->cap) return -1;
    o->out[o->n++] = v;
    return 0;
}

static int inflate_codes(Bits* b, Out* o, const Huff* lit, const Huff* dist) {
    for (;;) {
        int sym = huff_decode(b, lit);
        if (sym < 0) return -1;
        if (sym < 256) {
            if (out_byte(o, (uint8_t)sym)) return -1;
        } else if (sym == 256) {
            return 0;
        } else {
            sym -= 257;
            if (sym >= 29) return -1;
            int len = LBASE[sym] + bits(b, LEXT[sym]);
            int ds = huff_decode(b, dist);
            if (ds < 0 || ds >= 30) return -1;
            size_t back = (size_t)DBASE[ds] + (size_t)bits(b, DEXT[ds]);
            if (b->bad || back > o->n) return -1;
            for (int i = 0; i < len; i++)
                if (out_byte(o, o->out[o->n - back])) return -1;
        }
    }
}

static int inflate_stored(Bits* b, Out* o) {
    b->buf = 0, b->cnt = 0;   // the block starts on a byte boundary
    if (b->end - b->p < 4) return -1;
    int len = b->p[0] | b->p[1] << 8;
    int nlen = b->p[2] | b->p[3] << 8;
    b->p += 4;
    if ((len ^ 0xFFFF) != nlen || (size_t)(b->end - b->p) < (size_t)len) return -1;
    for (int i = 0; i < len; i++)
        if (out_byte(o, *b->p++)) return -1;
    return 0;
}

static int inflate_fixed(Bits* b, Out* o) {
    static Huff lit, dist;
    static int ready;
    if (!ready) {
        uint8_t lengths[288];
        int i = 0;
        for (; i < 144; i++) lengths[i] = 8;
        for (; i < 256; i++) lengths[i] = 9;
        for (; i < 280; i++) lengths[i] = 7;
        for (; i < 288; i++) lengths[i] = 8;
        huff_build(&lit, lengths, 288);
        for (i = 0; i < 30; i++) lengths[i] = 5;
        huff_build(&dist, lengths, 30);
        ready = 1;
    }
    return inflate_codes(b, o, &lit, &dist);
}

static int inflate_dynamic(Bits* b, Out* o) {
    static const uint8_t order[19] = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
    int nlen = bits(b, 5) + 257, ndist = bits(b, 5) + 1, ncode = bits(b, 4) + 4;
    if (b->bad || nlen > 286 || ndist > 30) return -1;
    uint8_t lengths[320];
    memset(lengths, 0, sizeof(lengths));
    for (int i = 0; i < ncode; i++) lengths[order[i]] = (uint8_t)bits(b, 3);
    Huff lencode, distcode;
    if (huff_build(&lencode, lengths, 19) != 0) return -1;
    int i = 0;
    while (i < nlen + ndist) {
        int sym = huff_decode(b, &lencode);
        if (sym < 0) return -1;
        if (sym < 16) {
            lengths[i++] = (uint8_t)sym;
            continue;
        }
        int len = 0, rep;
        if (sym == 16) {
            if (i == 0) return -1;
            len = lengths[i - 1];
            rep = 3 + bits(b, 2);
        } else if (sym == 17) {
            rep = 3 + bits(b, 3);
        } else {
            rep = 11 + bits(b, 7);
        }
        if (b->bad || i + rep > nlen + ndist) return -1;
        while (rep--) lengths[i++] = (uint8_t)len;
    }
    if (lengths[256] == 0) return -1;   // no end-of-block code
    int err = huff_build(&lencode, lengths, nlen);
    if (err < 0) return -1;
    err = huff_build(&distcode, lengths + nlen, ndist);
    if (err < 0) return -1;
    return inflate_codes(b, o, &lencode, &distcode);
}

// A whole zlib stream (2-byte header, deflate blocks, adler32) into out, which holds cap bytes.
static int inflate_zlib(const uint8_t* p, size_t n, uint8_t* out, size_t cap, size_t* out_n) {
    if (n < 2) return -2;
    int cmf = p[0], flg = p[1];
    if ((cmf & 0x0F) != 8 || ((cmf << 8) | flg) % 31) return -2;
    if (flg & 0x20) return -2;   // a preset dictionary: never in a PNG
    Bits b = {0, 0, p + 2, p + n, 0};
    Out o = {out, 0, cap};
    int last;
    do {
        last = bits(&b, 1);
        int type = bits(&b, 2);
        int r;
        if (b.bad) return -2;
        if (type == 0) r = inflate_stored(&b, &o);
        else if (type == 1) r = inflate_fixed(&b, &o);
        else if (type == 2) r = inflate_dynamic(&b, &o);
        else r = -1;
        if (r) return -2;
    } while (!last);
    *out_n = o.n;
    return 0;
}

// ---- PNG ------------------------------------------------------------------------------------------------------

static uint32_t be32(const uint8_t* p) {
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
}

enum { IMG_OK = 0, IMG_ERR_NOT_PNG = -1, IMG_ERR_BROKEN = -2, IMG_ERR_UNSUPPORTED = -3, IMG_ERR_NOMEM = -4 };

const char* img_strerror(int code) {
    switch (code) {
    case IMG_OK: return "ok";
    case IMG_ERR_NOT_PNG: return "Not a PNG image.";
    case IMG_ERR_BROKEN: return "The PNG image is broken.";
    case IMG_ERR_UNSUPPORTED: return "Unsupported PNG (interlaced, or fewer than 8 bits a sample).";
    default: return "Out of memory for the image.";
    }
}

static uint8_t unfilter_paeth(int a, int b, int c) {
    int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return (uint8_t)(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

int img_decode(const uint8_t* d, size_t n, Img* out, uint32_t bg) {
    out->w = out->h = 0, out->rgb = NULL;
    if (!d || n < 8 + 25 || memcmp(d, "\x89PNG\r\n\x1a\n", 8)) return IMG_ERR_NOT_PNG;
    int w = 0, h = 0, depth = 0, ctype = -1;
    uint8_t palette[256][4];
    int n_pal = 0;
    uint8_t* idat = NULL;
    size_t idat_n = 0, idat_cap = 0;
    for (size_t i = 8; i + 8 <= n;) {
        uint32_t len = be32(d + i);
        const uint8_t* type = d + i + 4;
        if (len > n - i - 8) {
            free(idat);
            return IMG_ERR_BROKEN;
        }
        const uint8_t* body = d + i + 8;
        if (!memcmp(type, "IHDR", 4)) {
            if (len < 13) {
                free(idat);
                return IMG_ERR_BROKEN;
            }
            w = (int)be32(body), h = (int)be32(body + 4);
            depth = body[8], ctype = body[9];
            if (body[12]) {
                free(idat);
                return IMG_ERR_UNSUPPORTED;   // interlaced
            }
        } else if (!memcmp(type, "PLTE", 4)) {
            n_pal = (int)(len / 3);
            for (int k = 0; k < n_pal && k < 256; k++)
                palette[k][0] = body[k * 3], palette[k][1] = body[k * 3 + 1], palette[k][2] = body[k * 3 + 2],
                palette[k][3] = 255;
        } else if (!memcmp(type, "tRNS", 4)) {
            if (ctype == 3)
                for (int k = 0; k < (int)len && k < n_pal; k++) palette[k][3] = body[k];
        } else if (!memcmp(type, "IDAT", 4)) {
            if (idat_n + len > idat_cap) {
                size_t cap = idat_cap ? idat_cap : 65536;
                while (cap < idat_n + len) cap *= 2;
                uint8_t* q = realloc(idat, cap);
                if (!q) {
                    free(idat);
                    return IMG_ERR_NOMEM;
                }
                idat = q, idat_cap = cap;
            }
            memcpy(idat + idat_n, body, len);
            idat_n += len;
        } else if (!memcmp(type, "IEND", 4)) {
            break;
        }
        i += 12 + (size_t)len;
    }
    int channels = ctype == 0 ? 1 : ctype == 2 ? 3 : ctype == 3 ? 1 : ctype == 4 ? 2 : ctype == 6 ? 4 : 0;
    if (!w || !h || !channels || w > 8192 || h > 8192) {
        free(idat);
        out->rgb = NULL;
        return ctype < 0 ? IMG_ERR_BROKEN : IMG_ERR_UNSUPPORTED;
    }
    if (depth != 8 && depth != 16) {
        free(idat);
        return IMG_ERR_UNSUPPORTED;
    }
    size_t bpp = (size_t)channels * (depth / 8);          // bytes a pixel takes in the raw rows
    size_t row = (size_t)w * bpp;
    size_t raw_n = (row + 1) * (size_t)h;
    uint8_t* raw = malloc(raw_n);
    out->rgb = malloc((size_t)w * (size_t)h * 3);
    if (!raw || !out->rgb || !idat) {
        free(idat);
        free(raw);
        free(out->rgb);
        out->rgb = NULL;
        return IMG_ERR_NOMEM;
    }
    size_t got = 0;
    int r = idat_n ? inflate_zlib(idat, idat_n, raw, raw_n, &got) : IMG_ERR_BROKEN;
    free(idat);
    if (r || got < raw_n) {
        free(raw);
        free(out->rgb);
        out->rgb = NULL;
        return IMG_ERR_BROKEN;
    }
    // undo the filters, row after row (the row before is the one just undone)
    for (int y = 0; y < h; y++) {
        uint8_t* cur = raw + (row + 1) * (size_t)y;
        int f = cur[0];
        uint8_t* p = cur + 1;
        const uint8_t* up = y ? raw + (row + 1) * (size_t)(y - 1) + 1 : NULL;
        for (size_t x = 0; x < row; x++) {
            int a = x >= bpp ? p[x - bpp] : 0;
            int u = up ? up[x] : 0;
            int c = up && x >= bpp ? up[x - bpp] : 0;
            switch (f) {
            case 0: break;
            case 1: p[x] = (uint8_t)(p[x] + a); break;
            case 2: p[x] = (uint8_t)(p[x] + u); break;
            case 3: p[x] = (uint8_t)(p[x] + ((a + u) >> 1)); break;
            case 4: p[x] = (uint8_t)(p[x] + unfilter_paeth(a, u, c)); break;
            default: {
                free(raw);
                free(out->rgb);
                out->rgb = NULL;
                return IMG_ERR_BROKEN;
            }
            }
        }
    }
    // to RGB, mixing alpha over bg
    uint32_t br = (bg >> 16) & 0xFF, bgc = (bg >> 8) & 0xFF, bb = bg & 0xFF;
    for (int y = 0; y < h; y++) {
        const uint8_t* p = raw + (row + 1) * (size_t)y + 1;
        uint8_t* o = out->rgb + (size_t)y * (size_t)w * 3;
        for (int x = 0; x < w; x++, o += 3) {
            int rr, gg, bl, al = 255;
            if (depth == 16) {
                const uint8_t* q = p + (size_t)x * bpp;
                if (channels >= 4) {
                    rr = q[0], gg = q[2], bl = q[4], al = q[6];
                } else if (channels == 3) {
                    rr = q[0], gg = q[2], bl = q[4];
                } else if (channels == 2) {
                    rr = gg = bl = q[0], al = q[2];
                } else {
                    rr = gg = bl = q[0];
                }
            } else if (ctype == 3) {
                int k = p[x];
                if (k >= n_pal) k = 0;
                rr = palette[k][0], gg = palette[k][1], bl = palette[k][2], al = palette[k][3];
            } else if (channels == 1) {
                rr = gg = bl = p[x];
            } else if (channels == 2) {
                rr = gg = bl = p[x * 2], al = p[x * 2 + 1];
            } else if (channels == 3) {
                rr = p[x * 3], gg = p[x * 3 + 1], bl = p[x * 3 + 2];
            } else {
                rr = p[x * 4], gg = p[x * 4 + 1], bl = p[x * 4 + 2], al = p[x * 4 + 3];
            }
            if (al < 255) {   // (over the caller's background: the transcript's card)
                rr = (rr * al + (int)br * (255 - al)) / 255;
                gg = (gg * al + (int)bgc * (255 - al)) / 255;
                bl = (bl * al + (int)bb * (255 - al)) / 255;
            }
            o[0] = (uint8_t)rr, o[1] = (uint8_t)gg, o[2] = (uint8_t)bl;
        }
    }
    free(raw);
    out->w = w, out->h = h;
    return IMG_OK;
}

void img_free(Img* im) {
    free(im->rgb);
    im->rgb = NULL, im->w = im->h = 0;
}

// ---- base64 ---------------------------------------------------------------------------------------------------

static int b64v(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
}

static uint8_t* b64_decode(const char* s, size_t n, size_t* out_n) {
    // a data: URL keeps its header out of the way
    if (n > 5 && !memcmp(s, "data:", 5)) {
        size_t i = 5;
        while (i < n && s[i] != ',') i++;
        if (i == n) return NULL;
        size_t skip = i + 1;
        s += skip, n -= skip;
    }
    uint8_t* out = malloc(n / 4 * 3 + 3);
    if (!out) return NULL;
    size_t o = 0;
    int acc = 0, bits_n = 0;
    for (size_t i = 0; i < n; i++) {
        int v = b64v((unsigned char)s[i]);
        if (v < 0) {
            if (s[i] == '=') break;
            if (s[i] == '\n' || s[i] == '\r' || s[i] == ' ' || s[i] == '\t') continue;
            free(out);
            return NULL;
        }
        acc = (acc << 6) | v;
        bits_n += 6;
        if (bits_n >= 8) {
            bits_n -= 8;
            out[o++] = (uint8_t)(acc >> bits_n);
            acc &= (1 << bits_n) - 1;
        }
    }
    *out_n = o;
    return out;
}

static const char B64[64] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char* img_b64_encode(const uint8_t* p, size_t n, size_t* out_n) {
    size_t o_n = (n + 2) / 3 * 4;
    char* out = malloc(o_n + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = (uint32_t)p[i] << 16;
        if (i + 1 < n) v |= (uint32_t)p[i + 1] << 8;
        if (i + 2 < n) v |= p[i + 2];
        out[o++] = B64[(v >> 18) & 63];
        out[o++] = B64[(v >> 12) & 63];
        out[o++] = i + 1 < n ? B64[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < n ? B64[v & 63] : '=';
    }
    out[o] = 0;
    if (out_n) *out_n = o;
    return out;
}

int img_decode_b64(const char* s, size_t n, Img* out, uint32_t bg) {
    out->w = out->h = 0, out->rgb = NULL;
    size_t n_bytes = 0;
    uint8_t* bytes = b64_decode(s, n, &n_bytes);
    if (!bytes) return IMG_ERR_BROKEN;
    int r = img_decode(bytes, n_bytes, out, bg);
    free(bytes);
    return r;
}

int img_load(const char* path, Img* out, uint32_t bg) {
    out->w = out->h = 0, out->rgb = NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return IMG_ERR_BROKEN;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return IMG_ERR_BROKEN;
    }
    uint8_t* d = malloc((size_t)n);
    if (!d) {
        fclose(f);
        return IMG_ERR_NOMEM;
    }
    size_t got = fread(d, 1, (size_t)n, f);
    fclose(f);
    int r = got == (size_t)n ? img_decode(d, got, out, bg) : IMG_ERR_BROKEN;
    free(d);
    return r;
}

// ---- scaling --------------------------------------------------------------------------------------------------

int img_scale(const Img* src, Img* dst, int max_w, int max_h) {
    dst->w = dst->h = 0, dst->rgb = NULL;
    if (!src->rgb || src->w <= 0 || src->h <= 0) return -1;
    int w = src->w, h = src->h;
    if (max_w > 0 && w > max_w) h = h * max_w / w, w = max_w;
    if (max_h > 0 && h > max_h) w = w * max_h / h, h = max_h;
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    dst->rgb = malloc((size_t)w * (size_t)h * 3);
    if (!dst->rgb) return -1;
    dst->w = w, dst->h = h;
    // one destination pixel averages the source block it covers (shrinking only: nothing to fill when enlarging)
    for (int y = 0; y < h; y++) {
        int sy0 = (int)((long)y * src->h / h), sy1 = (int)((long)(y + 1) * src->h / h);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int x = 0; x < w; x++) {
            int sx0 = (int)((long)x * src->w / w), sx1 = (int)((long)(x + 1) * src->w / w);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            uint32_t sum[3] = {0, 0, 0};
            int cnt = 0;
            for (int sy = sy0; sy < sy1; sy++)
                for (int sx = sx0; sx < sx1; sx++) {
                    const uint8_t* p = src->rgb + ((size_t)sy * src->w + sx) * 3;
                    sum[0] += p[0], sum[1] += p[1], sum[2] += p[2], cnt++;
                }
            uint8_t* o = dst->rgb + ((size_t)y * w + x) * 3;
            o[0] = (uint8_t)(sum[0] / cnt), o[1] = (uint8_t)(sum[1] / cnt), o[2] = (uint8_t)(sum[2] / cnt);
        }
    }
    return 0;
}
