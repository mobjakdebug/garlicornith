// Station AI: image models, through the API (imgen.h). A console this size cannot draw a picture with a diffusion
// model of its own, so the drawing is done by an image backend — a PC on the same network or a service — and this app
// is the model's front: POST /v1/images/generations is forwarded to it as it is (the OpenAI shape in, the same shape
// out), the picture that comes back is saved and shown on the console, and the answer reaches the client unchanged.
#include "imgen.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "api.h"
#include "api_proto.h"
#include "dl.h"
#include "img.h"
#include "png.h"
#include "ui.h"

#define IMG_CFG "/data/station_images.json"   // {"base_url", "api_key", "model"} — see README
#define IMG_ENDPOINT "/images/generations"
#define IMG_DIR "/data/station-ai/images"
#define IMG_MAX_PIXELS (4096 * 4096)          // (a bigger answer is a mistake, not a picture)

typedef struct {
    char base[400];     // "https://api.openai.com/v1"
    char key[300];      // "sk-…" ("" for a server on the LAN that wants none)
    char model[120];    // "gpt-image-1"
} ImgCfg;

static int cfg_str(JNode* pool, int root, const char* key, char* out, size_t cap, const char* def) {
    int n = json_get(pool, root, key);
    if (n < 0) {
        snprintf(out, cap, "%s", def ? def : "");
        return 0;
    }
    Sb s;
    sb_init(&s);
    json_str_sb(pool, n, &s);
    int ok = s.p && s.n < cap;
    snprintf(out, cap, "%s", s.p ? s.p : "");
    sb_free(&s);
    return ok ? 0 : -1;
}

// The backend's address, or -1 with why in msg (the file names the way to set one).
static int imgen_config(ImgCfg* c, char* msg, size_t msg_cap) {
    size_t n = 0;
    char* t = (char*)read_all(IMG_CFG, &n);
    memset(c, 0, sizeof(*c));
    if (!t || !n) {
        free(t);
        snprintf(msg, msg_cap,
                 "No image backend is set. Write " IMG_CFG " with {\"base_url\": \"http://192.168.1.10:8080/v1\", "
                 "\"api_key\": \"\", \"model\": \"sdxl\"} and ask again.");
        return -1;
    }
    JNode* pool = malloc(sizeof(JNode) * 256);
    int root = pool ? json_parse(t, n, pool, 256) : -1;
    free(t);
    if (!pool || root < 0 || pool[root].type != J_OBJ) {
        free(pool);
        snprintf(msg, msg_cap, IMG_CFG " is not a JSON object with base_url, api_key and model.");
        return -1;
    }
    int bad = cfg_str(pool, root, "base_url", c->base, sizeof(c->base), NULL);
    if (!bad && !c->base[0]) bad = cfg_str(pool, root, "url", c->base, sizeof(c->base), NULL);   // a whole endpoint
    cfg_str(pool, root, "api_key", c->key, sizeof(c->key), "");
    cfg_str(pool, root, "model", c->model, sizeof(c->model), "");
    free(pool);
    if (bad || !c->base[0]) {
        snprintf(msg, msg_cap, IMG_CFG " needs a base_url, as in \"http://192.168.1.10:8080/v1\".");
        return -1;
    }
    size_t bl = strlen(c->base);
    while (bl && c->base[bl - 1] == '/') c->base[--bl] = 0;
    if (!strstr(c->base, IMG_ENDPOINT)) snprintf(c->base + bl, sizeof(c->base) - bl, "%s", IMG_ENDPOINT);
    return 0;
}

static void imgen_reply(int fd, int status, const char* body, size_t n) {
    Sb h;
    sb_init(&h);
    sb_str(&h, status == 200 ? "HTTP/1.1 200 OK" : "HTTP/1.1 502 Bad Gateway");
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && api_send(fd, h.p, h.n) == 0) api_send(fd, body, n);
    sb_free(&h);
}

static void imgen_error(int fd, int code, const char* message, const char* type) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, NULL);
    if (!s.p) return;
    Sb h;
    sb_init(&h);
    sb_str(&h, code == 400   ? "HTTP/1.1 400 Bad Request"
              : code == 404 ? "HTTP/1.1 404 Not Found"
              : code == 501 ? "HTTP/1.1 501 Not Implemented"
                            : "HTTP/1.1 502 Bad Gateway");
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && api_send(fd, h.p, h.n) == 0) api_send(fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}

// Where the pictures go: /data/station-ai/images/img-<n>.png, numbering on from what is there.
static int imgen_save(const uint8_t* rgb, int w, int h, char* path, size_t cap) {
    mkdir("/data/station-ai", 0777);
    mkdir(IMG_DIR, 0777);
    for (unsigned i = 1; i < 10000; i++) {
        snprintf(path, cap, IMG_DIR "/img-%u.png", i);
        if (access(path, F_OK)) return png_write(path, rgb, w, h) ? -2 : 0;
    }
    return -1;
}

// The pictures of an answer: each data[i].b64_json is decoded, saved, and the first is shown on the console.
// Returns how many were saved (0 when the backend answered with URLs instead).
static int imgen_take(JNode* pool, int root, const char* prompt, char* first_path, size_t cap) {
    int data = json_get(pool, root, "data");
    int saved = 0;
    if (data < 0 || pool[data].type != J_ARR) return 0;
    for (int e = pool[data].first; e >= 0; e = pool[e].next) {
        int b64 = json_get(pool, e, "b64_json");
        if (b64 < 0 || pool[b64].type != J_STR) continue;
        Sb s;
        sb_init(&s);
        json_str_sb(pool, b64, &s);
        Img im;
        if (!s.p || img_decode_b64(s.p, s.n, &im, 0xFFFFFFFF)) {
            sb_free(&s);
            continue;
        }
        sb_free(&s);
        if ((long)im.w * im.h > IMG_MAX_PIXELS) {
            img_free(&im);
            continue;
        }
        char path[160];
        if (imgen_save(im.rgb, im.w, im.h, path, sizeof(path)) == 0) {
            if (!saved) snprintf(first_path, cap, "%s", path);
            saved++;
            say("imgen: %dx%d picture saved to %s", im.w, im.h, path);
            if (saved == 1) ui_image_add(&im, prompt, path);   // the first one, on the console
        }
        img_free(&im);
    }
    return saved;
}

// POST /v1/images/generations, on the main thread (api_run.inc calls it), so the screen can show the wait and the
// picture as it arrives.
void imgen_run(int fd, const char* body, size_t len) {
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    if (!pool) {
        imgen_error(fd, 500, "Out of memory.", "server_error");
        return;
    }
    int root = json_parse(body, len, pool, API_MAX_NODES);
    if (root < 0 || pool[root].type != J_OBJ) {
        free(pool);
        imgen_error(fd, 400, "The request is not a JSON object.", "invalid_request_error");
        return;
    }
    int prompt_node = json_get(pool, root, "prompt");
    Sb prompt;
    sb_init(&prompt);
    if (prompt_node < 0) {
        free(pool);
        imgen_error(fd, 400, "\"prompt\" is required.", "invalid_request_error");
        return;
    }
    json_str_sb(pool, prompt_node, &prompt);
    if (!prompt.p || !prompt.n) {
        sb_free(&prompt);
        free(pool);
        imgen_error(fd, 400, "\"prompt\" is empty.", "invalid_request_error");
        return;
    }
    // the request as the backend gets it: the fields an image server knows, and its own name for the model
    ImgCfg cfg;
    char why[400];
    if (imgen_config(&cfg, why, sizeof(why))) {
        sb_free(&prompt);
        free(pool);
        imgen_error(fd, 400, why, "invalid_request_error");
        return;
    }
    int size = json_get(pool, root, "size");
    int fmt = json_get(pool, root, "response_format");
    int quality = json_get(pool, root, "quality");
    int style = json_get(pool, root, "style");
    int n = json_get(pool, root, "n");
    Sb req, rfmt;
    sb_init(&req);
    sb_init(&rfmt);
    sb_str(&req, "{\"model\":");
    sb_json_str(&req, cfg.model, strlen(cfg.model));
    sb_str(&req, ",\"prompt\":");
    sb_json_str(&req, prompt.p, prompt.n);
    if (n >= 0 && pool[n].type == J_NUM && pool[n].num >= 1 && pool[n].num <= 4) {
        sb_str(&req, ",\"n\":");
        sb_int(&req, (long long)pool[n].num);
    }
    if (size >= 0) {   // (a string the client sent: decoded, then written out again escaped)
        sb_str(&req, ",\"size\":");
        json_str_sb(pool, size, &rfmt);
        sb_json_str(&req, rfmt.p ? rfmt.p : "", rfmt.n);
        rfmt.n = 0;
    }
    if (quality >= 0) {
        sb_str(&req, ",\"quality\":");
        json_str_sb(pool, quality, &rfmt);
        sb_json_str(&req, rfmt.p ? rfmt.p : "", rfmt.n);
        rfmt.n = 0;
    }
    if (style >= 0) {
        sb_str(&req, ",\"style\":");
        json_str_sb(pool, style, &rfmt);
        sb_json_str(&req, rfmt.p ? rfmt.p : "", rfmt.n);
        rfmt.n = 0;
    }
    // b64 unless the client says otherwise: a URL answer is something to pass on, not to keep
    sb_str(&req, ",\"response_format\":");
    if (fmt >= 0) {
        json_str_sb(pool, fmt, &rfmt);
        sb_json_str(&req, rfmt.p ? rfmt.p : "b64_json", rfmt.n ? rfmt.n : 8);
    } else {
        sb_json_str(&req, "b64_json", 8);
    }
    sb_str(&req, "}");
    sb_free(&rfmt);
    free(pool);
    if (!req.p || req.oom) {
        sb_free(&req);
        sb_free(&prompt);
        imgen_error(fd, 500, "Out of memory.", "server_error");
        return;
    }
    say("imgen: %zu bytes to %.200s for \"%.60s\"", req.n, cfg.base, prompt.p);
    draw_loading(-1, tr(S_IMG_WAIT));   // a card that says what the console is waiting for
    present();
    size_t out_n = 0;
    int status = 0;
    char* answer = dl_http_post(cfg.base, cfg.key[0] ? cfg.key : NULL, req.p, req.n, &out_n, &status);
    sb_free(&req);
    if (!answer) {
        sb_free(&prompt);
        imgen_error(fd, 502, "The image backend did not answer (is " IMG_CFG " right? the log has why).", "server_error");
        return;
    }
    if (status != 200) {
        // the backend's own error, when it sent one, says more than anything here
        size_t k = 0;
        while (k < out_n && (answer[k] == ' ' || answer[k] == '\r' || answer[k] == '\n')) k++;
        if (k < out_n && (answer[k] == '{' || answer[k] == '[')) imgen_reply(fd, 502, answer + k, out_n - k);
        else imgen_error(fd, 502, "The image backend refused the request.", "server_error");
        say("imgen: backend answered HTTP %d", status);
        free(answer);
        sb_free(&prompt);
        return;
    }
    JNode* pool2 = malloc(sizeof(JNode) * (API_MAX_NODES * 2));   // (a picture's answer is small: one entry, one text)
    int ok = pool2 ? json_parse(answer, out_n, pool2, API_MAX_NODES * 2) : -1;
    if (ok >= 0) {
        char path[160] = "";
        int saved = imgen_take(pool2, ok, prompt.p, path, sizeof(path));
        say("imgen: %d picture(s) saved", saved);
    } else {
        say("imgen: the backend's answer does not parse (%zu bytes)", out_n);
    }
    free(pool2);
    // the client gets exactly what the backend said
    imgen_reply(fd, 200, answer, out_n);
    free(answer);
    sb_free(&prompt);
}
