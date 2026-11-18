// Station AI: download a model from Hugging Face on the console itself (dl.h). The console's own HTTP client
// (libSceHttp + libSceSsl, the SDK's net_http sample's way in) fetches the repository's file list and then each
// file; what a dropped connection left stays and the next run asks for the rest (HTTP Range), so a 4.6 GB
// download in a living room does not start over. Nothing here runs unless the app has no weights to load.
#include "dl.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <orbis/CommonDialog.h>
#include <orbis/Http.h>
#include <orbis/ImeDialog.h>
#include <orbis/Net.h>
#include <orbis/NetCtl.h>
#include <orbis/Ssl.h>
#include <orbis/Sysmodule.h>
#include <orbis/libkernel.h>

#include "ui.h"

#define DL_DIR "/data/models"
#define DL_PICK "/data/station_model.txt"   // where the last download went, for the next start
#define DL_LAST "/data/station_hf.txt"      // the repository typed last, offered again
#define DL_DEFAULT_REPO "officialmobai/station-9b"   // what the downloader offers: the published model, ready to go
#define DL_API "https://huggingface.co/api/models/%s/tree/main?recursive=1"
#define DL_RESOLVE "https://huggingface.co/%s/resolve/main/%s"
#define DL_MAX_FILES 24
#define DL_MAX_TEXT (16 << 20)              // a file list that is bigger than this is not a file list
#define DL_CHUNK (64 << 10)
#define DL_MAX_JSON (48 << 20)              // an image backend's answer: a picture, base64, is a few megabytes

typedef struct {
    char path[160];      // as the repository has it (a .bin or an .idx)
    uint64_t size;
} DlFile;

static int net_pool = -1, ssl_ctx = -1, http_ctx = -1, tpl = -1;   // the console's HTTP client, set up once
static char model_dir[160];                                        // what dl_model_dir() answers

const char* dl_model_dir(void) {
    if (model_dir[0]) return model_dir;
    size_t n = 0;
    char* d = (char*)read_all(DL_PICK, &n);
    if (!d) return NULL;
    while (n && (d[n - 1] == '\n' || d[n - 1] == '\r' || d[n - 1] == ' ')) n--;
    if (!n || n >= sizeof(model_dir)) {
        free(d);
        return NULL;
    }
    memcpy(model_dir, d, n);
    model_dir[n] = 0;
    free(d);
    return model_dir;
}

// ---- the screen ----------------------------------------------------------------------------------------------
// A progress line (draw_loading draws the bar) and messages that stay a moment. Four languages, as the rest.
enum { DL_T_CONNECT, DL_T_LIST, DL_T_FILE, DL_T_DONE, DL_T_NOREPO, DL_T_NET, DL_T_NONE, DL_T_CANCEL,
       DL_T_WHAT, DL_T_INDEX, DL_T_KBD, DL_T_BADREPO, DL_T_NETWAIT, DL_T_DNS, N_DL };
static const char* DL_STR[N_DL][N_LANGS] = {
    {"Conectando con huggingface.co…", "Connecting to huggingface.co…", "Conectando a huggingface.co…",
     "Подключение к huggingface.co…"},
    {"Leyendo la lista de archivos…", "Reading the file list…", "Lendo a lista de arquivos…",
     "Читаю список файлов…"},
    {"%s: %u de %u MiB (%u %%)", "%s: %u of %u MiB (%u %%)", "%s: %u de %u MiB (%u %%)", "%s: %u из %u МиБ (%u %%)"},
    {"Modelo listo en %s. Pulsa X.", "Model ready in %s. Press X.", "Modelo pronto em %s. Pressione X.",
     "Модель готова в %s. Нажмите X."},
    {"Escribe un repositorio: usuario/modelo", "Type a repository: user/model", "Escreva um repositório: usuário/modelo",
     "Введите репозиторий: пользователь/модель"},
    {"Sin conexión: la consola no abre la red.", "No connection: the console cannot open the network.",
     "Sem conexão: o console não abre a rede.", "Нет связи: консоль не открывает сеть."},
    {"Ese repositorio no tiene model.idx y un .bin para la PS4.", "That repository has no model.idx and .bin for the PS4.",
     "Esse repositório não tem model.idx e .bin para o PS4.", "В этом репозитории нет model.idx и .bin для PS4."},
    {"Cancelado.", "Cancelled.", "Cancelado.", "Отменено."},
    {"Descargar un modelo", "Download a model", "Baixar um modelo", "Скачать модель"},
    {"Falta el índice del modelo (model.idx).", "The model's index (model.idx) is missing.",
     "Falta o índice do modelo (model.idx).", "Нет индекса модели (model.idx)."},
    {"Sin teclado: se usa %s.", "The keyboard is not available: using %s.", "Sem teclado: uso %s.",
     "Клавиатура недоступна: беру %s."},
    {"Escribe usuario/modelo, por ejemplo officialmobai/station-9b.",
     "Type user/model, for example officialmobai/station-9b.",
     "Escreva usuário/modelo, por exemplo officialmobai/station-9b.",
     "Введите пользователь/модель, например officialmobai/station-9b."},
    {"Esperando la red de la consola…", "Waiting for the console's network…", "Esperando a rede do console…",
     "Жду сеть консоли…"},
    {"huggingface.co no se resuelve: revisa el DNS de la consola (8.8.8.8).",
     "huggingface.co does not resolve: check the console's DNS (8.8.8.8).",
     "huggingface.co não resolve: veja o DNS do console (8.8.8.8).",
     "huggingface.co не разрешается: проверьте DNS консоли (8.8.8.8)."},
};

static const char* dl_tr(int id) { return DL_STR[id][ui_lang >= 0 && ui_lang < N_LANGS ? ui_lang : LANG_EN]; }

static void dl_progress(float frac, const char* line) {
    draw_loading(frac < 0 ? 0 : frac > 1 ? 1 : frac, line);
    present();
}

// The code the last failure left behind (an Orbis error, or an HTTP status), shown beside the message: a report
// from a console is a photograph of this screen, so the number has to be on it.
static unsigned dl_code;
static void dl_note(const char* msg);
static void dl_note_code(const char* msg, unsigned code) {
    char line[320];
    snprintf(line, sizeof(line), "%s [%08X]", msg, code);
    dl_note(line);
}

// A message on the screen for a second, and in the log: the player sees why nothing started.
static void dl_note(const char* msg) {
    say("dl: %s", msg);
    for (int i = 0; i < 6; i++) {
        dl_progress(1.0f, msg);
        sceKernelUsleep(250000);
    }
}

// ---- is there a network at all? ------------------------------------------------------------------------------
// Two questions the console answers itself, asked before a byte is sent, because both used to look like "stuck at
// the file list": an address (a download started in the seconds after the app appears could go out before the
// console had one) and a name (a console whose DNS is not answering fails every request at once).
static int net_address(char* ip, size_t cap) {
    OrbisNetCtlInfo in;
    sceNetCtlInit();
    for (int i = 0; i < 80; i++) {   // eight seconds, then it is not the network's slowness any more
        memset(&in, 0, sizeof(in));
        if (sceNetCtlGetInfo(ORBIS_NET_CTL_INFO_IP_ADDRESS, &in) == 0 && in.ip_address[0] &&
            strcmp(in.ip_address, "0.0.0.0")) {
            snprintf(ip, cap, "%s", in.ip_address);
            return 0;
        }
        sceKernelUsleep(100000);
    }
    return -1;
}

static int net_resolve(const char* host, char* out, size_t cap) {   // the address as dotted text, for the log
    int mem = sceNetPoolCreate("station_dns", 64 * 1024, 0);
    if (mem < 0) return mem;
    int rid = sceNetResolverCreate("station_resolver", mem, 0);
    if (rid < 0) {
        sceNetPoolDestroy(mem);
        return rid;
    }
    OrbisNetInAddr a;
    memset(&a, 0, sizeof(a));
    int rc = sceNetResolverStartNtoa(rid, host, &a, 5 * 1000 * 1000, 2, 0);
    sceNetResolverDestroy(rid);
    sceNetPoolDestroy(mem);
    if (rc == 0) {
        unsigned char* b = (unsigned char*)&a.s_addr;
        snprintf(out, cap, "%u.%u.%u.%u", b[0], b[1], b[2], b[3]);
    }
    return rc;
}

// ---- the keyboard: the repository's name ---------------------------------------------------------------------
// The console's own keyboard, set up the way the dialog's own samples do it: a zeroed setting, the languages the
// console carries, the buffer the player types into, then GetStatus until the dialog stops. What the first version
// did instead is why nothing could be typed: a status of NONE was taken for "it is over" after three hundred frames,
// so the dialog was terminated (sceImeDialogTerm) before the system had drawn it, and every run ended in
// "Cancelled". It now waits for the dialog, and says so in the log when the firmware refuses to open one at all.
static void dl_wide(const char* s, wchar_t* out, int max) {   // (the field is ASCII: one byte a character)
    int i = 0;
    for (; s[i] && i < max - 1; i++) out[i] = (wchar_t)(unsigned char)s[i];
    out[i] = 0;
}

// 1 = a name was typed (out holds it), 0 = the player cancelled, -1 = there is no keyboard to type on.
static int dl_ask(char* out, int max) {
    static wchar_t buf[257], placeholder[160], title[64];
    memset(buf, 0, sizeof(buf));
    dl_wide(dl_tr(DL_T_NOREPO), placeholder, 160);
    dl_wide("Station AI", title, 64);
    dl_wide(out, buf, 256);   // what is already known arrives typed: SEND accepts it as it is
    OrbisImeDialogSetting s;
    memset(&s, 0, sizeof(s));
    s.userId = (uint32_t)ui_user_id();
    s.type = ORBIS_TYPE_DEFAULT;
    s.supportedLanguages = 0x0001FFFFull;   // the console's seventeen keyboard languages (ORBIS_IME_LANG_*)
    s.enterLabel = ORBIS_BUTTON_LABEL_SEND;
    s.maxTextLength = 256;
    s.inputTextBuffer = buf;
    s.posx = SCREEN_W / 2;
    s.posy = SCREEN_H / 2;
    s.horizontalAlignment = ORBIS_H_CENTER;
    s.verticalAlignment = ORBIS_V_CENTER;
    s.title = title;
    s.placeholder = placeholder;
    int rc = sceImeDialogInit(&s, NULL);
    if (rc != 0) {
        say("dl: sceImeDialogInit 0x%08X (no keyboard)", (unsigned)rc);
        return -1;
    }
    int running = 0, ticks = 0;
    for (;;) {
        OrbisDialogStatus st = sceImeDialogGetStatus();
        if (st == ORBIS_DIALOG_STATUS_STOPPED) break;
        if (st == ORBIS_DIALOG_STATUS_RUNNING) {
            running = 1;   // (from here on the dialog is what the player looks at: nothing more is drawn under it)
        } else if (!running) {
            if (++ticks > 2000) {   // twenty seconds and it never came up: this screen is not what is waited on
                say("dl: the keyboard never appeared");
                sceImeDialogTerm();
                return -1;
            }
            if (ticks % 5 == 0) dl_progress(0.0f, dl_tr(DL_T_WHAT));
        }
        sceKernelUsleep(10000);
    }
    OrbisDialogResult res;
    memset(&res, 0, sizeof(res));
    rc = sceImeDialogGetResult(&res);
    sceImeDialogTerm();
    say("dl: keyboard: end status %d (get 0x%08X)", (int)res.endstatus, (unsigned)rc);
    if (res.endstatus != ORBIS_DIALOG_OK) return 0;
    int n = 0;
    for (int i = 0; buf[i] && n < max - 1; i++)
        if (buf[i] > 32 && buf[i] < 127) out[n++] = (char)buf[i];
    out[n] = 0;
    return n > 0 ? 1 : 0;
}

// ---- the console's HTTP client -------------------------------------------------------------------------------
static int ssl_skip(int32_t id, uint32_t err, void* const cert[], int32_t n_cert, void* arg) {
    (void)id; (void)err; (void)cert; (void)n_cert; (void)arg;
    return 1;   // the console's CA list is older than the web: take the certificate (the files are public)
}

static int net_up(void) {
    if (tpl > 0) return 0;
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_NET);
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_HTTP);
    sceSysmoduleLoadModuleInternal(ORBIS_SYSMODULE_INTERNAL_SSL);
    int r = sceNetInit();   // (0, or already up: the API may have started it)
    say("dl: sceNetInit 0x%08X", (unsigned)r);
    if (r < 0) dl_code = (unsigned)r;
    if (net_pool <= 0) net_pool = sceNetPoolCreate("station_ai_dl", 256 * 1024, 0);
    if (net_pool < 0) {
        say("dl: sceNetPoolCreate 0x%08X", (unsigned)net_pool);
        dl_code = (unsigned)net_pool;
        return -1;
    }
    // The pools are the SDK's own numbers times four: a modern server's chain (huggingface.co's is Cloudflare's, two
    // certificates and a long SAN list) needs more room than the 128 KiB the sample sizes for, and TLS that runs out
    // of memory fails the moment it starts.
    if (ssl_ctx <= 0) ssl_ctx = sceSslInit(512 * 1024);
    if (ssl_ctx < 0) {
        say("dl: sceSslInit 0x%08X", (unsigned)ssl_ctx);
        dl_code = (unsigned)ssl_ctx;
        return -1;
    }
    if (http_ctx <= 0) http_ctx = sceHttpInit(net_pool, ssl_ctx, 256 * 1024);
    if (http_ctx < 0) {
        say("dl: sceHttpInit 0x%08X", (unsigned)http_ctx);
        dl_code = (unsigned)http_ctx;
        return -1;
    }
    tpl = sceHttpCreateTemplate(http_ctx, "Station AI/1.23 (PLAYSTATION 4)", ORBIS_HTTP_VERSION_1_1, 1);
    if (tpl < 0) {
        say("dl: sceHttpCreateTemplate 0x%08X", (unsigned)tpl);
        dl_code = (unsigned)tpl;
        return -1;
    }
    int cb = sceHttpsSetSslCallback(tpl, ssl_skip, NULL);
    // The console's certificate store and clock are old, and a check it cannot make used to end the handshake
    // before the callback above was ever asked: the CN and the two dates are the ones to switch off.
    int opt = sceHttpsDisableOption(tpl, 0x1C);   // CN_CHECK | NOT_AFTER_CHECK | NOT_BEFORE_CHECK
    say("dl: TLS: callback 0x%08X, checks off 0x%08X", (unsigned)cb, (unsigned)opt);
    sceHttpSetConnectTimeOut(tpl, 10 * 1000000);   // ten seconds to answer, or the screen says why
    // (the header declares this with no prototype, as it does some others: the SDK's call takes the template and usec)
    sceHttpSetRecvTimeOut(tpl, 15 * 1000000);    // fifteen seconds of silence is a dead connection, and the
                                                 // download screen shows the code rather than waiting for ever
    say("dl: HTTP client ready (net pool %d, ssl %d, http %d, template %d)", net_pool, ssl_ctx, http_ctx, tpl);
    return 0;
}

static int dl_ieq(const char* a, const char* b, int n) {
    for (int i = 0; i < n; i++) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x += 32;
        if (y >= 'A' && y <= 'Z') y += 32;
        if (x != y || !x) return x == y;
    }
    return 1;
}

// The "Location:" of a response (an absolute URL, or a path on the same host as from), into out.
static int dl_location(const char* hdr, size_t n, const char* from, char* out, size_t cap) {
    size_t i = 0;
    while (i + 10 < n) {
        if (dl_ieq(hdr + i, "location:", 9)) {
            const char* p = hdr + i + 9;
            while (*p == ' ') p++;
            size_t k = 0;
            while (p[k] && p[k] != '\r' && p[k] != '\n' && k < cap - 1) {
                out[k] = p[k];
                k++;
            }
            out[k] = 0;
            if (k > 4 && dl_ieq(out, "http", 4)) return 1;
            if (k && out[0] == '/') {   // the same host
                const char* scheme = strstr(from, "://");
                const char* host_end = scheme ? strchr(scheme + 3, '/') : NULL;
                size_t pre = host_end ? (size_t)(host_end - from) : strlen(from);
                char tmp[700];
                if (pre + k + 1 >= cap) return 0;
                memcpy(tmp, from, pre);
                tmp[pre] = 0;
                char rel[700];
                snprintf(rel, sizeof(rel), "%s", out);
                snprintf(out, cap, "%s%s", tmp, rel);
            }
            return k > 0;
        }
        const char* nl = strchr(hdr + i, '\n');
        if (!nl) break;
        i = (size_t)(nl - hdr) + 1;
    }
    return 0;
}

// A GET that follows up to six redirects (huggingface.co sends the files to its CDN). *req is the open request to
// read the body from.
static int http_open(const char* url0, const char* range, int32_t* req_out, int* status_out) {
    char url[700];
    snprintf(url, sizeof(url), "%s", url0);
    for (int hop = 0; hop < 6; hop++) {
        int32_t conn = sceHttpCreateConnectionWithURL(tpl, url, 1);
        if (conn < 0) return (int)conn;
        int32_t req = sceHttpCreateRequestWithURL(conn, ORBIS_METHOD_GET, url, 0);
        if (req < 0) {
            sceHttpDeleteConnection(conn);
            return (int)req;
        }
        if (range && range[0]) sceHttpAddRequestHeader(req, "Range", range, 0);
        int r = sceHttpSendRequest(req, NULL, 0);
        int32_t st = 0;
        if (r >= 0) r = sceHttpGetStatusCode(req, &st);
        if (r < 0) {
            say("dl: GET %s -> 0x%08X", url, (unsigned)r);
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            return r;
        }
        if (st == 301 || st == 302 || st == 303 || st == 307 || st == 308) {
            char* hdr = NULL;
            size_t hn = 0;
            char next[700];
            int ok = sceHttpGetAllResponseHeaders(req, &hdr, &hn) >= 0 && hdr && dl_location(hdr, hn, url, next, sizeof(next));
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            if (!ok) {
                say("dl: %d without a Location", (int)st);
                return -1;
            }
            say("dl: %d -> %.200s", (int)st, next);
            snprintf(url, sizeof(url), "%s", next);
            continue;
        }
        *req_out = req;
        *status_out = (int)st;
        return 0;
    }
    say("dl: too many redirects from %.200s", url0);
    return -1;
}

// A whole small body (the file list) into a buffer the caller frees.
static char* http_text(const char* url, size_t* len, int* err) {
    int32_t req = -1;
    int st = 0;
    *err = http_open(url, NULL, &req, &st);
    if (*err) return NULL;
    if (st != 200) {
        say("dl: %s -> HTTP %d", url, st);
        sceHttpDeleteRequest(req);
        *err = -st;
        dl_code = (unsigned)st;
        return NULL;
    }
    size_t cap = 1 << 16, n = 0;
    char* buf = malloc(cap);
    while (buf) {
        if (n + DL_CHUNK + 1 > cap) {
            if (cap >= DL_MAX_TEXT) break;
            char* nb = realloc(buf, cap * 2);
            if (!nb) break;
            buf = nb;
            cap *= 2;
        }
        int r = sceHttpReadData(req, buf + n, DL_CHUNK);
        if (r < 0) {
            *err = r;
            free(buf);
            buf = NULL;
            break;
        }
        if (r == 0) break;
        n += (size_t)r;
    }
    sceHttpDeleteRequest(req);
    if (!buf) return NULL;
    buf[n] = 0;
    *len = n;
    return buf;
}

// ---- a POST, for the image backend (imgen.c) -----------------------------------------------------------------
// The same client the downloads use, with a body: a JSON request goes out, the answer (a generation's JSON, or one
// picture's) comes back whole. Redirects are followed only where the method is kept (307, 308).

char* dl_http_post(const char* url0, const char* auth, const char* body, size_t blen, size_t* out_len, int* status) {
    *out_len = 0, *status = 0;
    if (net_up()) {
        *status = -1;
        return NULL;
    }
    char url[700];
    snprintf(url, sizeof(url), "%s", url0);
    int32_t req = -1, conn = -1;
    int st = 0;
    for (int hop = 0; hop < 4; hop++) {
        conn = sceHttpCreateConnectionWithURL(tpl, url, 1);
        if (conn < 0) {
            say("dl: POST %.200s: connection 0x%08X", url, (unsigned)conn);
            *status = -1;
            return NULL;
        }
        req = sceHttpCreateRequestWithURL(conn, ORBIS_METHOD_POST, url, (uint64_t)blen);
        if (req < 0) {
            sceHttpDeleteConnection(conn);
            *status = -1;
            return NULL;
        }
        sceHttpAddRequestHeader(req, "Content-Type", "application/json", 0);
        if (auth && auth[0]) sceHttpAddRequestHeader(req, "Authorization", auth, 0);
        int r = sceHttpSendRequest(req, body, blen);
        if (r >= 0) r = sceHttpGetStatusCode(req, &st);
        if (r < 0) {
            say("dl: POST %.200s -> 0x%08X", url, (unsigned)r);
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            *status = -1;
            return NULL;
        }
        if (st == 307 || st == 308) {
            char* hdr = NULL;
            size_t hn = 0;
            char next[700];
            int ok = sceHttpGetAllResponseHeaders(req, &hdr, &hn) >= 0 && hdr && dl_location(hdr, hn, url, next, sizeof(next));
            sceHttpDeleteRequest(req);
            sceHttpDeleteConnection(conn);
            if (!ok) {
                *status = st;
                return NULL;
            }
            snprintf(url, sizeof(url), "%s", next);
            continue;
        }
        break;
    }
    *status = st;
    size_t cap = 1 << 16, n = 0;
    char* buf = malloc(cap);
    while (buf) {
        if (n + DL_CHUNK + 1 > cap) {
            if (cap >= DL_MAX_JSON) break;
            char* nb = realloc(buf, cap * 2);
            if (!nb) break;
            buf = nb;
            cap *= 2;
        }
        int r = sceHttpReadData(req, buf + n, DL_CHUNK);
        if (r < 0) {
            free(buf);
            buf = NULL;
            break;
        }
        if (r == 0) break;
        n += (size_t)r;
    }
    sceHttpDeleteRequest(req);
    if (!buf) {
        *status = -1;
        return NULL;
    }
    buf[n] = 0;
    *out_len = n;
    return buf;
}

// ---- the file list: Hugging Face's tree JSON ----------------------------------------------------------------
// {"type":"file","oid":"…","size":123,"path":"model.idx"} objects, scanned by hand: only the two fields this
// needs, and only the files the engine reads (.bin and .idx, at the repository's root).
static int dl_wants(const char* p) {
    size_t n = strlen(p);
    if (!n || strchr(p, '/')) return 0;   // (model.idx and the weights live at the root)
    return n > 4 && (!strcmp(p + n - 4, ".idx") || !strcmp(p + n - 4, ".bin"));
}

static int dl_is_idx(const char* p) {
    size_t n = strlen(p);
    return n > 4 && !strcmp(p + n - 4, ".idx");
}

static int dl_parse(const char* js, size_t n, DlFile* files, int* nf, uint64_t* total) {
    *nf = 0;
    *total = 0;
    for (size_t i = 0; i < n; i++) {
        if (js[i] != '{') continue;
        uint64_t size = 0;
        char path[160] = "";
        int depth = 0;
        for (size_t j = i + 1; j < n; j++) {
            if (js[j] == '{') depth++;
            else if (js[j] == '}' && depth-- == 0) break;   // (the entry's own brace: "lfs" nests one)
            if (!strncmp(js + j, "\"size\":", 7)) {
                j += 7;
                while (js[j] == ' ') j++;
                for (; js[j] >= '0' && js[j] <= '9'; j++) size = size * 10 + (uint64_t)(js[j] - '0');
            } else if (!strncmp(js + j, "\"path\":\"", 8)) {
                size_t k = 0;
                j += 8;
                while (js[j] && js[j] != '"' && k < sizeof(path) - 1) path[k++] = js[j++];
                path[k] = 0;
            }
        }
        if (!path[0] || !size || !dl_wants(path) || *nf >= DL_MAX_FILES) continue;
        int dup = 0;
        for (int k = 0; k < *nf; k++) dup |= !strcmp(files[k].path, path);
        if (dup) continue;
        snprintf(files[*nf].path, sizeof(files[*nf].path), "%s", path);
        files[*nf].size = size;
        *total += size;
        (*nf)++;
    }
    return *nf;
}

// ---- one file: resume, progress, verify ---------------------------------------------------------------------
static uint64_t dl_file_size(const char* path) {
    struct stat st;
    return stat(path, &st) == 0 ? (uint64_t)st.st_size : 0;
}

static void dl_mkdirs(const char* path) {   // every directory on the way to this file
    char tmp[400];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char* p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0777);
            *p = '/';
        }
}

// path gets exactly `size` bytes from `url`; what an earlier run left of it (path.part) is kept and the rest is
// asked for with a Range, so a download that a standby or a lost router cut continues where it stopped.
static int dl_file(const char* url, const char* path, uint64_t size, uint64_t* done, uint64_t total) {
    char part[420];
    snprintf(part, sizeof(part), "%s.part", path);
    dl_mkdirs(path);
    uint64_t have = dl_file_size(path);
    if (have == size) {
        *done += size;
        return 0;
    }
    if (have > size) unlink(path);   // (a leftover of another model: start over)
    uint64_t got = dl_file_size(part);
    if (got > size) {
        unlink(part);
        got = 0;
    }
    int fd = open(part, got ? (O_WRONLY | O_CREAT | O_APPEND) : (O_WRONLY | O_CREAT | O_TRUNC), 0666);
    if (fd < 0) {
        say("dl: cannot write %s", part);
        return -1;
    }
    char range[64] = "";
    if (got) snprintf(range, sizeof(range), "bytes=%llu-", (unsigned long long)got);
    int32_t req = -1;
    int st = 0;
    int r = http_open(url, range, &req, &st);
    if (r == 0 && got && st == 200) {   // the server sent all of it: start the file over
        say("dl: no resume for %s (HTTP 200)", path);
        close(fd);
        fd = open(part, O_WRONLY | O_CREAT | O_TRUNC, 0666);
        got = 0;
        if (fd < 0) {
            sceHttpDeleteRequest(req);
            return -1;
        }
    }
    if (r == 0 && st != 200 && st != 206) {
        say("dl: %s -> HTTP %d", url, st);
        r = -1;
    }
    char* buf = r == 0 ? malloc(DL_CHUNK) : NULL;
    int ticks = 0;
    while (r == 0 && buf) {
        int n = sceHttpReadData(req, buf, DL_CHUNK);
        if (n < 0) {
            say("dl: read error 0x%08X at %llu of %s", (unsigned)n, (unsigned long long)got, path);
            dl_code = (unsigned)n;
            r = -1;
            break;
        }
        if (n == 0) break;
        if (write(fd, buf, (size_t)n) != n) {
            say("dl: write error at %llu of %s", (unsigned long long)got, path);
            r = -1;
            break;
        }
        got += (uint64_t)n;
        if (++ticks >= 8) {   // (half a megabyte between frames: the picture stays smooth)
            ticks = 0;
            const char* nm = strrchr(path, '/');
            char line[300];
            snprintf(line, sizeof(line), dl_tr(DL_T_FILE), nm ? nm + 1 : path,
                     (unsigned)((*done + got) >> 20), (unsigned)(total >> 20),
                     (unsigned)(total ? (*done + got) * 100 / total : 0));
            dl_progress(total ? (float)(*done + got) / (float)total : 0.0f, line);
        }
    }
    free(buf);
    if (req >= 0) sceHttpDeleteRequest(req);
    close(fd);
    if (r) return -1;
    if (got != size) {
        say("dl: %s has %llu of %llu bytes (it stays as .part: running it again continues)", path,
            (unsigned long long)got, (unsigned long long)size);
        return -1;
    }
    unlink(path);
    if (rename(part, path)) {
        say("dl: cannot rename %s", part);
        return -1;
    }
    *done += size;
    say("dl: %s: %llu bytes", path, (unsigned long long)size);
    return 0;
}

// ---- the file list taken from the index itself ---------------------------------------------------------------
// model.idx and mtp.idx carry the name and the size of the weights file they describe (a 128-byte path at 52, the
// size at 180, see chat/engine.c), so two small GETs say everything the API's JSON would. That is the second way
// in: an API that is blocked, changed or slow no longer stops a download, and the file that names a weight is the
// file itself, so the two can never disagree.
static int dl_idx_add(DlFile* files, int* nf, uint64_t* total, const char* name, uint64_t size) {
    if (!size || *nf >= DL_MAX_FILES) return 0;
    snprintf(files[*nf].path, sizeof(files[*nf].path), "%s", name);
    files[*nf].size = size;
    *total += size;
    (*nf)++;
    return 1;
}

static int dl_plan_from_idx(const char* repo, DlFile* files, int* nf, uint64_t* total) {
    static const char* idx[2] = {"model.idx", "mtp.idx"};
    *nf = 0;
    *total = 0;
    for (int i = 0; i < 2; i++) {
        char url[700];
        snprintf(url, sizeof(url), DL_RESOLVE, repo, idx[i]);
        size_t n = 0;
        int err = 0;
        char* d = http_text(url, &n, &err);
        if (!d) {
            say("dl: %s: no answer (0x%08X)", idx[i], (unsigned)err);
            if (i == 0) {
                dl_code = (unsigned)(err < 0 ? -err : err);
                return -1;
            }
            continue;   // (no draft head in this repository: the weights alone are a model)
        }
        if (n > 188 && (!memcmp(d, "PMDL", 4) || !memcmp(d, "PMQ5", 4))) {
            char name[129];
            uint64_t size = 0;
            memcpy(name, d + 52, 128);
            name[128] = 0;
            memcpy(&size, d + 180, 8);
            if (dl_idx_add(files, nf, total, idx[i], (uint64_t)n))   // the index last, as the API's order does
                say("dl: %s names %s (%llu bytes)", idx[i], name, (unsigned long long)size);
            dl_idx_add(files, nf, total, name, size);
        } else {
            say("dl: %s is not an index of this app's", idx[i]);
            if (i == 0) return -1;
        }
        free(d);
    }
    return *nf ? 0 : -1;
}

// ---- the whole thing ----------------------------------------------------------------------------------------
static void dl_pick_write(const char* dir) {
    int fd = open(DL_PICK, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) return;
    if (write(fd, dir, strlen(dir)) < 0) say("dl: cannot write %s", DL_PICK);
    if (write(fd, "\n", 1) < 0) say("dl: cannot write %s", DL_PICK);
    close(fd);
}

int dl_run(void) {
    char repo[160] = "";
    size_t n = 0;
    char* last = (char*)read_all(DL_LAST, &n);
    if (last) {
        char* nl = strpbrk(last, "\r\n");
        if (nl) *nl = 0;
        snprintf(repo, sizeof(repo), "%s", last);
        free(last);
    }
    if (!repo[0]) snprintf(repo, sizeof(repo), "%s", DL_DEFAULT_REPO);   // out of the box: the published model
    int ask = dl_ask(repo, sizeof(repo));
    if (ask < 0) {   // no keyboard: go on with the repository already known rather than send the player away
        char line[300];
        snprintf(line, sizeof(line), dl_tr(DL_T_KBD), repo);
        dl_note(line);
    } else if (!ask) {
        dl_note(dl_tr(DL_T_CANCEL));
        return 0;
    }
    // "https://huggingface.co/user/model" or "user/model": the API wants the two names
    char* p = strstr(repo, "huggingface.co/");
    if (p) memmove(repo, p + strlen("huggingface.co/"), strlen(p + strlen("huggingface.co/")) + 1);
    for (char* s = repo; *s; s++)
        if (*s == '\n' || *s == '\r' || *s == ' ') *s = 0;
    while (repo[0] && repo[strlen(repo) - 1] == '/') repo[strlen(repo) - 1] = 0;
    say("dl: repository \"%s\"", repo);
    if (!strchr(repo, '/')) {
        dl_note(dl_tr(DL_T_BADREPO));
        return 0;
    }
    int fd = open(DL_LAST, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd >= 0) {
        if (write(fd, repo, strlen(repo)) < 0) say("dl: cannot write %s", DL_LAST);
        close(fd);
    }
    dl_progress(0.0f, dl_tr(DL_T_CONNECT));
    dl_code = 0;
    if (net_up()) {
        dl_note_code(dl_tr(DL_T_NET), dl_code);
        return 0;
    }
    char ip[32] = "", host[32] = "";
    dl_progress(0.0f, dl_tr(DL_T_NETWAIT));
    int have_ip = net_address(ip, sizeof(ip));
    say("dl: address %s", have_ip ? "(none)" : ip);
    int dns = net_resolve("huggingface.co", host, sizeof(host));
    say("dl: huggingface.co -> %s (0x%08X)", dns ? "(no answer)" : host, (unsigned)dns);
    if (dns) {   // said, not obeyed: libSceHttp has its own resolver, and the request is the real test
        char line[320];
        snprintf(line, sizeof(line), "%s [%08X]", dl_tr(DL_T_DNS), (unsigned)dns);
        dl_note(line);
        dl_code = (unsigned)dns;
    }
    char url[700];
    snprintf(url, sizeof(url), DL_API, repo);
    dl_progress(0.0f, dl_tr(DL_T_LIST));
    int err = 0;
    size_t jn = 0;
    char* js = http_text(url, &jn, &err);
    if (!js) {   // a resolver or a router drops the first one often enough: one more, a moment later
        say("dl: the list failed (0x%08X); trying once more", (unsigned)err);
        sceKernelUsleep(1500000);
        dl_progress(0.0f, dl_tr(DL_T_LIST));
        js = http_text(url, &jn, &err);
    }
    DlFile files[DL_MAX_FILES];
    int nf = 0;
    uint64_t total = 0;
    if (js) {
        dl_parse(js, jn, files, &nf, &total);
        free(js);
    }
    say("dl: %d files, %llu bytes%s", nf, (unsigned long long)total, err ? " (the request failed)" : "");
    if (!nf) {   // the API said nothing useful: the index itself knows what it needs
        say("dl: the API gave no list (0x%08X); asking the index", (unsigned)err);
        dl_progress(0.0f, dl_tr(DL_T_LIST));
        if (dl_plan_from_idx(repo, files, &nf, &total)) {
            // A 4xx from the API (a name that is not there, or a repository that is not public) is not a network
            // fault: the two used to be the same message, which sent the player looking at the router.
            int missing = dl_code > 0x80000000u ? 0 : (dl_code >= 400 && dl_code < 600);
            say("dl: nothing to fetch (%s)", missing ? "the repository is not there" : "no answer");
            dl_note_code(dl_tr(missing ? DL_T_NONE : DL_T_NET), dl_code);
            return 0;
        }
        say("dl: the index gave %d file(s), %llu bytes", nf, (unsigned long long)total);
    }
    int have_idx = 0;
    for (int i = 0; i < nf; i++) have_idx |= dl_is_idx(files[i].path);
    if (!have_idx) {
        dl_note(dl_tr(DL_T_INDEX));
        return 0;
    }
    const char* tail = strrchr(repo, '/') + 1;
    char dir[300];
    snprintf(dir, sizeof(dir), "%s/%s", DL_DIR, tail);
    dl_mkdirs(dir);
    uint64_t done = 0;
    for (int pass = 0; pass < 2; pass++) {   // the weights first, the index last: while it is missing, no model
        for (int i = 0; i < nf; i++) {
            if (dl_is_idx(files[i].path) != (pass == 1)) continue;
            char fu[900], fp[500];
            snprintf(fu, sizeof(fu), DL_RESOLVE, repo, files[i].path);
            snprintf(fp, sizeof(fp), "%s/%s", dir, files[i].path);
            if (dl_file(fu, fp, files[i].size, &done, total)) {
                dl_note_code(dl_tr(DL_T_NET), dl_code);   // (with the code: which file, and what said no)
                return 0;
            }
        }
    }
    dl_pick_write(dir);
    snprintf(model_dir, sizeof(model_dir), "%s", dir);
    char line[400];
    snprintf(line, sizeof(line), dl_tr(DL_T_DONE), dir);
    say("dl: %s", line);
    dl_progress(1.0f, line);
    sceKernelUsleep(1500000);
    return 1;
}
