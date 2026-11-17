// The app's screens rendered on this machine, with no console: the same chat/ui.c, into a PNG (pc/preview.sh).
//   gcc -D_WIN32 -O2 -I chat pc/ui_preview.c chat/ui.c chat/png.c -lm -o build/ui_preview
//   build/ui_preview build/shots/chat.png chat assets
// The point is to see a layout change before flashing a 4.9 GB package to a console, and to do it in seconds.
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "png.h"
#include <math.h>

#include "ui.h"

uint32_t* back;

void present(void) {}
int ui_user_id(void) { return 0; }

void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

uint8_t* read_all(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* p = malloc((size_t)n + 1);
    if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p);
        p = NULL;
    }
    fclose(f);
    if (p && size) *size = (size_t)n;
    return p;
}

static void shot(const char* path) {
    uint8_t* rgb = malloc((size_t)SCREEN_W * SCREEN_H * 3);
    for (int i = 0; i < SCREEN_W * SCREEN_H; i++) {
        uint32_t p = back[i];
        rgb[i * 3 + 0] = (uint8_t)(p >> 16);
        rgb[i * 3 + 1] = (uint8_t)(p >> 8);
        rgb[i * 3 + 2] = (uint8_t)p;
    }
    int r = png_write(path, rgb, SCREEN_W, SCREEN_H);
    printf("%s: %s\n", path, r ? "FAILED" : "written");
    free(rgb);
}

static void demo_messages(void) {
    msgs_clear();
    Msg* m = msg_new(MSG_USER);
    msg_append(m, "Explícame qué es un transformer y luego dame un ejemplo de código en Python que cuente "
                  "palabras en un texto. También compara las tres arquitecturas en una tabla.",
              strlen("Explícame qué es un transformer y luego dame un ejemplo de código en Python que cuente "
                     "palabras en un texto. También compara las tres arquitecturas en una tabla."));
    m = msg_new(MSG_AI);
    const char* a =
        "Un **transformer** es una red que procesa una secuencia entera a la vez: cada token mira a todos los demás "
        "con *atención*, en vez de recorrerla en orden como una RNN.\n\n"
        "## Cómo funciona\n\n"
        "1. Cada token se convierte en un vector.\n"
        "2. La atención mezcla esos vectores según lo que se parecen: `softmax(QK^T / sqrt(d)) V`.\n"
        "3. Una red densa (MLP) transforma cada posición por separado.\n"
        "4. Se repite por capas, y la salida predice el siguiente token.\n\n"
        "```python\n"
        "from collections import Counter\n\n"
        "def palabras(texto):\n"
        "    return Counter(texto.lower().split())\n\n"
        "print(palabras(\"hola hola mundo\").most_common(2))\n"
        "```\n\n"
        "| Modelo | Ventaja | Coste |\n"
        "| --- | --- | --- |\n"
        "| Transformer | calidad y paralelismo | memoria cuadrática |\n"
        "| RNN | secuencias largas baratas | lenta, se olvida |\n"
        "| CNN | local y rápida | poca visión global |\n\n"
        "> En esta consola la atención usa la GPU de la PS4, con la caché en 4 bits.";
    msg_append(m, a, strlen(a));
    m = msg_new(MSG_NOTE);
    msg_append(m, "Contexto lleno: he empezado una conversación nueva.",
              strlen("Contexto lleno: he empezado una conversación nueva."));
}

int main(int argc, char** argv) {
    const char* out = argc > 1 ? argv[1] : "build/shots/chat.png";
    const char* screen = argc > 2 ? argv[2] : "chat";
    const char* assets = argc > 3 ? argv[3] : "chat/assets";
    mkdir("build", 0777);
    mkdir("build/shots", 0777);
    back = calloc((size_t)SCREEN_W * SCREEN_H, 4);
    if (ui_init(assets)) {
        printf("ui_init failed: %s\n", ui_error);
        return 1;
    }
    ui_lang = argc > 4 ? atoi(argv[4]) : LANG_ES;
    ChatView v;
    memset(&v, 0, sizeof(v));
    v.model = "Ornith-1.5-9B";
    v.pos = 4180;
    v.max_seq = 40960;
    v.temperature = 0.65f;
    v.tok_per_s = 12.4;
    v.time = 3.5;
    v.think_avail = 1;
    if (!strcmp(screen, "empty")) {
        draw_chat(&v);
    } else if (!strcmp(screen, "thinking")) {
        demo_messages();
        v.generating = 1;
        v.thinking = 1;
        v.think_n = 96;
        v.think_s = 2.1;
        v.think_text = "Vale: primero el qué, luego el código.\nUn transformer procesa la secuencia entera a la "
                       "vez, y la atención…";
        v.think_len = strlen(v.think_text);
        draw_chat(&v);
    } else if (!strcmp(screen, "generating")) {
        demo_messages();
        v.generating = 1;
        v.tok_per_s = 12.4;
        draw_chat(&v);
    } else if (!strcmp(screen, "loading")) {
        draw_loading(0.42f, "Cargando el modelo (2.1 de 4.6 GB)…");
    } else if (!strcmp(screen, "error")) {
        draw_error("No se pudo cargar model.idx: revisa /data/models.");
    } else if (!strcmp(screen, "keyboard")) {
        demo_messages();
        Kb kb;
        kb_init(&kb, KB_ES);
        draw_keyboard(&kb, "Una pregunta sobre la atención multi-cabeza", 1.0);
    } else if (!strcmp(screen, "voice")) {
        float levels[24];
        for (int i = 0; i < 24; i++) levels[i] = 0.2f + 0.8f * (float)((i * 37) % 11) / 11.0f;
        draw_voice(VOICE_RECORDING, levels, 24, 1.8, 30.0, 1.8);
    } else if (!strncmp(screen, "panel", 5)) {   // the control panel, each section (panel, panel1 … panel4)
        static PanelView pv;
        memset(&pv, 0, sizeof(pv));
        pv.tabs[0] = S_P_TAB_API, pv.tabs[1] = S_P_TAB_IMAGES, pv.tabs[2] = S_P_TAB_MODEL,
        pv.tabs[3] = S_P_TAB_SYSTEM, pv.tabs[4] = S_P_ABOUT;
        pv.n_tabs = 5;
        pv.tab = screen[5] ? screen[5] - '0' : 0;
        pv.api_on = 1;
        pv.api_addr = "192.168.1.42:8080";
        pv.sel = 1;
        if (pv.tab == 0) {
            pv.rows[0] = (PanelRow){S_P_ENABLED, "", PANEL_SWITCH, 1};
            pv.rows[1] = (PanelRow){S_P_ADDRESS, "192.168.1.42:8080", PANEL_VALUE, 0};
            pv.rows[2] = (PanelRow){S_P_PORT, "8080", PANEL_STEP, 0};
            pv.rows[3] = (PanelRow){S_P_KEY, "\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2\xe2\x80\xa2",
                                    PANEL_ACTION, 0};
            pv.rows[4] = (PanelRow){S_P_REQUESTS, "37", PANEL_VALUE, 0};
            pv.n_rows = 5;
            pv.status = "POST /v1/chat/completions · /v1/completions · /v1/responses · /v1/images/generations";
            pv.tip[0] = "curl http://192.168.1.42:8080/v1/chat/completions \\";
            pv.tip[1] = "  -H 'Content-Type: application/json' \\";
            pv.tip[2] = "  -d '{\"model\": \"Ornith-1.5-9B\", \"messages\": []}'";
        } else if (pv.tab == 1) {
            pv.rows[0] = (PanelRow){S_P_BACKEND, "http://192.168.1.10:8080/v1", PANEL_VALUE, 0};
            pv.rows[1] = (PanelRow){S_MODEL_LABEL, "sdxl", PANEL_STEP, 0};
            pv.rows[2] = (PanelRow){S_P_KEY, "not set", PANEL_ACTION, 0};
            pv.rows[3] = (PanelRow){S_P_SAVED, "/data/station-ai/images", PANEL_VALUE, 0};
            pv.rows[4] = (PanelRow){S_P_CLEAR, "", PANEL_ACTION, 0};
            pv.n_rows = 5;
            pv.status = "/data/station_images.json";
            pv.tip[0] = "{ \"base_url\": \"http://<image server>:8000/v1\",";
            pv.tip[1] = "  \"api_key\": \"…\", \"model\": \"sdxl-turbo\" }";
            pv.tip[2] = "tools > images > generate  \xe2\x86\x92  saved, shown in chat";
        } else if (pv.tab == 2) {
            pv.rows[0] = (PanelRow){S_P_ENGINE, "Ornith-1.5-9B", PANEL_VALUE, 0};
            pv.rows[1] = (PanelRow){S_TEMPERATURE, "0.70", PANEL_STEP, 0};
            pv.rows[2] = (PanelRow){S_P_THINKING, "", PANEL_SWITCH, 1};
            pv.rows[3] = (PanelRow){S_P_THINK_CAP, "1024", PANEL_STEP, 0};
            pv.rows[4] = (PanelRow){S_P_CONTEXT, "4.2k / 40k", PANEL_VALUE, 0};
            pv.rows[5] = (PanelRow){S_VOICE_MODE, "ready", PANEL_VALUE, 0};
            pv.n_rows = 6;
            pv.status = "L2 / R2 change the value; \xe2\x9c\x95 turns it on and off.";
            pv.tip[0] = "temperature  0 = steady, 1.5 = wild";
            pv.tip[1] = "cap          reasoning tokens one reply may spend";
            pv.tip[2] = "thinking     saved in /data/garlic_think.txt";
        } else if (pv.tab == 3) {
            pv.rows[0] = (PanelRow){S_P_LANGUAGE, "English", PANEL_STEP, 0};
            pv.rows[1] = (PanelRow){S_P_DOWNLOAD, "", PANEL_ACTION, 0};
            pv.rows[2] = (PanelRow){S_P_CLEAR_CHAT, "", PANEL_ACTION, 0};
            pv.n_rows = 3;
            pv.status = "/data/station_lang.txt";
            pv.tip[0] = "/data/garlic_api_port.txt   the port the API listens on";
            pv.tip[1] = "/data/garlic_api_key.txt    the key, if one is set";
            pv.tip[2] = "/data/garlic_think.txt      thinking, and its cap";
        } else {
            pv.rows[0] = (PanelRow){S_P_VERSION, "1.22", PANEL_VALUE, 0};
            pv.rows[1] = (PanelRow){S_P_ENGINE, "Ornith-1.5-9B · q4s", PANEL_VALUE, 0};
            pv.rows[2] = (PanelRow){S_P_KERNELS, "419 052 B", PANEL_VALUE, 0};
            pv.rows[3] = (PanelRow){S_P_MEM, "1 812 MiB", PANEL_VALUE, 0};
            pv.rows[4] = (PanelRow){S_P_REQUESTS, "37", PANEL_VALUE, 0};
            pv.n_rows = 5;
            pv.tip[0] = "Station AI \xc2\xb7 1.22";
            pv.tip[1] = "github.com/mobjakdebug/garlicornith";
            pv.tip[2] = "PS4 payload \xc2\xb7 100% local";
        }
        draw_panel(&pv);
    } else if (!strcmp(screen, "picture")) {
        demo_messages();
        // a stand-in for what an image backend draws: sky, sun, hills
        enum { W = 640, H = 400 };
        static uint8_t rgb[W * H * 3];
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float t = (float)y / H;
                int r = (int)(28 + 150 * t * t), g = (int)(22 + 96 * t), b = (int)(46 + 70 * t);
                float dx = (float)(x - W * 0.72f), dy = (float)(y - H * 0.24f);
                float d = sqrtf(dx * dx + dy * dy);
                if (d < 54) {   // the sun, with a soft edge
                    float k = d < 44 ? 1.0f : (54 - d) / 10.0f;
                    r = (int)(r + (255 - r) * k), g = (int)(g + (232 - g) * k), b = (int)(b + (170 - b) * k);
                }
                float h1 = H * 0.74f + 26 * sinf((float)x / 90.0f), h2 = H * 0.84f + 18 * sinf((float)x / 140.0f + 2);
                if (y > h2) r = 22, g = 30, b = 34;
                else if (y > h1) r = 34, g = 48, b = 46;
                uint8_t* p = rgb + ((size_t)y * W + x) * 3;
                p[0] = (uint8_t)r, p[1] = (uint8_t)g, p[2] = (uint8_t)b;
            }
        Img im = {W, H, rgb};
        ui_image_add(&im, "Un atardecer sobre las colinas, con el sol bajo y el cielo en degradado.", "/data/station-ai/images/img-3.png");
        draw_chat(&v);
    } else if (!strcmp(screen, "top")) {
        demo_messages();
        v.scroll = 100000;   // (clamped to the oldest message)
        draw_chat(&v);
    } else {
        demo_messages();
        draw_chat(&v);
    }
    shot(out);
    return 0;
}
