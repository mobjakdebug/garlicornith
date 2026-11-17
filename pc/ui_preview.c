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
