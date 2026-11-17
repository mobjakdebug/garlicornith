// Station AI: the control panel (drawn by ui.c's draw_panel, included in main.c). Everything the app can be told to
// do is a row here — the API and its port and key, the image backend, the model's sampling and thinking, the
// language, the on-console downloader — and every change is kept in /data, so it comes back at the next start.
//
// It opens with OPTIONS on the chat screen and with the "Panel" hint on the empty one. Up/Down move the cursor,
// R1/L1 change section, Cross opens or flips what is under it, L2/R2 change a number or a choice, Circle returns.

#define PANEL_LANG_FILE "/data/station_lang.txt"
#define PANEL_IMAGES_FILE "/data/station_images.json"
#define PANEL_KEY_FILE "/data/garlic_api_key.txt"
#define PANEL_THINK_FILE "/data/garlic_think.txt"

typedef struct {
    char base[400];
    char key[300];
    char model[120];
} PanelImg;

static void panel_write(const char* path, const char* text, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        say("panel: cannot write %s", path);
        return;
    }
    fwrite(text, 1, n, f);
    fclose(f);
}

static void panel_row(PanelView* pv, StrId label, const char* value, int kind, int on) {
    if (pv->n_rows >= PANEL_ROWS) return;
    PanelRow* r = &pv->rows[pv->n_rows++];
    memset(r, 0, sizeof(*r));
    r->label = label, r->kind = kind, r->on = on;
    snprintf(r->value, sizeof(r->value), "%s", value ? value : "");
}

// The image backend as /data/station_images.json has it (empty when there is none).
static void panel_img_read(PanelImg* c) {
    memset(c, 0, sizeof(*c));
    size_t n = 0;
    char* t = (char*)read_all(PANEL_IMAGES_FILE, &n);
    if (t && n) {
        JNode* pool = malloc(sizeof(JNode) * 64);
        int root = pool ? json_parse(t, n, pool, 64) : -1;
        if (root >= 0 && pool[root].type == J_OBJ) {
            static const char* keys[] = {"base_url", "url", "api_key", "model"};
            char* outs[] = {c->base, c->base, c->key, c->model};
            size_t caps[] = {sizeof(c->base), sizeof(c->base), sizeof(c->key), sizeof(c->model)};
            for (int i = 0; i < 4; i++) {
                int v = json_get(pool, root, keys[i]);
                if (v < 0 || pool[v].type != J_STR || outs[i][0]) continue;   // (base_url first, url if not)
                Sb s;
                sb_init(&s);
                json_str_sb(pool, v, &s);
                if (s.p) snprintf(outs[i], caps[i], "%s", s.p);
                sb_free(&s);
            }
        }
        free(pool);
    }
    free(t);
}

static void panel_img_write(const PanelImg* c) {
    if (!c->base[0]) {
        remove(PANEL_IMAGES_FILE);
        say("panel: image backend removed");
        return;
    }
    Sb s;
    sb_init(&s);
    sb_str(&s, "{\"base_url\":");
    sb_json_str(&s, c->base, strlen(c->base));
    sb_str(&s, ",\"api_key\":");
    sb_json_str(&s, c->key, strlen(c->key));
    sb_str(&s, ",\"model\":");
    sb_json_str(&s, c->model, strlen(c->model));
    sb_str(&s, "}\n");
    if (s.p) panel_write(PANEL_IMAGES_FILE, s.p, s.n);
    say("panel: image backend %s", c->base);
    sb_free(&s);
}

// A string the player types, with the app's own keyboard over the panel. 1 when it changed.
static int panel_edit(char* out, int max) {
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s", out);
    if (!keyboard_ex(buf, max, 1, 1)) return 0;
    if (!strcmp(buf, out)) return 0;
    snprintf(out, (size_t)max, "%s", buf);
    return 1;
}

static const char* panel_lang_name(int l) {
    static const char* names[N_LANGS] = {"Español", "English", "Português", "Русский"};
    return l >= 0 && l < N_LANGS ? names[l] : names[LANG_EN];
}

// Whether /data has an API key in it (the key itself is never shown).
static int panel_have_key(void) {
    size_t n = 0;
    uint8_t* kd = read_all(PANEL_KEY_FILE, &n);
    int have = 0;
    for (size_t i = 0; kd && i < n; i++)
        if (kd[i] != '\n' && kd[i] != '\r' && kd[i] != ' ') have = 1;
    free(kd);
    return have;
}

// The rows of one section, and the line under them.
static void panel_build(PanelView* pv, int tab, int sel) {
    memset(pv, 0, sizeof(*pv));
    pv->tabs[0] = S_P_TAB_API, pv->tabs[1] = S_P_TAB_IMAGES, pv->tabs[2] = S_P_TAB_MODEL,
    pv->tabs[3] = S_P_TAB_SYSTEM, pv->tabs[4] = S_P_ABOUT;
    pv->n_tabs = PANEL_TABS;
    pv->tab = tab;
    pv->api_on = api_enabled();
    pv->api_addr = api_address();
    char v[128];
    if (tab == 0) {
        panel_row(pv, S_P_ENABLED, "", PANEL_SWITCH, api_enabled());
        panel_row(pv, S_P_ADDRESS, api_address(), PANEL_VALUE, 0);
        snprintf(v, sizeof(v), "%d", api_port());
        panel_row(pv, S_P_PORT, v, PANEL_STEP, 0);
        panel_row(pv, S_P_KEY, panel_have_key() ? "••••••" : tr(S_P_NO_KEY), PANEL_ACTION, 0);
        snprintf(v, sizeof(v), "%u", api_requests());
        panel_row(pv, S_P_REQUESTS, v, PANEL_VALUE, 0);
        pv->status = "POST /v1/chat/completions · /v1/completions · /v1/responses · /v1/images/generations";
        static char curl[160];
        snprintf(curl, sizeof(curl), "curl http://%s/v1/chat/completions \\", api_address());
        pv->tip[0] = curl;
        pv->tip[1] = "  -H 'Content-Type: application/json' \\";
        pv->tip[2] = "  -d '{\"model\": \"" MODEL_NAME "\", \"messages\": []}'";
    } else if (tab == 1) {
        static PanelImg img;
        panel_img_read(&img);
        panel_row(pv, S_P_BACKEND, img.base[0] ? img.base : tr(S_P_NOTHING), PANEL_ACTION, 0);
        panel_row(pv, S_MODEL_LABEL, img.model[0] ? img.model : "—", PANEL_ACTION, 0);
        panel_row(pv, S_P_KEY, img.key[0] ? "••••••" : tr(S_P_NO_KEY), PANEL_ACTION, 0);
        panel_row(pv, S_P_SAVED, "/data/station-ai/images", PANEL_VALUE, 0);
        panel_row(pv, S_P_CLEAR, "", PANEL_ACTION, 0);
        pv->status = PANEL_IMAGES_FILE;
        pv->tip[0] = "{ \"base_url\": \"http://<image server>:8000/v1\",";
        pv->tip[1] = "  \"api_key\": \"…\", \"model\": \"sdxl-turbo\" }";
        pv->tip[2] = "tools > images > generate  →  saved, shown in chat";
    } else if (tab == 2) {
        panel_row(pv, S_P_ENGINE, MODEL_NAME, PANEL_VALUE, 0);
        snprintf(v, sizeof(v), "%.2f", view.temperature);
        panel_row(pv, S_TEMPERATURE, v, PANEL_STEP, 0);
        panel_row(pv, S_P_THINKING, "", PANEL_SWITCH, think_on);
        snprintf(v, sizeof(v), "%d", think_cap);
        panel_row(pv, S_P_THINK_CAP, v, PANEL_STEP, 0);
        snprintf(v, sizeof(v), "%d / %d", pos, (int)eng.ctx_limit);
        panel_row(pv, S_P_CONTEXT, v, PANEL_VALUE, 0);
        panel_row(pv, S_VOICE_MODE, tts_ready ? tr(S_P_ON) : tr(S_P_OFF), PANEL_VALUE, 0);
        pv->status = tr(S_P_CHANGE);
        pv->tip[0] = "temperature  0 = steady, 1.5 = wild";
        pv->tip[1] = "cap          reasoning tokens one reply may spend";
        pv->tip[2] = "thinking     saved in /data/garlic_think.txt";
    } else if (tab == 3) {
        panel_row(pv, S_P_LANGUAGE, panel_lang_name(ui_lang), PANEL_STEP, 0);
        panel_row(pv, S_P_DOWNLOAD, "", PANEL_ACTION, 0);
        panel_row(pv, S_P_CLEAR_CHAT, "", PANEL_ACTION, 0);
        pv->status = PANEL_LANG_FILE;
        pv->tip[0] = "/data/garlic_api_port.txt   the port the API listens on";
        pv->tip[1] = "/data/garlic_api_key.txt    the key, if one is set";
        pv->tip[2] = "/data/garlic_think.txt      thinking, and its cap";
    } else {
        panel_row(pv, S_P_VERSION, "1.22", PANEL_VALUE, 0);
        snprintf(v, sizeof(v), "%s · %d tokens", MODEL_NAME, (int)eng.ctx_limit);
        panel_row(pv, S_P_ENGINE, v, PANEL_VALUE, 0);
        size_t flex = 0;
        ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
        snprintf(v, sizeof(v), "%u MiB", (unsigned)(flex >> 20));
        panel_row(pv, S_P_MEM, v, PANEL_VALUE, 0);
        snprintf(v, sizeof(v), "%u", api_requests());
        panel_row(pv, S_P_REQUESTS, v, PANEL_VALUE, 0);
        pv->tip[0] = "Station AI \xc2\xb7 1.22";
        pv->tip[1] = "github.com/mobjakdebug/garlicornith";
        pv->tip[2] = "PS4 payload \xc2\xb7 100% local";
    }
    pv->n_rows = pv->n_rows ? pv->n_rows : 1;   // (never zero: the cursor has somewhere to be)
    pv->sel = sel >= pv->n_rows ? pv->n_rows - 1 : sel < 0 ? 0 : sel;
}

// The API's key, typed: an empty one removes it, and the server reads it again straight away.
static void panel_api_key(void) {
    static char buf[200];
    snprintf(buf, sizeof(buf), "%s", "");   // (the key is not shown while it is typed)
    if (!keyboard_ex(buf, (int)sizeof(buf), 1, 1)) return;
    if (buf[0]) {
        char line[220];
        int k = snprintf(line, sizeof(line), "%s\n", buf);
        panel_write(PANEL_KEY_FILE, line, (size_t)k);
        say("panel: API key set (%d bytes)", k - 1);
    } else {
        remove(PANEL_KEY_FILE);
        say("panel: API key removed");
    }
    api_reload_key();
}

// What Cross, R2 (dir 1) and L2 (dir -1) do on the row under the cursor.
static void panel_act(int tab, int sel, int dir) {
    if (tab == 0) {
        if (sel == 0 && dir == 0) {
            int want = !api_enabled();
            if (api_enable(want)) toast(S_API_FAILED);
            else toast(want ? S_API_ON : S_API_OFF);
        } else if (sel == 2 && dir) {
            api_set_port(api_port() + dir);
        } else if (sel == 3 && dir == 0) {
            panel_api_key();
        }
    } else if (tab == 1 && dir == 0) {
        static PanelImg img;
        panel_img_read(&img);
        if (sel == 0) {
            if (panel_edit(img.base, sizeof(img.base))) panel_img_write(&img);
        } else if (sel == 1) {
            if (panel_edit(img.model, sizeof(img.model))) panel_img_write(&img);
        } else if (sel == 2) {
            if (panel_edit(img.key, sizeof(img.key))) panel_img_write(&img);
        } else if (sel == 4) {
            remove(PANEL_IMAGES_FILE);
            say("panel: image backend removed");
        }
    } else if (tab == 2) {
        if (sel == 1 && dir) {
            float t = view.temperature + 0.1f * (float)dir;
            view.temperature = t < 0 ? 0 : t > 1.5f ? 1.5f : t;
        } else if (sel == 2 && dir == 0) {
            think_toggle();
        } else if (sel == 3 && dir) {
            int c = think_cap + 256 * dir;
            if (c < 64) c = 64;
            if (c > 4096) c = 4096;
            think_cap = c;
            char line[16];
            int k = snprintf(line, sizeof(line), "%d\n", c);
            panel_write(PANEL_THINK_FILE, line, (size_t)k);
            say("panel: reasoning cap %d tokens", c);
        }
    } else if (tab == 3) {
        if (sel == 0 && dir) {
            ui_lang = (ui_lang + (dir > 0 ? 1 : N_LANGS - 1)) % N_LANGS;
            char line[8];
            int k = snprintf(line, sizeof(line), "%d\n", ui_lang);
            panel_write(PANEL_LANG_FILE, line, (size_t)k);
            say("panel: language %d", ui_lang);
        } else if (sel == 1 && dir == 0) {
            dl_run();   // (its own screen and its own waits; the panel is drawn again after)
        } else if (sel == 2 && dir == 0) {
            msgs_clear();
            pos = 0, need_close = 0, view.scroll = 0;
            toast(S_NEW_CHAT);
        }
    }
}

static void panel_mode(void) {
    int tab = 0, sel = 0, rows = 1;
    uint32_t prev = read_buttons();
    for (;;) {
        PanelView pv;
        panel_build(&pv, tab, sel);
        rows = pv.n_rows;
        draw_panel(&pv);
        present();
        uint32_t b = read_buttons(), pressed = b & ~prev;
        prev = b;
        if (pressed & ORBIS_PAD_BUTTON_CIRCLE) return;
        if (pressed & (ORBIS_PAD_BUTTON_DOWN | ORBIS_PAD_BUTTON_UP)) {
            sel += (pressed & ORBIS_PAD_BUTTON_DOWN) ? 1 : -1;
            if (sel < 0) sel = rows - 1;
            if (sel >= rows) sel = 0;
        }
        if (pressed & (ORBIS_PAD_BUTTON_R1 | ORBIS_PAD_BUTTON_L1)) {
            tab = (tab + ((pressed & ORBIS_PAD_BUTTON_R1) ? 1 : PANEL_TABS - 1)) % PANEL_TABS;
            sel = 0;
        }
        if (pressed & ORBIS_PAD_BUTTON_CROSS) panel_act(tab, sel, 0);
        if (pressed & ORBIS_PAD_BUTTON_R2) panel_act(tab, sel, 1);
        if (pressed & ORBIS_PAD_BUTTON_L2) panel_act(tab, sel, -1);
        if (!(b & (ORBIS_PAD_BUTTON_CROSS | ORBIS_PAD_BUTTON_R2 | ORBIS_PAD_BUTTON_L2))) sceKernelUsleep(8000);
    }
}
