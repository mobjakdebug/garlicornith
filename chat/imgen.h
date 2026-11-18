// Station AI: image models through the API (imgen.c). A console this size cannot run a diffusion model, so
// POST /v1/images/generations is answered by an image backend — a PC on the LAN or a service — that this app
// forwards to: the request goes out as it came, the answer (the same OpenAI shape) comes back to the client
// unchanged, and the picture itself is saved under /data/station-ai/images and shown on the console.
//
// The backend is named in /data/station_images.json:
//   {"base_url": "http://192.168.1.10:8080/v1", "api_key": "", "model": "sdxl"}
// (a full endpoint in "url" instead of "base_url" also works). Without that file, an image request is refused with
// a message that says so.
#pragma once
#include <stddef.h>

// Answers one POST /v1/images/generations on fd (the whole request body in body). Runs on the main thread
// (api_run.inc calls it), so the wait and the picture can be drawn: it writes the reply itself.
void imgen_run(int fd, const char* body, size_t len);
