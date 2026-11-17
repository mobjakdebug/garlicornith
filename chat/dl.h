// Station AI: the model downloaded on the console itself (dl.c, the "insane option": no PC and no FTP).
// The screen lists a Hugging Face repository's files (the API's tree endpoint), downloads the ones this engine
// reads — model.idx and the weights .bin — into /data/models/<repo>/, resuming what a cut connection left, and
// then points the app at that directory.
#pragma once

// Runs the screen end to end: asks for the repository (the console's keyboard), connects, downloads, verifies the
// sizes. Returns 1 when a model is ready in the directory below, 0 when the player cancelled or it failed (the
// reason is in the log, and on the screen for a few seconds).
int dl_run(void);

// A JSON POST to an OpenAI-compatible server (the image backend, imgen.c): the whole answer in a buffer the caller
// frees, *out_len its length and *status the HTTP code (-1 when the request itself failed). NULL on failure.
char* dl_http_post(const char* url, const char* auth, const char* body, size_t blen, size_t* out_len, int* status);

// The directory of the last model downloaded here (/data/station_model.txt), or NULL if there is none. The caller
// still checks that its model.idx is there.
const char* dl_model_dir(void);
