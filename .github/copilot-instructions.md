## Copilot Instructions: llama-cpp-python

Purpose: Python bindings + high-level / server wrappers around vendored `llama.cpp` (under `vendor/llama.cpp`). Performance, minimal deps, alignment with upstream C API.

### Big Picture
- Core C/C++ library is built via `scikit-build-core` + CMake (`CMakeLists.txt` installs shared libs into `llama_cpp/lib`). We expose:
  - Low-level API: thin `ctypes` mirror in `llama_cpp/llama_cpp.py` (auto-synced w/ upstream `llama.h`).
  - High-level API: `llama_cpp/llama.py` (`Llama` class wraps model load, chat formats, speculative decoding, embeddings, caching).
  - Server layer: `llama_cpp/server/` FastAPI-based OpenAI-compatible endpoints (`app.py`, `model.py`, `settings.py`). Chat formats & multimodal handlers in `llama_cpp/llama_chat_format.py`.
- Vendored upstream sources live in `vendor/llama.cpp`; modify only when syncing upstream; Python layer adapts interface and avoids adding heavy deps.

### Architecture Essentials
- Model load path: `Llama.__init__` builds `model_params` + `context_params` (mirrors C structs). GPU/offload flags: `n_gpu_layers`, `offload_kqv`, `tensor_split`, etc.
- Chat formatting resolution order (see README & code): `chat_handler` > explicit `chat_format` > `tokenizer.chat_template` metadata > fallback `llama-2`.
- Multimodal: Handlers subclass `LlamaChatCompletionHandler` (e.g. `Llava15ChatHandler`) managing image preprocessing + prompt assembly.
- Speculative decoding: Provide `draft_model` (e.g. `LlamaPromptLookupDecoding`) to `Llama` ctor; accelerates generation without changing external API.
- Caching: `llama_cache.py` offers in-RAM / disk caches for KV reuse; ensure compatibility when adjusting context size or quantization.
- Server model routing: Config file (`--config_file`) maps `model_alias` to concrete GGUF paths; only one model resident at a time (lazy load + unload).

### Key Workflows
- Editable dev install (preferred for Python changes): `pip install -e .` (Make target `make build`). Use env var `CMAKE_ARGS` for backend flags (e.g. `-DGGML_METAL=on`).
- Debug build: `make build.debug` (adds `-ggdb -O0`); for ASAN use `make build.debug.extra`.
- Backend builds (examples): `make build.cuda`, `make build.openblas`, etc. Each sets `CMAKE_ARGS` appropriately.
- Tests: `make test` or `pytest` (paths under `tests/`). Ensure new high-level behaviors reflected with minimal deterministic tests (avoid network/model downloads; use small local GGUF fixtures if adding).
- Server run (single model): `python -m llama_cpp.server --model <path.gguf> [--chat_format chatml]`.

### Conventions & Patterns
- Keep new high-level features inside `llama_cpp/` with minimal external imports (only stdlib + listed dependencies in `pyproject.toml`). Avoid adding runtime deps beyond optional extras.
- Mirror upstream C changes: when `vendor/llama.cpp/include/llama.h` changes, sync signatures/types in `llama_cpp/llama_cpp.py` and adjust wrapper logic in `llama.py` (search for changed enum names/constants first).
- Public surface: Do not silently change return schema of high-level completion (`__call__`, `create_completion`, `create_chat_completion`); preserve OpenAI-compatible dict shape.
- Embeddings: Require `embedding=True` at ctor; tests or examples calling `create_embedding` must set it explicitly.
- Chat formats registration via decorators `@register_chat_format` in `llama_chat_format.py`; follow existing pattern for new formats (provide deterministic `apply_chat_template`).
- Multimodal: Increase `n_ctx` to accommodate image embeddings; ensure handler sets appropriate special tokens; avoid breaking pure text models.
- Performance flags: Keep defaults lean; any experimental knob (e.g. rope scaling, flash attn) must pass through untouched if unset (0 / UNSPECIFIED).

### Safe Change Guidance
- Before modifying model loading or context params, scan for uses of those fields in `llama.py` generation paths (search `self.model_params` / `self.context_params`). Changing defaults can regress memory use.
- When adding server options, extend `ModelSettings` / `ServerSettings` in `llama_cpp/server/settings.py` and ensure env var + CLI parity.
- Updating vendor code: perform submodule update (`git submodule update --init --recursive`), then `make clean` and rebuild; validate by importing and calling a trivial `Llama(model_path=..., vocab_only=True)`.

### Examples References
- High-level usage: README section “High-level API” shows completion + chat examples.
- Multimodal reference: `examples/multimodal_qwen3vl_example.py` and handlers in `llama_chat_format.py`.
- Speculative decoding: `llama_cpp/llama_speculative.py` and README “Speculative Decoding”.

### Typical Validation Sequence After Changes
1. `make build` (or specific backend). 2. `pytest -q`. 3. Smoke load: `python - <<'PY'\nfrom llama_cpp import Llama; print(Llama(model_path='path/to/model.gguf', vocab_only=True))\nPY`. 4. Optional server start & single chat request.

### What NOT to Do
- Don’t add heavyweight ML frameworks (Torch, TF) to core; leverage ggml only.
- Don’t change OpenAI response keys (`choices`, `usage`, `finish_reason`) without version gating.
- Avoid network-dependent tests; keep CI deterministic.

### If Unsure
Prefer inspecting upstream `vendor/llama.cpp` for authoritative behavior, then replicate minimally in Python wrappers.

Provide targeted edits; ask if a change impacts public API or multi-backend compatibility.
