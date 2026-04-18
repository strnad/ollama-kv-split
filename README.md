<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama KV Split

A small fork of [`ollama/ollama`](https://github.com/ollama/ollama) v0.21.0 that
adds two API extensions around the KV cache:

1. **Per-request `kv_cache_type`** — override the KV cache type per API call
   instead of only through the `OLLAMA_KV_CACHE_TYPE` environment variable.
   Useful for trying different cache precisions without restarting the server
   or reloading through an env change.

2. **Split `k_cache_type` / `v_cache_type`** — set K and V cache precision
   independently. K is more sensitive to quantization than V, so a recipe like
   `k_cache_type=q8_0, v_cache_type=q4_0` keeps attention quality high while
   roughly halving V memory. This mirrors `llama.cpp`'s `--cache-type-k` /
   `--cache-type-v`, exposed through the Ollama HTTP API.

Everything upstream continues to work unchanged; nothing is removed. When the
new fields are unset the server behaves exactly like upstream v0.21.0.

---

## Install

### One-line installer (Linux)

```shell
curl -fsSL https://github.com/strnad/ollama-kv-split/releases/latest/download/install.sh | sh
```

The installer detects `nvidia-smi` and pulls the CUDA tarball when present,
falling back to the CPU tarball otherwise. It drops a binary at
`/usr/local/bin/ollama` and registers a `systemd` unit named `ollama.service`
(the upstream Ollama unit is stopped first to avoid port conflicts).

### Docker

```shell
# CUDA (sm_86 / RTX 30xx, Ampere)
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama \
  ghcr.io/strnad/ollama-kv-split:cuda-latest

# CPU only
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama \
  ghcr.io/strnad/ollama-kv-split:cpu-latest
```

### Manual tarball

Download from
[releases](https://github.com/strnad/ollama-kv-split/releases) and extract:

```shell
tar -C /usr -xzf ollama-linux-amd64.tgz           # CPU
tar -C /usr -xzf ollama-linux-amd64-cuda.tgz      # with CUDA runtime
```

---

## Usage

### Per-request KV cache override (symmetric K=V)

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "options": {
    "kv_cache_type": "q8_0"
  }
}'
```

### Split K/V (asymmetric, llama.cpp parity)

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "options": {
    "k_cache_type": "q8_0",
    "v_cache_type": "q4_0"
  }
}'
```

Requests that change the cache type for a loaded model trigger a clean reload
through the scheduler's existing `needsReload()` path — no state carries over.

### Supported cache types

| Value    | Bytes/elem | Flash attention required |
|----------|-----------:|:-------------------------|
| `f16`    | 2.0        | No                       |
| `q8_0`   | 1.0        | Yes                      |
| `q4_0`   | 0.5        | Yes                      |
| `q5_0`   | 0.625      | Yes                      |
| `q5_1`   | 0.75       | Yes                      |
| `iq4_nl` | 0.5        | Yes                      |

Quantized cache types require `OLLAMA_FLASH_ATTENTION=1` (or the model's
default FA path on the new engine). Runtime feasibility on a specific model
is further constrained by the model's head layout — the server logs a warning
and falls back to `f16` on an unsupported combination.

### Environment variables

Three env vars provide server-wide defaults for operators who don't want to
touch every API call:

- `OLLAMA_KV_CACHE_TYPE` — symmetric K=V (original upstream var, still works)
- `OLLAMA_K_CACHE_TYPE` — K only (split K/V)
- `OLLAMA_V_CACHE_TYPE` — V only (split K/V)

Per-request `kv_cache_type` / `k_cache_type` / `v_cache_type` in
`api.Options` win over env vars.

### Resolution order

The server resolves the final (K, V) pair from four sources, highest priority
first:

1. `k_cache_type` / `v_cache_type` in the request body (split K/V)
2. `kv_cache_type` in the request body (symmetric shortcut)
3. `OLLAMA_K_CACHE_TYPE` / `OLLAMA_V_CACHE_TYPE` env vars
4. `OLLAMA_KV_CACHE_TYPE` env var (symmetric fallback)

If only one of K/V is specified at a given level, the other side inherits it
(symmetric behavior). Both sides empty means `f16` on both.

---

## Benchmarks

Tail-token throughput on a single RTX 3090 Ti (24 GB), measured
2026-04-18, Ollama `v0.21.0-split.1`, gemma4-like 30B MoE at 32K context.
Numbers are rough — they illustrate the memory vs throughput trade-off, not an
absolute claim against any other runtime.

| ctx  | K cache | V cache | KV MiB | tok/s |
|-----:|:--------|:--------|-------:|------:|
| 32K  | f16     | f16     |  ~1320 |  ~19  |
| 32K  | q8_0    | q8_0    |   ~660 |  ~19  |
| 32K  | q8_0    | q4_0    |   ~495 |  ~38  |
| 32K  | q4_0    | q4_0    |   ~330 |  ~38  |

K at q8 keeps attention quality close to f16; dropping V to q4 recovers memory
for bigger batches / longer context without the quality loss that K=q4 would
bring.

---

## Build from source

Standard upstream Ollama build — CMake for the `ggml` backends, Go for the
front-end. On a GPU host you'll want `CUDA_ARCHITECTURES` pinned to just the
arch you're targeting (the full multi-arch build is slow and produces a huge
binary).

### CUDA (sm_86, RTX 30xx / Ampere)

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel $(nproc)
go build -o ollama .
```

### CPU only

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --parallel $(nproc)
go build -o ollama .
```

### Docker

`Dockerfile.slim` is a single-arch CUDA build (sm_86 by default; override via
`--build-arg CUDA_ARCHITECTURES=...`). `Dockerfile.cpu` is a CPU-only build.

```shell
# CUDA
docker build -f Dockerfile.slim -t ollama-kv-split:cuda .

# CPU
docker build -f Dockerfile.cpu -t ollama-kv-split:cpu .
```

---

## Compatibility

- Linux x86_64 (Ubuntu 22.04+ tested)
- NVIDIA GPUs, compute capability 6.0 (Pascal) and newer; Flash Attention
  requires 7.5 (Turing) or newer for most cache types
- CUDA 12.x runtime drivers
- macOS / Windows: not tested in this fork; upstream `ollama/ollama` still
  covers those targets if you don't need split K/V

---

## License

MIT, same as upstream Ollama. See [`LICENSE`](LICENSE).

---

# Upstream Ollama README

Start building with open models.

## Download

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

or [download manually](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

or [download manually](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Community

- [Discord](https://discord.gg/ollama)
- [X (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Get started

```
ollama
```

You'll be prompted to run a model or connect Ollama to your existing agents or
applications such as `Claude Code`, `OpenClaw`, `OpenCode`, `Codex`, `Copilot`
and more. Run `ollama help` for a list of available commands.
