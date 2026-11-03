# Dory Loop Chat

An experimental local Windows chat for NVIDIA Dory, with a small llama.cpp patch that changes its recurrent loop count **between requests without reloading model weights**. By John.

Select Light, Medium, Intelligent, or Genius for 1, 2, 3, or 4 loops. The names are convenient labels, not quality guarantees. The app streams the model's generated thinking and answers; thinking is optional. It supports alternate input/recurrent/output layer counts at startup, but changing Dory's trained range is experimental and can degrade outputs.

## Status and limits

- A CUDA run with the Q5_K_M model switched 1 → 2 → 4 → 3 → 1 in one process; each greedy result matched a separately loaded baseline on one prompt. This is not comprehensive numerical or quality validation.
- The server invalidates and reprocesses prompt state after a change. This prototype requires one server slot and `--cache-ram 0`. A mode change applies to the next response, not midway through generation.
- This is not an Ollama patch: Ollama 0.32.1 rejected the Dory architecture during our test. The app has no tools, MCP, or user authentication system.
- Thinking can consume the whole output limit. Longer context, concurrency, different quantizations, and alternate layer ranges need more testing. The model and built binaries are not included.

## Setup

1. Install Python with Tkinter, Git, CMake, a C++ toolchain, and, for NVIDIA acceleration, a compatible CUDA toolkit and driver.
2. Download NVIDIA's [Dory GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF). The Q5_K_M file tested here has SHA-256 `96080be49ffd67b916018b7bdef8c484762576fb52780e10ab98ac4978a660e0`. Model weights have their own [OpenMDW 1.1 license](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF); they are not redistributed here.
3. Clone NVIDIA contributor Brian Chmiel's Dory-enabled llama.cpp fork at commit `98a74c97ea09ac72f6ceda1559808fc0a2f447ce`, apply `dory-live-switch.patch`, and build `llama-server`. The underlying [Dory support PR](https://github.com/ggml-org/llama.cpp/pull/30306) is a draft; our patch is on top of that pinned revision, not stock llama.cpp. Preserve the upstream MIT license in `LICENSE.llama.cpp`.

Example in PowerShell, from this repository (build requirements vary by machine):

```powershell
$patch = (Resolve-Path .\dory-live-switch.patch).Path
git clone https://github.com/brianchmielNV/llama.cpp.git runtime
git -C runtime checkout 98a74c97ea09ac72f6ceda1559808fc0a2f447ce
git -C runtime apply --check $patch
git -C runtime apply $patch
cmake -S runtime -B runtime/build -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON
cmake --build runtime/build --config Release --target llama-server
```

Locate the built `llama-server.exe` and downloaded GGUF, then launch:

```powershell
python .\dory-loop-chat.py --server C:\path\to\llama-server.exe --model C:\path\to\dory.gguf --device CUDA0
```

Use `--device cpu` for CPU inference. Use `--layers 21,30,21` to try a different partition; the three positive counts must add up to 72. A partition change requires a restart. The defaults expect `runtime/build/bin/llama-server.exe` and the Q5_K_M GGUF in `models/` next to the script; the `DORY_SERVER` and `DORY_MODEL` environment variables can override those paths.

The local server binds to loopback with a per-run key. The model process is stopped when the window closes. Do not expose the server to a network.

## Attribution

Dory is NVIDIA's model. Base runtime and Dory architecture support are from llama.cpp and the contributors to the linked draft PR. The patch and desktop prototype build on that work; they do not imply endorsement by NVIDIA or llama.cpp.
