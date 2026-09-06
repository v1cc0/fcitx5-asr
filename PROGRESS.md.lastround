# Project Progress: fcitx5-asr

## 1. Architectural Plans & Technical Analysis
- **Goal**: Build a high-performance, crash-isolated Linux speech recognition (ASR) input method for Fcitx5 in Rust using local ONNX models.
- **Key Discovery & Reality Check (Measure Twice, Cut Once)**:
  - Fcitx5 D-Bus interface (`org.fcitx.Fcitx.InputContext1`) defines `CommitString` and `ForwardKey` as **D-Bus signals** (outbound events to client windows), not inbound callable methods. Arbitrary external processes cannot directly invoke `CommitString` on the focused window over raw D-Bus.
  - Two distinct technical directions exist:
    1. **Direction A (Recommended: Native Fcitx5 Shim Addon + Rust Core Daemon)**: A minimal C++ addon (~80 LOC) inside Fcitx5 listening on a local Unix socket (`/run/user/$UID/fcitx5-asr.sock`), delivering text via `currentIC->commitString(text)` and supporting real-time Preedit UI.
    2. **Direction B (Pure Rust Standalone Daemon + Virtual Input Protocol)**: No Fcitx5 addon; text injected via Wayland `zwp_virtual_keyboard_v1` / X11 `xtest` or clipboard paste.
- **Target Pipeline**:
  - `Audio`: `cpal` microphone capture -> ring buffer -> `rubato` resampler (16kHz f32 mono).
  - `VAD`: Silero-VAD ONNX or energy VAD for speech boundary detection.
  - `ASR Inference`: `ort` (ONNX Runtime) + `tokenizers` decoding.
  - `IPC / Commit`: Unix domain socket client to Fcitx5 shim (or virtual keyboard).

## 2. Direction Shifts
- Clarified that Fcitx5 D-Bus does not allow raw `CommitString` calls without a companion addon or frontend protocol bridge.

## 3. Completions
- [x] Workspace initialized with Rust 2024 edition toolchain.
- [x] Initialized CodeGraph index (`codegraph init -i`).
- [x] Initialized `INCIDENTS.md` and `PROGRESS.md`.
- [x] Verified Fcitx5 D-Bus API surface and header files on system.
- [x] Added cpal microphone capture with mono f32 chunk delivery.
- [x] Added stateful linear resampling to the 16 kHz ASR target rate.
- [x] Built the native Fcitx5 addon and Rust crate successfully; resampler self-check passes.

## 4. Next Steps (To-Dos)
- [x] Confirmed Direction A: Native Addon Bridge + Rust core daemon.
- [x] Set up Rust project workspace and crate structure (`Cargo.toml`).
- [x] Implement initial audio capture & resampling pipeline (`cpal`; minimal stdlib resampler; unused `rubato` removed).
- [ ] Implement ONNX runtime inference engine & tokenizer decoding (`ort` + `tokenizers`).
- [x] Implement IPC bridge / Fcitx5 integration (Unix socket addon plus Rust CLI commands).
- [x] Fixed IPC EOF race so short-lived CLI commands are processed reliably.
- [x] Prefer Fcitx5's last focused input context for external commit requests.
- [x] Added `OK` acknowledgements so CLI waits for addon command processing.
- [x] Added pure-Rust ARK-ASR Hub downloader and `ort` session loader (`download` command).
- [x] Added `listen` microphone mode with resampling and RMS speech segmentation.
- [x] Added Rust Whisper-128 log-mel feature extraction with rustfft.
- [x] Added ARK encoder invocation with `[1,128,frames]` tensors via `ort`.
- [x] Added `inspect` command to enumerate ARK decoder KV-cache contract (24 layers, 896 hidden, 163958 vocab).
- [x] Parsed ARK prompt/EOS/suppression config and added byte-level vocabulary decoding helpers.
- [x] Added ARK `embed_tokens` invocation and typed empty 24-layer KV cache allocation.
- [x] Implemented and validated ARK decoder prompt pass with zero KV cache (1,639,580 logits for 10 tokens).
- [x] Added config-aware greedy token selection (suppression list applied).
- [x] Added ARK audio/prompt embedding composition for decoder input.
- [x] Added last-position logits extraction and suppression-aware next-token probe (silent prompt returns EOS).
- [x] Extracted all 48 ARK `present.*` tensors after prompt pass for incremental KV-cache decoding.
- [x] Added single-token decoder step API that feeds prior KV cache and refreshes all 48 tensors.
- [x] Connected `listen --int8` end-to-end: microphone segments → ARK transcription → Fcitx5 commit.
- [x] Materialized Hugging Face ONNX external-data files into a shared directory so `listen` can load ARK safely.
- [x] Added periodic microphone RMS diagnostics to verify capture and tune speech gate.
- [x] Fixed Whisper feature padding to ARK's fixed 30-second/3000-frame encoder window; int8 encoder probe succeeds (336000 values).
- [ ] Implement push-to-talk and VAD trigger mechanism.

Next: implement ARK-ASR Whisper-128 log-mel preprocessing and decoder KV-cache loop, then replace segment logging with bridge commits.
