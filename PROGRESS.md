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
- [ ] Confirm architecture direction with user (Direction A: Native Addon Bridge vs Direction B: Pure Rust Virtual Input).
- [x] Set up Rust project workspace and crate structure (`Cargo.toml`).
- [x] Implement initial audio capture & resampling pipeline (`cpal`; minimal stdlib resampler).
- [ ] Implement ONNX runtime inference engine & tokenizer decoding (`ort` + `tokenizers`).
- [ ] Implement IPC bridge / Fcitx5 integration.
- [ ] Implement push-to-talk and VAD trigger mechanism.

Next: confirm Direction A/B with the user, then wire capture chunks into VAD/ASR rather than adding speculative model code.
