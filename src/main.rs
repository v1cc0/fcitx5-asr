mod audio;
mod ipc;
mod model;
mod vad;

use anyhow::Context;
use audio::{AudioCapture, AudioResampler, TARGET_SAMPLE_RATE, load_wav_16k};
use clap::{Parser, Subcommand, ValueEnum};
use ipc::{DEFAULT_PTT_KEYS, IpcClient, PttEvent, PttSocket, format_ptt_hold};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use vad::{EnergyVad, PttGate};

const RECOGNIZING: &str = "识别中...";
const LISTENING: &str = "聆听中...";

const VAD_MIN_ABS: f32 = 0.015;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Default, ValueEnum)]
enum Trigger {
    /// Energy VAD: auto-cut on silence.
    #[default]
    Vad,
    /// Hold the configured PTT key in a focused Fcitx5 window.
    Ptt,
}

#[derive(Subcommand)]
enum Command {
    Ping,
    Commit {
        text: String,
    },
    Preedit {
        text: String,
    },
    Clear,
    /// Download and validate Audio8-ASR-0.1B ONNX files into the Hugging Face cache.
    Download {
        #[arg(long)]
        int8: bool,
    },
    Inspect {
        #[arg(long)]
        int8: bool,
    },
    Probe {
        #[arg(long)]
        int8: bool,
    },
    Prompt {
        #[arg(long)]
        int8: bool,
    },
    Listen {
        #[arg(long)]
        int8: bool,
        #[arg(long, value_enum, default_value_t = Trigger::Vad)]
        trigger: Trigger,
        /// Fcitx5 key name such as Control_R, Super_L, Pause, F8 (repeatable). Implies --trigger ptt.
        #[arg(long = "ptt-key", value_name = "KEY")]
        ptt_keys: Vec<String>,
    },
    /// Transcribe a WAV file (offline, no Fcitx5).
    Transcribe {
        path: String,
        #[arg(long)]
        int8: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    match Args::parse().command {
        Command::Ping => {
            let mut c = IpcClient::connect().await?;
            println!("{}", c.ping().await?);
        }
        Command::Commit { text } => {
            let mut c = IpcClient::connect().await?;
            c.commit(&text).await?;
        }
        Command::Preedit { text } => {
            let mut c = IpcClient::connect().await?;
            c.preedit(&text).await?;
        }
        Command::Clear => {
            let mut c = IpcClient::connect().await?;
            c.clear().await?;
        }
        Command::Download { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            println!("Audio8-ASR-0.1B loaded ({})", model.config.display());
        }
        Command::Inspect { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            for (name, sess) in [
                ("audio", &model.audio),
                ("prefill", &model.prefill),
                ("decode", &model.decode),
            ] {
                println!("[{name}]");
                for io in sess.inputs() {
                    println!("  input {}: {:?}", io.name(), io.dtype());
                }
                for io in sess.outputs() {
                    println!("  output {}: {:?}", io.name(), io.dtype());
                }
            }
        }
        Command::Probe { int8 } => {
            let mut model = model::ArkModel::download_and_load(int8)?;
            let embeds = model.audio_embeddings(&vec![0.0; 16_000])?;
            println!(
                "audio embeddings: {} values ({} tokens)",
                embeds.len(),
                embeds.len() / 512
            );
        }
        Command::Prompt { int8 } => {
            let mut model = model::ArkModel::download_and_load(int8)?;
            let result = model.transcribe(&vec![0.0; 16_000], 16)?;
            println!("silence transcript: {:?}", result.text);
        }
        Command::Listen {
            int8,
            trigger,
            ptt_keys,
        } => listen(int8, trigger, ptt_keys).await?,
        Command::Transcribe { path, int8 } => {
            let samples = load_wav_16k(&path)?;
            let rms =
                (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            println!(
                "wav: {} samples ({:.2}s) rms={rms:.4}",
                samples.len(),
                samples.len() as f32 / TARGET_SAMPLE_RATE as f32
            );
            let mut model = model::ArkModel::download_and_load(int8)?;
            let started = std::time::Instant::now();
            let result = model.transcribe(&samples, model::MAX_NEW_TOKENS)?;
            println!(
                "inference finished in {:.1}s",
                started.elapsed().as_secs_f32()
            );
            println!(
                "recognition result: {}",
                if result.text.is_empty() {
                    "<empty/EOS>"
                } else {
                    &result.text
                }
            );
        }
    }
    Ok(())
}

fn default_lock_path() -> PathBuf {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("fcitx5-asr.lock");
        }
    }
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/fcitx5-asr-{uid}.lock"))
}

fn take_listen_lock_at(path: &Path) -> anyhow::Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.try_lock()
        .with_context(|| format!("fcitx5-asr listen is already running ({})", path.display()))?;
    Ok(file)
}

fn take_listen_lock() -> anyhow::Result<File> {
    take_listen_lock_at(&default_lock_path())
}

async fn connect_retry(give_up_after: Option<std::time::Duration>) -> anyhow::Result<IpcClient> {
    let deadline = give_up_after.map(|d| tokio::time::Instant::now() + d);
    let mut warned = false;
    loop {
        match IpcClient::connect().await {
            Ok(client) => return Ok(client),
            Err(err) => {
                if deadline.is_some_and(|t| tokio::time::Instant::now() >= t) {
                    return Err(err);
                }
                if !warned {
                    eprintln!("waiting for Fcitx5 ASR addon ({err}); will retry");
                    warned = true;
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

async fn listen(int8: bool, trigger: Trigger, ptt_keys: Vec<String>) -> anyhow::Result<()> {
    let trigger = if ptt_keys.is_empty() {
        trigger
    } else {
        Trigger::Ptt
    };
    let _lock = take_listen_lock()?;
    let ptt_sock = if matches!(trigger, Trigger::Ptt) {
        Some(PttSocket::bind()?)
    } else {
        None
    };
    let mut model = model::ArkModel::download_and_load(int8)?;
    let mut client = connect_retry(None).await?;
    let capture = if matches!(trigger, Trigger::Ptt) {
        AudioCapture::paused()?
    } else {
        AudioCapture::default()?
    };
    let mut resampler = AudioResampler::new(capture.sample_rate);
    if matches!(trigger, Trigger::Vad) {
        println!(
            "listening at {} Hz, {} ch (VAD)",
            capture.sample_rate, capture.channels
        );
    }
    loop {
        let result = match trigger {
            Trigger::Vad => listen_vad(&mut model, &mut client, &capture, &mut resampler).await,
            Trigger::Ptt => {
                listen_ptt(
                    &mut model,
                    &mut client,
                    &capture,
                    &mut resampler,
                    ptt_sock.as_ref().expect("ptt socket"),
                    &ptt_keys,
                )
                .await
            }
        };
        match result {
            Ok(()) => break,
            Err(err) => {
                eprintln!("fcitx5 bridge lost ({err}); retrying");
                match connect_retry(Some(std::time::Duration::from_secs(5))).await {
                    Ok(next) => client = next,
                    Err(_) => break,
                }
            }
        }
    }
    Ok(())
}

async fn listen_vad(
    model: &mut model::ArkModel,
    client: &mut IpcClient,
    capture: &AudioCapture,
    resampler: &mut AudioResampler,
) -> anyhow::Result<()> {
    let mut segmenter = EnergyVad::new(VAD_MIN_ABS);
    let mut chunks = 0usize;
    loop {
        let chunk = match tokio::task::block_in_place(|| capture.receiver().recv()) {
            Ok(chunk) => chunk,
            Err(_) => return Ok(()),
        };
        let samples = resampler.process(&chunk);
        chunks += 1;
        if chunks % 100 == 0 {
            let rms =
                (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            println!(
                "audio: {chunks} chunks, rms={rms:.4}, noise={:.4}",
                segmenter.noise_floor()
            );
            match client.ping().await {
                Ok(true) => {}
                Ok(false) | Err(_) => anyhow::bail!("fcitx5 bridge ping failed"),
            }
        }
        for samples in segmenter.push(&samples) {
            commit_transcript(model, client, &samples).await?;
        }
    }
}

async fn listen_ptt(
    model: &mut model::ArkModel,
    client: &mut IpcClient,
    capture: &AudioCapture,
    resampler: &mut AudioResampler,
    ptt: &PttSocket,
    ptt_keys: &[String],
) -> anyhow::Result<()> {
    if !ptt_keys.is_empty() {
        client.set_ptt_keys(ptt_keys).await?;
    }
    let shown = client
        .get_ptt_keys()
        .await
        .ok()
        .filter(|keys| !keys.is_empty())
        .unwrap_or_else(|| {
            if ptt_keys.is_empty() {
                DEFAULT_PTT_KEYS.iter().map(|k| (*k).to_string()).collect()
            } else {
                ptt_keys.to_vec()
            }
        });
    println!(
        "listening at {} Hz, {} ch (PTT: hold {}); Ctrl-C to stop",
        capture.sample_rate,
        capture.channels,
        format_ptt_hold(&shown)
    );
    let mut gate = PttGate::new();
    let mut ticks = 0u32;
    loop {
        ticks += 1;
        if ticks.is_multiple_of(50) {
            match client.ping().await {
                Ok(true) => {}
                Ok(false) | Err(_) => anyhow::bail!("fcitx5 bridge ping failed"),
            }
        }
        for event in ptt.poll() {
            match event {
                PttEvent::Down => {
                    if !gate.is_held() {
                        println!("ptt: recording");
                        let _ = capture.play();
                        let _ = client.preedit(LISTENING).await;
                    }
                    gate.set_held(true);
                }
                PttEvent::Up => {
                    let _ = capture.pause();
                    if let Some(samples) = gate.release() {
                        commit_transcript(model, client, &samples).await?;
                    } else {
                        let _ = client.clear().await;
                        println!("ptt: discarded short utterance");
                    }
                }
            }
        }
        match capture
            .receiver()
            .recv_timeout(std::time::Duration::from_millis(20))
        {
            Ok(chunk) => gate.push(&resampler.process(&chunk)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

async fn commit_transcript(
    model: &mut model::ArkModel,
    client: &mut IpcClient,
    samples: &[f32],
) -> anyhow::Result<()> {
    let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
    println!(
        "segment detected: {:.2}s rms={rms:.4}; running ARK...",
        samples.len() as f32 / TARGET_SAMPLE_RATE as f32
    );
    let _ = client.preedit(RECOGNIZING).await;
    let started = std::time::Instant::now();
    let result =
        match tokio::task::block_in_place(|| model.transcribe(samples, model::MAX_NEW_TOKENS)) {
            Ok(result) => result,
            Err(err) => {
                eprintln!("inference failed: {err}");
                let _ = client.clear().await;
                return Ok(());
            }
        };
    println!(
        "inference finished in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    let text = result.text;
    if text.is_empty() {
        let _ = client.clear().await;
        println!("recognition result: <empty/EOS> (model selected EOS)");
    } else if model::is_filler(&text) {
        let _ = client.clear().await;
        println!("recognition result: {text} (skipped filler)");
    } else {
        println!("recognition result: {text}");
        client.commit(&text).await?;
        println!("committed to Fcitx5");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn ptt_key_flag_is_repeatable() {
        let args = Args::parse_from([
            "fcitx5-asr",
            "listen",
            "--ptt-key",
            "Super_L",
            "--ptt-key",
            "F8",
        ]);
        match args.command {
            Command::Listen {
                ptt_keys, trigger, ..
            } => {
                assert_eq!(ptt_keys, ["Super_L", "F8"]);
                assert!(matches!(trigger, Trigger::Vad));
            }
            _ => panic!("expected listen"),
        }
    }

    #[test]
    fn listen_lock_is_exclusive() {
        let path =
            std::env::temp_dir().join(format!("fcitx5-asr-lock-test-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let held = take_listen_lock_at(&path).unwrap();
        assert!(take_listen_lock_at(&path).is_err());
        drop(held);
        take_listen_lock_at(&path).unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
