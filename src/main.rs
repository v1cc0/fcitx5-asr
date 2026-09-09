mod audio;
mod ipc;
mod model;
mod vad;

use audio::{AudioCapture, AudioResampler, TARGET_SAMPLE_RATE, load_wav_16k};
use clap::{Parser, Subcommand, ValueEnum};
use ipc::{DEFAULT_PTT_KEYS, IpcClient, PttEvent, PttSocket, format_ptt_hold};
use vad::{EnergyVad, PttGate};

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

async fn listen(int8: bool, trigger: Trigger, ptt_keys: Vec<String>) -> anyhow::Result<()> {
    let trigger = if ptt_keys.is_empty() {
        trigger
    } else {
        Trigger::Ptt
    };
    let mut model = model::ArkModel::download_and_load(int8)?;
    let mut client = IpcClient::connect().await?;
    let capture = AudioCapture::default()?;
    let mut resampler = AudioResampler::new(capture.sample_rate);
    match trigger {
        Trigger::Vad => {
            println!(
                "listening at {} Hz, {} ch (VAD); Ctrl-C to stop",
                capture.sample_rate, capture.channels
            );
            listen_vad(&mut model, &mut client, &capture, &mut resampler).await
        }
        Trigger::Ptt => {
            listen_ptt(&mut model, &mut client, &capture, &mut resampler, &ptt_keys).await
        }
    }
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
        let chunk = tokio::task::block_in_place(|| capture.receiver().recv())?;
        let samples = resampler.process(&chunk);
        chunks += 1;
        if chunks % 100 == 0 {
            let rms =
                (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            println!(
                "audio: {chunks} chunks, rms={rms:.4}, noise={:.4}",
                segmenter.noise_floor()
            );
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
    ptt_keys: &[String],
) -> anyhow::Result<()> {
    let ptt = PttSocket::bind()?;
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
    loop {
        for event in ptt.poll() {
            match event {
                PttEvent::Down => {
                    if !gate.is_held() {
                        println!("ptt: recording");
                        let _ = client.preedit("...").await;
                    }
                    gate.set_held(true);
                }
                PttEvent::Up => {
                    let _ = client.clear().await;
                    if let Some(samples) = gate.release() {
                        commit_transcript(model, client, &samples).await?;
                    } else {
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
    let started = std::time::Instant::now();
    let result = tokio::task::block_in_place(|| model.transcribe(samples, model::MAX_NEW_TOKENS))?;
    println!(
        "inference finished in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    let text = result.text;
    if text.is_empty() {
        println!("recognition result: <empty/EOS> (model selected EOS)");
    } else if model::is_filler(&text) {
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
}
