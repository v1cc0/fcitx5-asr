mod audio;
mod ipc;
mod model;
mod vad;

use audio::{AudioCapture, AudioResampler, DcBlocker, TARGET_SAMPLE_RATE, load_wav_16k};
use clap::{Parser, Subcommand, ValueEnum};
use ipc::{IpcClient, PttEvent, PttSocket};
use vad::{EnergyVad, PttGate};

const VAD_MIN_ABS: f32 = 0.015;
const MIN_FIRST_LOGIT: f32 = 1.5;

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
    /// Hold Right Ctrl or Pause in a focused Fcitx5 window.
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
    /// Download and validate ARK-ASR ONNX files into the Hugging Face cache.
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
            println!("ARK-ASR loaded (config: {})", model.config.display());
        }
        Command::Inspect { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            for io in model.decoder.inputs() {
                println!("input {}: {:?}", io.name(), io.dtype());
            }
            for io in model.decoder.outputs() {
                println!("output {}: {:?}", io.name(), io.dtype());
            }
        }
        Command::Probe { int8 } => {
            let mut model = model::ArkModel::download_and_load(int8)?;
            let features = audio::features::whisper_log_mel(&vec![0.0; 16_000]);
            let embeds = model.encode(&features)?;
            println!("encoder probe: {} values", embeds.len());
        }
        Command::Prompt { int8 } => {
            let mut model = model::ArkModel::download_and_load(int8)?;
            let mut ids = model.settings.prompt_prefix_ids.clone();
            ids.extend_from_slice(&model.settings.prompt_suffix_ids);
            let embeds = model.embed(&ids)?;
            let (logits, mut cache) = model.decode_prompt_with_cache(&embeds, ids.len())?;
            let last = &logits[logits.len().saturating_sub(163_958)..];
            let token = model.greedy_token(last);
            println!("top tokens: {:?}", model.top_tokens(last, 5));
            let next_embed = model.embed(&[token])?;
            let next_logits = model.decode_next(&next_embed, ids.len(), &mut cache)?;
            let next =
                model.greedy_token(&next_logits[next_logits.len().saturating_sub(163_958)..]);
            println!(
                "prompt next tokens: {token}, {next}; KV tensors: {}",
                cache.len()
            );
        }
        Command::Listen { int8, trigger } => listen(int8, trigger).await?,
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
            let result = model.transcribe(&samples, 64)?;
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

async fn listen(int8: bool, trigger: Trigger) -> anyhow::Result<()> {
    let mut model = model::ArkModel::download_and_load(int8)?;
    let mut client = IpcClient::connect().await?;
    let capture = AudioCapture::default()?;
    let mut resampler = AudioResampler::new(capture.sample_rate);
    println!(
        "listening at {} Hz, {} ch ({}); Ctrl-C to stop",
        capture.sample_rate,
        capture.channels,
        match trigger {
            Trigger::Vad => "VAD",
            Trigger::Ptt => "PTT: hold Right Ctrl or Pause",
        }
    );
    match trigger {
        Trigger::Vad => listen_vad(&mut model, &mut client, &capture, &mut resampler).await,
        Trigger::Ptt => listen_ptt(&mut model, &mut client, &capture, &mut resampler).await,
    }
}

async fn listen_vad(
    model: &mut model::ArkModel,
    client: &mut IpcClient,
    capture: &AudioCapture,
    resampler: &mut AudioResampler,
) -> anyhow::Result<()> {
    let mut segmenter = EnergyVad::new(VAD_MIN_ABS);
    let mut dc = DcBlocker::new();
    let mut chunks = 0usize;
    loop {
        let chunk = tokio::task::block_in_place(|| capture.receiver().recv())?;
        let mut samples = resampler.process(&chunk);
        dc.process_in_place(&mut samples);
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
) -> anyhow::Result<()> {
    let ptt = PttSocket::bind()?;
    let mut gate = PttGate::new();
    let mut dc = DcBlocker::new();
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
            Ok(chunk) => {
                let mut samples = resampler.process(&chunk);
                dc.process_in_place(&mut samples);
                gate.push(&samples);
            }
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
    let result = tokio::task::block_in_place(|| model.transcribe(samples, 64))?;
    println!(
        "inference finished in {:.1}s",
        started.elapsed().as_secs_f32()
    );
    let text = result.text;
    if text.is_empty() {
        println!("recognition result: <empty/EOS> (model selected EOS)");
    } else if model::is_filler(&text) {
        println!("recognition result: {text} (skipped filler)");
    } else if result.first_logit < MIN_FIRST_LOGIT {
        println!(
            "recognition result: {text} (skipped low confidence {:.2})",
            result.first_logit
        );
    } else {
        println!("recognition result: {text}");
        client.commit(&text).await?;
        println!("committed to Fcitx5");
    }
    Ok(())
}
