mod audio;
mod ipc;
mod model;

use clap::{Parser, Subcommand};
use ipc::IpcClient;
use audio::{AudioCapture, AudioResampler};
use audio::segmenter::SpeechSegmenter;

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Ping,
    Commit { text: String },
    Preedit { text: String },
    Clear,
    /// Download and validate ARK-ASR ONNX files into the Hugging Face cache.
    Download { #[arg(long)] int8: bool },
    Inspect { #[arg(long)] int8: bool },
    Probe { #[arg(long)] int8: bool },
    Prompt { #[arg(long)] int8: bool },
    Listen { #[arg(long)] int8: bool },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    match Args::parse().command {
        Command::Ping => { let mut c = IpcClient::connect().await?; println!("{}", c.ping().await?); }
        Command::Commit { text } => { let mut c = IpcClient::connect().await?; c.commit(&text).await?; }
        Command::Preedit { text } => { let mut c = IpcClient::connect().await?; c.preedit(&text).await?; }
        Command::Clear => { let mut c = IpcClient::connect().await?; c.clear().await?; }
        Command::Download { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            println!("ARK-ASR loaded (config: {})", model.config.display());
        }
        Command::Inspect { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            for io in model.decoder.inputs() { println!("input {}: {:?}", io.name(), io.dtype()); }
            for io in model.decoder.outputs() { println!("output {}: {:?}", io.name(), io.dtype()); }
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
            let token = model.greedy_token(&logits[logits.len() - 163_958..]);
            println!("top tokens: {:?}", model.top_tokens(&logits[logits.len() - 163_958..], 5));
            let next_embed = model.embed(&[token])?;
            let next_logits = model.decode_next(&next_embed, ids.len(), &mut cache)?;
            let next = model.greedy_token(&next_logits[next_logits.len() - 163_958..]);
            println!("prompt next tokens: {token}, {next}; KV tensors: {}", cache.len());
        }
        Command::Listen { int8 } => listen(int8).await?,
    }
    Ok(())
}

async fn listen(int8: bool) -> anyhow::Result<()> {
    let mut model = model::ArkModel::download_and_load(int8)?;
    let mut client = IpcClient::connect().await?;
    let capture = AudioCapture::default()?;
    let mut resampler = AudioResampler::new(capture.sample_rate);
    let mut segmenter = SpeechSegmenter::new(0.01, 12);
    let mut chunks = 0usize;
    println!("listening at {} Hz; Ctrl-C to stop", capture.sample_rate);
    loop {
        let chunk = tokio::task::block_in_place(|| capture.receiver().recv())?;
        let samples = resampler.process(&chunk);
        chunks += 1;
        if chunks % 100 == 0 {
            let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len().max(1) as f32).sqrt();
            println!("audio: {chunks} chunks, rms={rms:.4}");
        }
        if let Some(samples) = segmenter.push(&samples) {
            println!("segment detected: {:.2}s; running ARK...", samples.len() as f32 / 16_000.0);
            capture.pause();
            while capture.receiver().try_recv().is_ok() {}
            let started = std::time::Instant::now();
            let result = tokio::task::block_in_place(|| model.transcribe(&samples, 16));
            capture.resume();
            let text = result?;
            println!("inference finished in {:.1}s", started.elapsed().as_secs_f32());
            if text.is_empty() {
                println!("recognition result: <empty/EOS> (model selected EOS)");
            } else {
                println!("recognition result: {text}");
                client.commit(&text).await?;
                println!("committed to Fcitx5");
            }
        }
    }
}
