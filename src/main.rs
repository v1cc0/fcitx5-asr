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
    Listen,
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
        Command::Listen => listen().await?,
    }
    Ok(())
}

async fn listen() -> anyhow::Result<()> {
    let capture = AudioCapture::default()?;
    let mut resampler = AudioResampler::new(capture.sample_rate);
    let mut segmenter = SpeechSegmenter::new(0.01, 12);
    println!("listening at {} Hz; Ctrl-C to stop", capture.sample_rate);
    loop {
        let chunk = tokio::task::block_in_place(|| capture.receiver().recv())?;
        if let Some(samples) = segmenter.push(&resampler.process(&chunk)) {
            println!("speech segment: {:.2}s", samples.len() as f32 / 16_000.0);
        }
    }
}
