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
    Listen,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let mut client = IpcClient::connect().await?;
    match Args::parse().command {
        Command::Ping => println!("{}", client.ping().await?),
        Command::Commit { text } => client.commit(&text).await?,
        Command::Preedit { text } => client.preedit(&text).await?,
        Command::Clear => client.clear().await?,
        Command::Download { int8 } => {
            let model = model::ArkModel::download_and_load(int8)?;
            println!("ARK-ASR loaded (config: {})", model.config.display());
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
