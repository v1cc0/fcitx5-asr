mod audio;
mod ipc;
mod model;

use clap::{Parser, Subcommand};
use ipc::IpcClient;

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
    }
    Ok(())
}
