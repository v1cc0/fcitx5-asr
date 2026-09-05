use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::debug;

pub fn default_socket_path() -> PathBuf {
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        let p = PathBuf::from(&runtime_dir).join("fcitx5-asr.sock");
        if p.exists() {
            return p;
        }
    }
    let uid = unsafe { libc::getuid() };
    let run_user = PathBuf::from(format!("/run/user/{}/fcitx5-asr.sock", uid));
    if run_user.exists() {
        return run_user;
    }
    let tmp_path = PathBuf::from(format!("/tmp/fcitx5-asr-{}.sock", uid));
    if tmp_path.exists() {
        return tmp_path;
    }
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(runtime_dir).join("fcitx5-asr.sock")
    } else {
        run_user
    }
}

pub struct IpcClient {
    stream: BufReader<UnixStream>,
}

impl IpcClient {
    pub async fn connect() -> Result<Self> {
        let path = default_socket_path();
        Self::connect_path(&path).await
    }

    pub async fn connect_path(path: &Path) -> Result<Self> {
        debug!("Connecting to Fcitx5 ASR bridge at {:?}", path);
        let stream = UnixStream::connect(path)
            .await
            .with_context(|| format!("Failed to connect to Fcitx5 socket at {:?}", path))?;
        Ok(Self {
            stream: BufReader::new(stream),
        })
    }

    pub async fn commit(&mut self, text: &str) -> Result<()> {
        let cmd = format!("COMMIT {}\n", text.trim());
        self.stream.get_mut().write_all(cmd.as_bytes()).await?;
        self.stream.get_mut().flush().await?;
        debug!("Sent commit: {:?}", text);
        Ok(())
    }

    pub async fn preedit(&mut self, text: &str) -> Result<()> {
        let cmd = format!("PREEDIT {}\n", text.trim());
        self.stream.get_mut().write_all(cmd.as_bytes()).await?;
        self.stream.get_mut().flush().await?;
        debug!("Sent preedit: {:?}", text);
        Ok(())
    }

    pub async fn clear(&mut self) -> Result<()> {
        self.stream.get_mut().write_all(b"CLEAR\n").await?;
        self.stream.get_mut().flush().await?;
        debug!("Sent clear");
        Ok(())
    }

    pub async fn ping(&mut self) -> Result<bool> {
        self.stream.get_mut().write_all(b"PING\n").await?;
        self.stream.get_mut().flush().await?;
        let mut response = String::new();
        self.stream.read_line(&mut response).await?;
        Ok(response.trim() == "PONG")
    }
}
