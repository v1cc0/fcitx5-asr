use super::ctl::{pttkeys_command, validate_ptt_key};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::time::{Duration, timeout};
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
        self.expect_ok().await?;
        debug!("Sent commit: {:?}", text);
        Ok(())
    }

    pub async fn preedit(&mut self, text: &str) -> Result<()> {
        let cmd = format!("PREEDIT {}\n", text.trim());
        self.stream.get_mut().write_all(cmd.as_bytes()).await?;
        self.stream.get_mut().flush().await?;
        self.expect_ok().await?;
        debug!("Sent preedit: {:?}", text);
        Ok(())
    }

    pub async fn clear(&mut self) -> Result<()> {
        self.stream.get_mut().write_all(b"CLEAR\n").await?;
        self.stream.get_mut().flush().await?;
        self.expect_ok().await?;
        debug!("Sent clear");
        Ok(())
    }

    pub async fn set_ptt_keys(&mut self, keys: &[String]) -> Result<()> {
        anyhow::ensure!(!keys.is_empty(), "at least one --ptt-key is required");
        for key in keys {
            validate_ptt_key(key)?;
        }
        self.write_line(&pttkeys_command(keys)).await?;
        let _ = self.read_ok(true).await?;
        debug!("Set PTT keys: {:?}", keys);
        Ok(())
    }

    pub async fn get_ptt_keys(&mut self) -> Result<Vec<String>> {
        self.write_line(&pttkeys_command(&[])).await?;
        let rest = self.read_ok(true).await?;
        Ok(rest
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect())
    }

    async fn write_line(&mut self, cmd: &str) -> Result<()> {
        self.stream.get_mut().write_all(cmd.as_bytes()).await?;
        self.stream.get_mut().flush().await?;
        Ok(())
    }

    async fn expect_ok(&mut self) -> Result<()> {
        let _ = self.read_ok(false).await?;
        Ok(())
    }

    async fn read_ok(&mut self, required: bool) -> Result<String> {
        let mut response = String::new();
        match timeout(
            Duration::from_millis(500),
            self.stream.read_line(&mut response),
        )
        .await
        {
            Ok(result) => {
                result?;
                let line = response.trim();
                if let Some(rest) = line.strip_prefix("OK") {
                    return Ok(rest.trim().to_string());
                }
                anyhow::bail!("bridge rejected command: {line}");
            }
            Err(_) if required => anyhow::bail!(
                "bridge did not acknowledge command (update and restart the Fcitx5 ASR addon?)"
            ),
            Err(_) => {
                tracing::warn!("bridge acknowledgement timed out; command may still be processed")
            }
        }
        Ok(String::new())
    }

    pub async fn ping(&mut self) -> Result<bool> {
        self.stream.get_mut().write_all(b"PING\n").await?;
        self.stream.get_mut().flush().await?;
        let mut response = String::new();
        self.stream.read_line(&mut response).await?;
        Ok(response.trim() == "PONG")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::UnixListener;

    async fn spawn_bridge(
        path: &Path,
        replies: Vec<&'static str>,
    ) -> tokio::task::JoinHandle<Vec<String>> {
        let listener = UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut seen = Vec::new();
            for reply in replies {
                let mut line = String::new();
                reader.read_line(&mut line).await.unwrap();
                seen.push(line);
                reader.get_mut().write_all(reply.as_bytes()).await.unwrap();
                reader.get_mut().flush().await.unwrap();
            }
            seen
        })
    }

    #[tokio::test]
    async fn pttkeys_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "fcitx5-asr-pttkeys-{}-{}.sock",
            std::process::id(),
            1
        ));
        let _ = std::fs::remove_file(&path);
        let server = spawn_bridge(&path, vec!["OK\n", "OK Super_L\n"]).await;
        let mut client = IpcClient::connect_path(&path).await.unwrap();
        client.set_ptt_keys(&["Super_L".into()]).await.unwrap();
        let keys = client.get_ptt_keys().await.unwrap();
        assert_eq!(keys, vec!["Super_L".to_string()]);
        let seen = server.await.unwrap();
        assert_eq!(
            seen,
            vec!["PTTKEYS Super_L\n".to_string(), "PTTKEYS\n".to_string()]
        );
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn pttkeys_bridge_error() {
        let path = std::env::temp_dir().join(format!(
            "fcitx5-asr-pttkeys-{}-{}.sock",
            std::process::id(),
            2
        ));
        let _ = std::fs::remove_file(&path);
        let server = spawn_bridge(&path, vec!["ERR invalid key: Nope\n"]).await;
        let mut client = IpcClient::connect_path(&path).await.unwrap();
        let err = client.set_ptt_keys(&["Nope".into()]).await.unwrap_err();
        assert!(err.to_string().contains("invalid key"));
        server.await.unwrap();
        let _ = std::fs::remove_file(&path);
    }
}
