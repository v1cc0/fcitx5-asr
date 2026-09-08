use anyhow::{Context, Result};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

pub fn default_ctl_path() -> PathBuf {
    if let Ok(runtime_dir) = std::env::var("XDG_RUNTIME_DIR") {
        if !runtime_dir.is_empty() {
            return PathBuf::from(runtime_dir).join("fcitx5-asr-ctl.sock");
        }
    }
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/fcitx5-asr-ctl-{uid}.sock"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PttEvent {
    Down,
    Up,
}

/// Datagram socket the Fcitx5 addon uses to signal push-to-talk.
pub struct PttSocket {
    sock: UnixDatagram,
    path: PathBuf,
}

impl PttSocket {
    pub fn bind() -> Result<Self> {
        Self::bind_path(&default_ctl_path())
    }

    pub fn bind_path(path: &Path) -> Result<Self> {
        let _ = std::fs::remove_file(path);
        let sock = UnixDatagram::bind(path)
            .with_context(|| format!("binding PTT socket at {}", path.display()))?;
        sock.set_nonblocking(true)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        Ok(Self { sock, path: path.to_path_buf() })
    }

    pub fn poll(&self) -> Vec<PttEvent> {
        let mut events = Vec::new();
        let mut buf = [0u8; 16];
        for _ in 0..8 {
            match self.sock.recv(&mut buf) {
                Ok(n) => match &buf[..n] {
                    b"DOWN" => events.push(PttEvent::Down),
                    b"UP" => events.push(PttEvent::Up),
                    _ => {}
                },
                Err(_) => break,
            }
        }
        events
    }
}

impl Drop for PttSocket {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}
