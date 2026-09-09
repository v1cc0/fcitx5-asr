use anyhow::{Context, Result};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixDatagram;
use std::path::{Path, PathBuf};

/// Addon defaults: Right Ctrl or Pause in a focused Fcitx5 window.
pub const DEFAULT_PTT_KEYS: &[&str] = &["Control_R", "Pause"];

pub fn pttkeys_command(keys: &[String]) -> String {
    if keys.is_empty() {
        "PTTKEYS\n".to_string()
    } else {
        format!("PTTKEYS {}\n", keys.join(" "))
    }
}

pub fn validate_ptt_key(key: &str) -> Result<()> {
    anyhow::ensure!(!key.is_empty(), "empty --ptt-key");
    anyhow::ensure!(
        !key.chars()
            .any(|c| c.is_whitespace() || c == '\n' || c == '\r'),
        "invalid --ptt-key {key:?}: use a Fcitx5 key name such as Control_R, Super_L, Pause, F8"
    );
    Ok(())
}

pub fn display_ptt_key(name: &str) -> String {
    match name {
        "Control_R" => "Right Ctrl".to_string(),
        "Control_L" => "Left Ctrl".to_string(),
        "Super_L" => "Left Super".to_string(),
        "Super_R" => "Right Super".to_string(),
        "Alt_L" => "Left Alt".to_string(),
        "Alt_R" => "Right Alt".to_string(),
        "Caps_Lock" => "Caps Lock".to_string(),
        other => other.replace('_', " "),
    }
}

pub fn format_ptt_hold(keys: &[String]) -> String {
    let shown: Vec<String> = if keys.is_empty() {
        DEFAULT_PTT_KEYS
            .iter()
            .map(|k| display_ptt_key(k))
            .collect()
    } else {
        keys.iter().map(|k| display_ptt_key(k)).collect()
    };
    shown.join(" or ")
}

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
        Ok(Self {
            sock,
            path: path.to_path_buf(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pttkeys_command_get_and_set() {
        assert_eq!(pttkeys_command(&[]), "PTTKEYS\n");
        assert_eq!(
            pttkeys_command(&["Super_L".into(), "Pause".into()]),
            "PTTKEYS Super_L Pause\n"
        );
    }

    #[test]
    fn validate_ptt_key_rejects_protocol_breakers() {
        assert!(validate_ptt_key("Super_L").is_ok());
        assert!(validate_ptt_key("Control_R").is_ok());
        assert!(validate_ptt_key("").is_err());
        assert!(validate_ptt_key("Control R").is_err());
        assert!(validate_ptt_key("Super_L\nPTTKEYS").is_err());
    }

    #[test]
    fn format_ptt_hold_uses_defaults_and_aliases() {
        assert_eq!(format_ptt_hold(&[]), "Right Ctrl or Pause");
        assert_eq!(
            format_ptt_hold(&["Super_L".into(), "F8".into()]),
            "Left Super or F8"
        );
    }
}
