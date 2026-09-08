/// Speech-boundary gates over 16 kHz mono PCM.

const TARGET_HZ: usize = 16_000;
pub const MIN_VAD_SAMPLES: usize = TARGET_HZ * 8 / 10;
pub const MIN_PTT_SAMPLES: usize = TARGET_HZ * 4 / 10;
pub const MAX_SAMPLES: usize = TARGET_HZ * 30;

fn rms(chunk: &[f32]) -> f32 {
    (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32).sqrt()
}

/// Energy gate with hangover, matching the previous listen-mode segmenter.
pub struct EnergyVad {
    active: Vec<f32>,
    silence_chunks: usize,
    threshold: f32,
    hangover: usize,
}

impl EnergyVad {
    pub fn new(threshold: f32, hangover: usize) -> Self {
        Self { active: Vec::new(), silence_chunks: 0, threshold, hangover }
    }

    pub fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        if rms(chunk) >= self.threshold {
            self.active.extend_from_slice(chunk);
            self.silence_chunks = 0;
            if self.active.len() >= MAX_SAMPLES {
                return Some(std::mem::take(&mut self.active));
            }
        } else if !self.active.is_empty() {
            self.active.extend_from_slice(chunk);
            self.silence_chunks += 1;
            if self.silence_chunks >= self.hangover {
                self.silence_chunks = 0;
                if self.active.len() >= MIN_VAD_SAMPLES {
                    return Some(std::mem::take(&mut self.active));
                }
                self.active.clear();
            }
        }
        None
    }
}

/// Hold-to-talk buffer. Audio is kept only while the key is down.
pub struct PttGate {
    held: bool,
    buf: Vec<f32>,
}

impl PttGate {
    pub fn new() -> Self {
        Self { held: false, buf: Vec::new() }
    }

    pub fn is_held(&self) -> bool {
        self.held
    }

    pub fn set_held(&mut self, held: bool) {
        if held {
            self.held = true;
        }
    }

    pub fn push(&mut self, chunk: &[f32]) {
        if !self.held {
            return;
        }
        self.buf.extend_from_slice(chunk);
        if self.buf.len() > MAX_SAMPLES {
            let extra = self.buf.len() - MAX_SAMPLES;
            self.buf.drain(..extra);
        }
    }

    pub fn release(&mut self) -> Option<Vec<f32>> {
        self.held = false;
        if self.buf.len() >= MIN_PTT_SAMPLES {
            Some(std::mem::take(&mut self.buf))
        } else {
            self.buf.clear();
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_emits_after_hangover() {
        let mut vad = EnergyVad::new(0.05, 2);
        let speech = vec![0.2; 8_000];
        let silence = vec![0.0; 8_000];
        assert!(vad.push(&speech).is_none());
        assert!(vad.push(&silence).is_none());
        let seg = vad.push(&silence).expect("segment");
        assert_eq!(seg.len(), 24_000);
    }

    #[test]
    fn vad_drops_short_noise() {
        let mut vad = EnergyVad::new(0.05, 1);
        assert!(vad.push(&vec![0.2; 100]).is_none());
        assert!(vad.push(&vec![0.0; 100]).is_none());
    }

    #[test]
    fn ptt_ignores_audio_until_held() {
        let mut ptt = PttGate::new();
        ptt.push(&vec![0.1; 8_000]);
        assert!(ptt.release().is_none());
        ptt.set_held(true);
        ptt.push(&vec![0.1; 8_000]);
        let seg = ptt.release().expect("ptt segment");
        assert_eq!(seg.len(), 8_000);
        assert!(!ptt.is_held());
    }
}
