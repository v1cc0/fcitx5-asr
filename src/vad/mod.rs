/// Speech-boundary gates over 16 kHz mono PCM.

const TARGET_HZ: usize = 16_000;
pub const MIN_VAD_SAMPLES: usize = TARGET_HZ * 8 / 10;
pub const MIN_PTT_SAMPLES: usize = TARGET_HZ * 4 / 10;
pub const MAX_SAMPLES: usize = TARGET_HZ * 30;
const PRE_ROLL_SAMPLES: usize = TARGET_HZ * 3 / 10;
const HANGOVER_SAMPLES: usize = TARGET_HZ * 4 / 10;

fn rms(chunk: &[f32]) -> f32 {
    (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32).sqrt()
}

/// Energy gate with time-based hangover and pre-roll.
/// Hangover is measured in samples so PipeWire's 128-frame periods do not
/// slice an utterance into 1-second fragments.
pub struct EnergyVad {
    active: Vec<f32>,
    preroll: Vec<f32>,
    silence_samples: usize,
    threshold: f32,
}

impl EnergyVad {
    pub fn new(threshold: f32) -> Self {
        Self { active: Vec::new(), preroll: Vec::new(), silence_samples: 0, threshold }
    }

    pub fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        if rms(chunk) >= self.threshold {
            if self.active.is_empty() {
                self.active.extend_from_slice(&self.preroll);
                self.preroll.clear();
            }
            self.active.extend_from_slice(chunk);
            self.silence_samples = 0;
            if self.active.len() >= MAX_SAMPLES {
                return Some(std::mem::take(&mut self.active));
            }
        } else if !self.active.is_empty() {
            self.active.extend_from_slice(chunk);
            self.silence_samples += chunk.len();
            if self.silence_samples >= HANGOVER_SAMPLES {
                self.silence_samples = 0;
                let seg = std::mem::take(&mut self.active);
                if seg.len() >= MIN_VAD_SAMPLES {
                    return Some(seg);
                }
            }
        } else {
            self.preroll.extend_from_slice(chunk);
            if self.preroll.len() > PRE_ROLL_SAMPLES {
                let extra = self.preroll.len() - PRE_ROLL_SAMPLES;
                self.preroll.drain(..extra);
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
    fn vad_emits_after_time_hangover() {
        let mut vad = EnergyVad::new(0.05);
        let speech = vec![0.2; 8_000];
        let silence = vec![0.0; 4_000];
        assert!(vad.push(&speech).is_none());
        assert!(vad.push(&silence).is_none());
        let seg = vad.push(&silence).expect("segment");
        assert_eq!(seg.len(), 16_000);
    }

    #[test]
    fn vad_keeps_preroll() {
        let mut vad = EnergyVad::new(0.05);
        assert!(vad.push(&vec![0.0; 4_800]).is_none());
        assert!(vad.push(&vec![0.2; 8_000]).is_none());
        let silence = vec![0.0; 6_400];
        let seg = vad.push(&silence).expect("segment");
        assert_eq!(seg.len(), 4_800 + 8_000 + 6_400);
    }

    #[test]
    fn vad_drops_short_noise() {
        let mut vad = EnergyVad::new(0.05);
        assert!(vad.push(&vec![0.2; 100]).is_none());
        assert!(vad.push(&vec![0.0; HANGOVER_SAMPLES]).is_none());
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
