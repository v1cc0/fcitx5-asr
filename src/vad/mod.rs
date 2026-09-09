//! Speech-boundary gates over 16 kHz mono PCM.
//!
//! Energy is measured on 20 ms frames, never on raw cpal callback size.
//! PipeWire's 128-frame periods are ~2.7 ms at 48 kHz; RMS on that scale
//! treats clicks as speech. Hangover and minimum duration count *speech*
//! frames so preroll/hangover padding cannot promote a click into an
//! utterance.

const TARGET_HZ: usize = 16_000;
const FRAME: usize = 320; // 20 ms @ 16 kHz
const ONSET_FRAMES: usize = 4; // 80 ms consecutive speech to start
const HANGOVER_FRAMES: usize = 40; // 800 ms of silence to end
const PRE_ROLL_SAMPLES: usize = FRAME * 15; // 300 ms
const MIN_SPEECH_FRAMES: usize = 8; // 160 ms of actual speech
const MAX_SAMPLES: usize = TARGET_HZ * 30;
const NOISE_START_RATIO: f32 = 4.0;
const NOISE_CONTINUE_RATIO: f32 = 2.5;

pub const MIN_PTT_SAMPLES: usize = TARGET_HZ * 4 / 10;

fn rms(chunk: &[f32]) -> f32 {
    (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32).sqrt()
}

/// Energy gate with frame-based hangover, onset debounce, and a noise floor.
pub struct EnergyVad {
    pending: Vec<f32>,
    active: Vec<f32>,
    preroll: Vec<f32>,
    hangover: usize,
    onset: usize,
    speech_frames: usize,
    noise: f32,
    min_abs: f32,
}

impl EnergyVad {
    pub fn new(min_abs: f32) -> Self {
        Self {
            pending: Vec::new(),
            active: Vec::new(),
            preroll: Vec::new(),
            hangover: 0,
            onset: 0,
            speech_frames: 0,
            noise: min_abs / NOISE_START_RATIO,
            min_abs,
        }
    }

    pub fn noise_floor(&self) -> f32 {
        self.noise
    }

    fn start_threshold(&self) -> f32 {
        self.min_abs.max(self.noise * NOISE_START_RATIO)
    }

    fn continue_threshold(&self) -> f32 {
        (self.min_abs * 0.6).max(self.noise * NOISE_CONTINUE_RATIO)
    }

    fn observe_noise(&mut self, energy: f32) {
        if energy < self.noise {
            self.noise = 0.8 * self.noise + 0.2 * energy;
        } else {
            self.noise = 0.995 * self.noise + 0.005 * energy;
        }
    }

    pub fn push(&mut self, chunk: &[f32]) -> Vec<Vec<f32>> {
        self.pending.extend_from_slice(chunk);
        let mut segs = Vec::new();
        while self.pending.len() >= FRAME {
            if let Some(seg) = self.push_frame() {
                segs.push(seg);
            }
        }
        segs
    }

    fn push_frame(&mut self) -> Option<Vec<f32>> {
        let energy = rms(&self.pending[..FRAME]);
        if self.active.is_empty() {
            self.preroll.extend(self.pending.drain(..FRAME));
            if self.preroll.len() > PRE_ROLL_SAMPLES {
                let extra = self.preroll.len() - PRE_ROLL_SAMPLES;
                self.preroll.drain(..extra);
            }
            if energy >= self.start_threshold() {
                self.onset += 1;
                if self.onset >= ONSET_FRAMES {
                    self.active = std::mem::take(&mut self.preroll);
                    self.speech_frames = ONSET_FRAMES;
                    self.hangover = 0;
                    self.onset = 0;
                }
            } else {
                self.onset = 0;
                self.observe_noise(energy);
            }
            return None;
        }

        self.active.extend(self.pending.drain(..FRAME));
        if energy >= self.continue_threshold() {
            self.hangover = 0;
            self.speech_frames += 1;
        } else {
            self.hangover += 1;
        }
        if self.hangover >= HANGOVER_FRAMES || self.active.len() >= MAX_SAMPLES {
            self.finish_segment()
        } else {
            None
        }
    }

    fn finish_segment(&mut self) -> Option<Vec<f32>> {
        self.hangover = 0;
        self.onset = 0;
        let speech_frames = self.speech_frames;
        self.speech_frames = 0;
        let seg = std::mem::take(&mut self.active);
        if speech_frames >= MIN_SPEECH_FRAMES {
            Some(seg)
        } else {
            None
        }
    }
}

/// Hold-to-talk buffer. Audio is kept only while the key is down.
pub struct PttGate {
    held: bool,
    buf: Vec<f32>,
}

impl PttGate {
    pub fn new() -> Self {
        Self {
            held: false,
            buf: Vec::new(),
        }
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

    fn samples(seconds: f32, value: f32) -> Vec<f32> {
        vec![value; (seconds * TARGET_HZ as f32) as usize]
    }

    fn utterance(silence_head: f32, speech: f32, silence_tail: f32) -> Vec<f32> {
        let mut v = samples(silence_head, 0.0);
        v.extend(samples(speech, 0.2));
        v.extend(samples(silence_tail, 0.0));
        v
    }

    #[test]
    fn vad_emits_after_time_hangover() {
        let mut vad = EnergyVad::new(0.015);
        assert!(vad.push(&samples(0.5, 0.2)).is_empty());
        assert!(vad.push(&samples(0.4, 0.0)).is_empty());
        let segs = vad.push(&samples(0.5, 0.0));
        assert_eq!(segs.len(), 1);
        assert!(segs[0].len() >= 16_000);
    }

    #[test]
    fn vad_keeps_preroll() {
        let mut vad = EnergyVad::new(0.015);
        assert!(vad.push(&samples(0.3, 0.0)).is_empty());
        assert!(vad.push(&samples(0.5, 0.2)).is_empty());
        let segs = vad.push(&samples(0.9, 0.0));
        assert_eq!(segs.len(), 1);
        assert!(segs[0].len() >= 4_800 + 8_000);
    }

    #[test]
    fn vad_drops_short_noise() {
        let mut vad = EnergyVad::new(0.015);
        let mut click = samples(0.02, 0.5);
        click.extend(samples(1.0, 0.0));
        assert!(vad.push(&click).is_empty());
    }

    #[test]
    fn vad_ignores_constant_low_noise() {
        let mut vad = EnergyVad::new(0.015);
        assert!(vad.push(&samples(3.0, 0.012)).is_empty());
    }

    #[test]
    fn vad_padding_does_not_promote_click() {
        let mut vad = EnergyVad::new(0.015);
        // Old bug: 100 ms spike + 300 ms preroll + 400 ms hangover >= 800 ms min.
        let mut audio = samples(0.3, 0.0);
        audio.extend(samples(0.08, 0.5));
        audio.extend(samples(1.0, 0.0));
        assert!(vad.push(&audio).is_empty());
    }

    #[test]
    fn vad_is_independent_of_chunk_size() {
        let audio = utterance(0.3, 0.5, 0.9);
        let mut whole = EnergyVad::new(0.015);
        let a = whole.push(&audio);
        let mut chunked = EnergyVad::new(0.015);
        let mut b = Vec::new();
        for c in audio.chunks(42) {
            b.extend(chunked.push(c));
        }
        assert_eq!(a.len(), 1);
        assert_eq!(b.len(), 1);
        assert_eq!(a[0].len(), b[0].len());
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
