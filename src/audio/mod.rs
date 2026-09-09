pub mod capture;
pub mod features;
pub mod resampler;
pub mod wav;

pub use capture::AudioCapture;
pub use resampler::{AudioResampler, TARGET_SAMPLE_RATE};
pub use wav::load_wav_16k;

/// One-pole DC blocker. Laptop mics often sit on a ~0.01 offset that
/// looks like speech to an energy VAD and biases Whisper log-mel.
pub struct DcBlocker {
    x1: f32,
    y1: f32,
}

impl DcBlocker {
    pub fn new() -> Self {
        Self { x1: 0.0, y1: 0.0 }
    }

    pub fn process_in_place(&mut self, samples: &mut [f32]) {
        const R: f32 = 0.995;
        for x in samples {
            let x0 = *x;
            let y = x0 - self.x1 + R * self.y1;
            self.x1 = x0;
            self.y1 = y;
            *x = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DcBlocker;

    #[test]
    fn dc_blocker_kills_offset() {
        let mut dc = DcBlocker::new();
        let mut samples = vec![0.01f32; 16_000];
        dc.process_in_place(&mut samples);
        let rms = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        assert!(rms < 0.002, "rms={rms}");
    }
}
