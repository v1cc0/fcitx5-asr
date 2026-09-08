pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Stateful 16 kHz converter. Integer ratios (48 kHz → 16 kHz) average each
/// factor-sized block so chunk boundaries do not drop samples.
pub struct AudioResampler {
    input_rate: u32,
    step: f32,
    position: f32,
    tail: Vec<f32>,
}

impl AudioResampler {
    pub fn new(input_rate: u32) -> Self {
        assert!(input_rate > 0);
        Self {
            input_rate,
            step: input_rate as f32 / TARGET_SAMPLE_RATE as f32,
            position: 0.0,
            tail: Vec::new(),
        }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.input_rate == TARGET_SAMPLE_RATE {
            return input.to_vec();
        }
        if input.is_empty() {
            return Vec::new();
        }
        let mut buf = std::mem::take(&mut self.tail);
        buf.extend_from_slice(input);
        if self.input_rate % TARGET_SAMPLE_RATE == 0 {
            self.decimate_integer(&mut buf)
        } else {
            self.lerp(&mut buf)
        }
    }

    fn decimate_integer(&mut self, buf: &mut Vec<f32>) -> Vec<f32> {
        let factor = (self.input_rate / TARGET_SAMPLE_RATE) as usize;
        let n = buf.len() / factor;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let s = &buf[i * factor..(i + 1) * factor];
            out.push(s.iter().sum::<f32>() / factor as f32);
        }
        self.tail = buf[n * factor..].to_vec();
        out
    }

    fn lerp(&mut self, buf: &mut Vec<f32>) -> Vec<f32> {
        let mut out = Vec::with_capacity((buf.len() as f32 / self.step) as usize + 1);
        while self.position + 1.0 < buf.len() as f32 {
            let i = self.position as usize;
            let frac = self.position - i as f32;
            out.push(buf[i] * (1.0 - frac) + buf[i + 1] * frac);
            self.position += self.step;
        }
        let keep = self.position as usize;
        if keep < buf.len() {
            self.tail = buf[keep..].to_vec();
            self.position -= keep as f32;
        } else {
            self.tail.clear();
            self.position -= buf.len() as f32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_unchanged() {
        let mut r = AudioResampler::new(TARGET_SAMPLE_RATE);
        assert_eq!(r.process(&[0.0, 0.5, 1.0]), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn integer_decimate_averages_and_survives_chunks() {
        let src: Vec<f32> = (0..4800).map(|i| i as f32).collect();
        let mut whole = AudioResampler::new(48_000);
        let a = whole.process(&src);
        let mut chunked = AudioResampler::new(48_000);
        let mut b = Vec::new();
        for c in src.chunks(128) {
            b.extend(chunked.process(c));
        }
        assert_eq!(a.len(), 1600);
        assert_eq!(b, a);
        assert!((a[0] - 1.0).abs() < 1e-5);
        assert!((a[1] - 4.0).abs() < 1e-5);
    }
}
