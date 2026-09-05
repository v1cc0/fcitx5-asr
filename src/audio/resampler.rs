pub const TARGET_SAMPLE_RATE: u32 = 16_000;

pub struct AudioResampler { input_rate: u32, position: f32 }

impl AudioResampler {
    pub fn new(input_rate: u32) -> Self { assert!(input_rate > 0); Self { input_rate, position: 0.0 } }
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if input.len() < 2 || self.input_rate == TARGET_SAMPLE_RATE { return input.to_vec(); }
        let step = self.input_rate as f32 / TARGET_SAMPLE_RATE as f32;
        let mut out = Vec::with_capacity((input.len() as f32 / step) as usize + 1);
        while self.position + 1.0 < input.len() as f32 {
            let i = self.position as usize;
            let frac = self.position - i as f32;
            out.push(input[i] * (1.0 - frac) + input[i + 1] * frac);
            self.position += step;
        }
        self.position -= input.len() as f32 - 1.0;
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
}
