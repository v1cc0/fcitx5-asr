use rustfft::{FftPlanner, num_complex::Complex};

const NFFT: usize = 400;
const HOP: usize = 160;
const MEL: usize = 128;

/// Whisper-compatible 128-bin log-mel features, shaped `[mel][frame]`.
pub fn whisper_log_mel(samples: &[f32]) -> Vec<Vec<f32>> {
    // ARK exports the Whisper encoder with a fixed 30-second (3000-frame) window.
    let frames = 3000;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(NFFT);
    let filters = mel_filters();
    let mut out = vec![vec![0.0; frames]; MEL];
    let mut buf = vec![Complex::default(); NFFT];
    for frame in 0..frames {
        for (i, z) in buf.iter_mut().enumerate() {
            let s = samples.get(frame * HOP + i).copied().unwrap_or(0.0);
            *z = Complex::new(s * (std::f32::consts::PI * (i as f32 + 0.5) / NFFT as f32).sin().powi(2), 0.0);
        }
        fft.process(&mut buf);
        let power: Vec<f32> = buf[..=NFFT / 2].iter().map(|z| z.norm_sqr()).collect();
        for m in 0..MEL { out[m][frame] = filters[m].iter().zip(&power).map(|(w, p)| w * p).sum::<f32>().max(1e-10).log10(); }
    }
    let max = out.iter().flatten().copied().fold(f32::NEG_INFINITY, f32::max);
    for x in out.iter_mut().flatten() { *x = ((*x).max(max - 8.0) + 4.0) / 4.0; }
    out
}

fn mel_filters() -> Vec<Vec<f32>> {
    let hz = |m: f32| 700.0 * (10f32.powf(m / 2595.0) - 1.0);
    let mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let lo = mel(0.0); let hi = mel(8000.0);
    let bins: Vec<usize> = (0..MEL + 2).map(|i| ((NFFT as f32 + 1.0) * hz(lo + (hi-lo)*i as f32/(MEL+1) as f32) / 16000.0).floor() as usize).collect();
    (0..MEL).map(|m| (0..=NFFT/2).map(|k| {
        let (a,b,c) = (bins[m], bins[m+1], bins[m+2]);
        if k < a || k > c { 0.0 } else if k <= b { (k-a) as f32 / (b-a).max(1) as f32 } else { (c-k) as f32 / (c-b).max(1) as f32 }
    }).collect()).collect()
}

#[cfg(test)]
mod tests { use super::*; #[test] fn shape() { let x = whisper_log_mel(&vec![0.0; 1600]); assert_eq!(x.len(), MEL); assert!(!x[0].is_empty()); } }
