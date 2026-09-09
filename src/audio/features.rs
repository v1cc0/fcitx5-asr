use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::OnceLock;

use super::resampler::TARGET_SAMPLE_RATE;

const NFFT: usize = 400;
const HOP: usize = 160;
const MEL: usize = 128;
const N_FREQ: usize = NFFT / 2 + 1;
const MAX_SAMPLES: usize = 30 * TARGET_SAMPLE_RATE as usize;
const FRAMES: usize = MAX_SAMPLES / HOP;

/// Whisper-128 log-mel features, shaped `[mel][frame]`.
///
/// Matches onnx-asr `whisper128`: Slaney-Slaney filterbank, periodic Hann,
/// centered reflect STFT, last time frame dropped. The encoder is exported
/// against a fixed 30-second / 3000-frame window; short utterances are
/// zero-padded, not truncated to a variable-length spectrogram.
pub fn whisper_log_mel(samples: &[f32]) -> Vec<Vec<f32>> {
    let n = samples.len().min(MAX_SAMPLES);
    let mut waveform = vec![0.0f32; MAX_SAMPLES];
    waveform[..n].copy_from_slice(&samples[..n]);

    let pad = NFFT / 2;
    let mut padded = vec![0.0f32; MAX_SAMPLES + NFFT];
    for i in 0..pad {
        padded[i] = waveform[pad - i];
        padded[pad + MAX_SAMPLES + i] = waveform[MAX_SAMPLES - 2 - i];
    }
    padded[pad..pad + MAX_SAMPLES].copy_from_slice(&waveform);

    let window = hann_window();
    let filters = mel_filters();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(NFFT);
    let mut buf = vec![Complex::default(); NFFT];
    let mut out = vec![vec![0.0; FRAMES]; MEL];

    for frame in 0..FRAMES {
        for (i, z) in buf.iter_mut().enumerate() {
            *z = Complex::new(padded[frame * HOP + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        let mut power = [0.0f32; N_FREQ];
        for (k, slot) in power.iter_mut().enumerate() {
            *slot = buf[k].norm_sqr();
        }
        for m in 0..MEL {
            let row = &filters[m * N_FREQ..(m + 1) * N_FREQ];
            let energy: f32 = row.iter().zip(power.iter()).map(|(w, p)| w * p).sum();
            out[m][frame] = energy.max(1e-10).log10();
        }
    }

    let max = out
        .iter()
        .flatten()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    for x in out.iter_mut().flatten() {
        *x = ((*x).max(max - 8.0) + 4.0) / 4.0;
    }
    out
}

fn hann_window() -> &'static [f32; NFFT] {
    static WINDOW: OnceLock<[f32; NFFT]> = OnceLock::new();
    WINDOW.get_or_init(|| {
        let mut w = [0.0f32; NFFT];
        for (i, slot) in w.iter_mut().enumerate() {
            *slot = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / NFFT as f32).cos();
        }
        w
    })
}

fn mel_filters() -> &'static [f32] {
    static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
    FILTERS.get_or_init(slaney_mel_filters)
}

fn hz_to_mel(freq: f64) -> f64 {
    if freq < 1000.0 {
        3.0 * freq / 200.0
    } else {
        15.0 + 27.0 * (freq / 1000.0 + f32::EPSILON as f64).ln() / 6.4f64.ln()
    }
}

fn mel_to_hz(mels: f64) -> f64 {
    if mels < 15.0 {
        200.0 * mels / 3.0
    } else {
        1000.0 * 6.4f64.powf((mels - 15.0) / 27.0)
    }
}

fn slaney_mel_filters() -> Vec<f32> {
    let nyquist = TARGET_SAMPLE_RATE as f64 / 2.0;
    let all_freqs: Vec<f64> = (0..N_FREQ)
        .map(|i| i as f64 * nyquist / (N_FREQ - 1) as f64)
        .collect();
    let m_min = hz_to_mel(0.0);
    let m_max = hz_to_mel(nyquist);
    let m_pts: Vec<f64> = (0..MEL + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (MEL + 1) as f64))
        .collect();
    let mut fb = vec![0.0f32; MEL * N_FREQ];
    for m in 0..MEL {
        let (left, center, right) = (m_pts[m], m_pts[m + 1], m_pts[m + 2]);
        let norm = 2.0 / (right - left);
        for (k, &freq) in all_freqs.iter().enumerate() {
            let up = (freq - left) / (center - left);
            let down = (right - freq) / (right - center);
            fb[m * N_FREQ + k] = (up.min(down).max(0.0) * norm) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} vs {b} (eps={eps})");
    }

    #[test]
    fn always_emits_whisper_30s_window() {
        let feat = whisper_log_mel(&vec![0.0; 1600]);
        assert_eq!(feat.len(), MEL);
        assert_eq!(feat[0].len(), FRAMES);
        assert_eq!(FRAMES % 8, 0);
    }

    #[test]
    fn periodic_hann_matches_numpy_hanning() {
        let w = hann_window();
        close(w[0], 0.0, 1e-7);
        close(w[200], 1.0, 1e-6);
        close(w[1], 6.1683759e-5, 1e-8);
        close(w[399], 6.1683759e-5, 1e-8);
    }

    #[test]
    fn slaney_filters_match_onnx_asr() {
        let fb = mel_filters();
        close(fb[0 * N_FREQ], 0.0, 0.0);
        close(fb[0 * N_FREQ + 1], 0.012373987, 1e-8);
        close(fb[1 * N_FREQ + 1], 0.030392564, 1e-8);
        close(fb[127 * N_FREQ + 200], 3.3208136e-9, 1e-15);
        let sum0: f32 = (0..N_FREQ).map(|k| fb[k]).sum();
        close(sum0, 0.012373987, 1e-8);
    }

    #[test]
    fn silence_normalizes_to_minus_one_point_five() {
        let feat = whisper_log_mel(&vec![0.0; 1280]);
        for &x in feat.iter().flatten() {
            close(x, -1.5, 1e-6);
        }
    }

    #[test]
    fn impulse_matches_numpy_reference() {
        let mut samples = vec![0.0; 1280];
        samples[0] = 1.0;
        let feat = whisper_log_mel(&samples);
        assert_eq!(feat[0].len(), FRAMES);
        close(feat[0][0], 0.5231274, 2e-3);
        close(feat[0][1], 0.01310978, 2e-3);
        close(feat[0][2], -1.3450135, 2e-3);
        close(feat[64][0], 0.59879314, 2e-3);
        close(feat[127][0], 0.5992069, 2e-3);
    }

    #[test]
    fn sine_440hz_matches_numpy_reference() {
        let sine: Vec<f32> = (0..16_000)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let feat = whisper_log_mel(&sine);
        assert_eq!(feat[0].len(), FRAMES);
        close(feat[0][0], 0.9075196, 3e-3);
        close(feat[20][0], 1.3302985, 3e-3);
        close(feat[40][0], 0.7472001, 3e-3);
        let (idx, max) = feat
            .iter()
            .map(|row| row[0])
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert_eq!(idx, 16);
        close(max, 1.4069679, 3e-3);
    }
}
