use anyhow::{Context, Result, anyhow};
use std::path::Path;

use super::resampler::AudioResampler;

/// Load a WAV file as 16 kHz mono f32 in [-1, 1].
pub fn load_wav_16k(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    anyhow::ensure!(channels > 0, "wav has no channels");
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1u32 << spec.bits_per_sample.saturating_sub(1).max(1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let mut mono = Vec::with_capacity(samples.len() / channels);
    for frame in samples.chunks(channels) {
        if frame.len() != channels {
            return Err(anyhow!("truncated wav frame"));
        }
        mono.push(frame.iter().sum::<f32>() / channels as f32);
    }
    Ok(AudioResampler::new(spec.sample_rate).process(&mono))
}
