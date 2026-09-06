use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use std::sync::mpsc::{self, Receiver, SyncSender};

/// Microphone capture as mono f32 chunks. The stream owns the callback thread;
/// callers drain `receiver` without touching cpal from async code.
pub struct AudioCapture {
    stream: Stream,
    receiver: Receiver<Vec<f32>>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioCapture {
    pub fn default() -> Result<Self> {
        let device = cpal::default_host()
            .default_input_device()
            .context("no default input device")?;
        let supported = device.default_input_config().context("no input config")?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let (tx, receiver) = mpsc::sync_channel(32);
        let channels = config.channels;
        let err = |e| tracing::error!("audio stream error: {e}");
        let stream = match sample_format {
            SampleFormat::F32 => build(&device, &config, channels, tx, err, |s: f32| s),
            SampleFormat::I16 => build(&device, &config, channels, tx, err, |s: i16| s as f32 / i16::MAX as f32),
            SampleFormat::U16 => build(&device, &config, channels, tx, err, |s: u16| (s as f32 / 32768.0) - 1.0),
            other => return Err(anyhow!("unsupported sample format: {other:?}")),
        }?;
        stream.play().context("starting input stream")?;
        Ok(Self { stream, receiver, sample_rate: config.sample_rate, channels })
    }

    pub fn receiver(&self) -> &Receiver<Vec<f32>> { &self.receiver }
    pub fn pause(&self) { let _ = self.stream.pause(); }
    pub fn resume(&self) { let _ = self.stream.play(); }
}

fn build<T: cpal::SizedSample, F: Fn(T) -> f32 + Send + 'static>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: u16,
    tx: SyncSender<Vec<f32>>,
    err: impl FnMut(cpal::Error) + Send + 'static,
    convert: F,
) -> Result<Stream> {
    let stream = device.build_input_stream(
        *config,
        move |data: &[T], _| {
            let mut mono = Vec::with_capacity(data.len() / channels as usize);
            for frame in data.chunks(channels as usize) {
                mono.push(frame.iter().map(|s| convert(*s)).sum::<f32>() / frame.len() as f32);
            }
            let _ = tx.try_send(mono);
        },
        err,
        None,
    )?;
    Ok(stream)
}
