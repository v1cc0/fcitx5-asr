/// Splits microphone chunks using a cheap RMS gate.
pub struct SpeechSegmenter { active: Vec<f32>, silence_chunks: usize, threshold: f32, hangover: usize }

impl SpeechSegmenter {
    pub fn new(threshold: f32, hangover: usize) -> Self { Self { active: Vec::new(), silence_chunks: 0, threshold, hangover } }
    pub fn push(&mut self, chunk: &[f32]) -> Option<Vec<f32>> {
        let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len().max(1) as f32).sqrt();
        if rms >= self.threshold {
            self.active.extend_from_slice(chunk);
            self.silence_chunks = 0;
        } else if !self.active.is_empty() {
            self.active.extend_from_slice(chunk);
            self.silence_chunks += 1;
            if self.silence_chunks >= self.hangover {
                self.silence_chunks = 0;
                if self.active.len() >= 12_800 { return Some(std::mem::take(&mut self.active)); }
                self.active.clear();
            }
        }
        None
    }
}
