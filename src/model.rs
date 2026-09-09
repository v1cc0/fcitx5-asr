use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType, api::sync::Api};
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Array4;
use ort::session::Session;
use ort::value::DynValue;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize)]
pub struct ArkConfig {
    pub eos_token_ids: Vec<i64>,
    pub prompt_prefix_ids: Vec<i64>,
    pub prompt_suffix_ids: Vec<i64>,
    pub suppress_token_ids: Vec<i64>,
    #[serde(default = "default_trim_features")]
    pub trim_features: usize,
}

fn default_trim_features() -> usize {
    8
}

const REPO: &str = "OpenVoiceOS/ARK-ASR-0.6B-onnx";
const VOCAB: usize = 163_958;
const HIDDEN: usize = 896;

pub struct ArkModel {
    pub encoder: Session,
    pub embed_tokens: Session,
    pub decoder: Session,
    pub config: PathBuf,
    pub settings: ArkConfig,
    tokenizer: Tokenizer,
    suppress_mask: Vec<u8>,
}

pub type KvCache = Vec<DynValue>;

pub struct Transcript {
    pub text: String,
    pub first_logit: f32,
}

/// True when the whole transcript is a hesitation marker or punctuation.
/// ARK's silence prior is "嗯。"; committing that to an IME is noise.
pub fn is_filler(text: &str) -> bool {
    let stripped: String = text
        .chars()
        .filter(|c| !c.is_whitespace() && !is_filler_punct(*c))
        .collect::<String>()
        .to_lowercase();
    if stripped.is_empty() {
        return true;
    }
    if stripped.chars().all(|c| {
        matches!(
            c,
            '嗯' | '啊' | '呃' | '哦' | '噢' | '喔' | '额' | '唔' | '哼' | '诶'
        )
    }) {
        return true;
    }
    matches!(
        stripped.as_str(),
        "uh" | "uhh"
            | "um"
            | "umm"
            | "hmm"
            | "hm"
            | "mm"
            | "mmm"
            | "mhm"
            | "mmhmm"
            | "ah"
            | "oh"
            | "er"
            | "eh"
            | "huh"
    )
}

fn is_filler_punct(c: char) -> bool {
    matches!(
        c,
        '.' | '。'
            | ','
            | '，'
            | '!'
            | '！'
            | '?'
            | '？'
            | ';'
            | '；'
            | ':'
            | '：'
            | '"'
            | '\''
            | '“'
            | '”'
            | '‘'
            | '’'
            | '…'
            | '~'
            | '～'
            | '-'
            | '—'
            | '、'
    )
}

impl ArkModel {
    pub fn transcribe(&mut self, samples: &[f32], max_tokens: usize) -> Result<Transcript> {
        let features = crate::audio::features::whisper_log_mel(samples);
        let trim = self.settings.trim_features.max(1);
        anyhow::ensure!(
            features[0].len() % trim == 0,
            "mel frames {} not divisible by trim_features {trim}",
            features[0].len()
        );
        let (audio, audio_tokens) = self.encode_with_shape(&features)?;
        anyhow::ensure!(
            audio.len() == audio_tokens * HIDDEN,
            "encoder embed mismatch"
        );
        anyhow::ensure!(
            audio_tokens == features[0].len() / trim,
            "encoder tokens {audio_tokens} != frames/trim {}",
            features[0].len() / trim
        );
        let (prompt, seq) = self.compose_prompt(&audio, audio_tokens)?;
        let (logits, mut cache) = self.decode_prompt_with_cache(&prompt, seq)?;
        let first = last_logits(&logits);
        let mut token = self.greedy_token(first);
        let first_logit = first
            .get(token as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        println!(
            "ARK first token: {token} ({:?}), top={:?}",
            self.decode_tokens(&[token]),
            self.top_tokens(first, 5)
        );
        let mut ids = Vec::new();
        let mut last_token = None;
        let mut repeats = 0;
        for _ in 0..max_tokens {
            if self.settings.eos_token_ids.contains(&token) {
                break;
            }
            ids.push(token);
            if last_token == Some(token) {
                repeats += 1;
                if repeats >= 4 {
                    break;
                }
            } else {
                repeats = 0;
                last_token = Some(token);
            }
            let embed = self.embed(&[token])?;
            let logits = self.decode_next(&embed, seq + ids.len() - 1, &mut cache)?;
            token = self.greedy_token(last_logits(&logits));
        }
        Ok(Transcript {
            text: self.decode_tokens(&ids).trim().to_owned(),
            first_logit,
        })
    }

    pub fn embed(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        let input =
            ort::value::Tensor::from_array(Array2::from_shape_vec((1, ids.len()), ids.to_vec())?)?;
        let outputs = self.embed_tokens.run(ort::inputs!["input_ids" => input])?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    /// Allocate the decoder's 24-layer empty KV cache (past length zero).
    pub fn empty_kv() -> Vec<ort::value::Tensor<f32>> {
        (0..48)
            .map(|_| {
                ort::value::Tensor::from_array((vec![1usize, 2, 0, 64], Vec::<f32>::new())).unwrap()
            })
            .collect()
    }

    pub fn decode_prompt_with_cache(
        &mut self,
        embeds: &[f32],
        tokens: usize,
    ) -> Result<(Vec<f32>, KvCache)> {
        let mut inputs = ort::inputs![
            "inputs_embeds" => ort::value::Tensor::from_array(Array3::from_shape_vec((1, tokens, HIDDEN), embeds.to_vec())?)?,
            "attn_bias" => ort::value::Tensor::from_array(Array4::from_shape_vec((1, 1, tokens, tokens), causal_bias(tokens))?)?,
            "position_ids" => ort::value::Tensor::from_array(Array2::from_shape_vec((1, tokens), (0..tokens as i64).collect())?)?,
        ];
        let kv = Self::empty_kv();
        for layer in 0..24 {
            inputs.push((
                format!("past_key_values.{layer}.key").into(),
                kv[layer * 2].clone().upcast().into(),
            ));
            inputs.push((
                format!("past_key_values.{layer}.value").into(),
                kv[layer * 2 + 1].clone().upcast().into(),
            ));
        }
        let mut outputs = self.decoder.run(inputs)?;
        let logits = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        let cache = (0..48)
            .filter_map(|i| {
                outputs.remove(format!(
                    "present.{}.{}",
                    i / 2,
                    if i % 2 == 0 { "key" } else { "value" }
                ))
            })
            .collect();
        Ok((logits, cache))
    }

    pub fn decode_next(
        &mut self,
        embed: &[f32],
        position: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            embed.len() == HIDDEN && cache.len() == 48,
            "invalid decoder step inputs"
        );
        let mut inputs = ort::inputs![
            "inputs_embeds" => ort::value::Tensor::from_array(Array3::from_shape_vec((1, 1, HIDDEN), embed.to_vec())?)?,
            "attn_bias" => ort::value::Tensor::from_array(Array4::from_shape_vec((1, 1, 1, position + 1), vec![0.0f32; position + 1])?)?,
            "position_ids" => ort::value::Tensor::from_array(Array2::from_shape_vec((1, 1), vec![position as i64])?)?,
        ];
        for layer in 0..24 {
            inputs.push((
                format!("past_key_values.{layer}.key").into(),
                (&cache[layer * 2]).into(),
            ));
            inputs.push((
                format!("past_key_values.{layer}.value").into(),
                (&cache[layer * 2 + 1]).into(),
            ));
        }
        let mut outputs = self.decoder.run(inputs)?;
        let logits = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        *cache = (0..48)
            .filter_map(|i| {
                outputs.remove(format!(
                    "present.{}.{}",
                    i / 2,
                    if i % 2 == 0 { "key" } else { "value" }
                ))
            })
            .collect();
        Ok(logits)
    }

    pub fn encode(&mut self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        Ok(self.encode_with_shape(features)?.0)
    }

    pub fn encode_with_shape(&mut self, features: &[Vec<f32>]) -> Result<(Vec<f32>, usize)> {
        anyhow::ensure!(
            features.len() == 128 && !features[0].is_empty(),
            "expected [128][frames] features"
        );
        let frames = features[0].len();
        let mut data = Vec::with_capacity(128 * frames);
        for row in features {
            anyhow::ensure!(row.len() == frames, "inconsistent feature frame count");
            data.extend_from_slice(row);
        }
        let input =
            ort::value::Tensor::from_array(Array3::from_shape_vec((1, 128, frames), data)?)?;
        let outputs = self.encoder.run(ort::inputs!["input_features" => input])?;
        let (shape, embeds) = outputs[0].try_extract_tensor::<f32>()?;
        Ok((embeds.to_vec(), *shape.get(1).unwrap_or(&0) as usize))
    }

    pub fn compose_prompt(
        &mut self,
        audio_embeds: &[f32],
        audio_frames: usize,
    ) -> Result<(Vec<f32>, usize)> {
        anyhow::ensure!(
            audio_embeds.len() == audio_frames * HIDDEN,
            "invalid audio embedding shape"
        );
        let mut ids = self.settings.prompt_prefix_ids.clone();
        ids.extend_from_slice(&self.settings.prompt_suffix_ids);
        let text = self.embed(&ids)?;
        let prefix_len = self.settings.prompt_prefix_ids.len() * HIDDEN;
        let mut out = Vec::with_capacity(text.len() + audio_embeds.len());
        out.extend_from_slice(&text[..prefix_len]);
        out.extend_from_slice(audio_embeds);
        out.extend_from_slice(&text[prefix_len..]);
        Ok((out, ids.len() + audio_frames))
    }

    pub fn download_and_load(int8: bool) -> Result<Self> {
        let api = Api::new().context("initializing Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(
            REPO.to_owned(),
            RepoType::Model,
            "main".to_owned(),
        ));
        let suffix = if int8 { ".int8.onnx" } else { ".onnx" };
        let encoder_src = repo.get(&format!("encoder{suffix}"))?;
        let decoder_src = repo.get(&format!("decoder{suffix}"))?;
        let embed_src = repo.get(&format!("embed_tokens{suffix}"))?;
        let encoder_data = repo.get(&format!("encoder{suffix}.data")).ok();
        let decoder_data = repo.get(&format!("decoder{suffix}.data")).ok();
        let embed_data = repo.get(&format!("embed_tokens{suffix}.data")).ok();
        let encoder_path = materialize(&encoder_src, encoder_data.as_deref())?;
        let decoder_path = materialize(&decoder_src, decoder_data.as_deref())?;
        let embed_path = materialize(&embed_src, embed_data.as_deref())?;
        let config = repo.get("config.json")?;
        let tok_repo = api.repo(Repo::with_revision(
            "Audio8/ARK-ASR-0.6B".to_owned(),
            RepoType::Model,
            "main".to_owned(),
        ));
        let tokenizer = Tokenizer::from_file(tok_repo.get("tokenizer.json")?)
            .map_err(|e| anyhow::anyhow!(e))?;
        let settings: ArkConfig = serde_json::from_slice(&std::fs::read(&config)?)?;
        let mut suppress_mask = vec![0u8; VOCAB];
        for &id in &settings.suppress_token_ids {
            if let Some(slot) = usize::try_from(id)
                .ok()
                .and_then(|i| suppress_mask.get_mut(i))
            {
                *slot = 1;
            }
        }
        ort::init().with_name("fcitx5-asr").commit();
        Ok(Self {
            encoder: Session::builder()?.commit_from_file(encoder_path)?,
            embed_tokens: Session::builder()?.commit_from_file(embed_path)?,
            decoder: Session::builder()?.commit_from_file(decoder_path)?,
            config,
            settings,
            tokenizer,
            suppress_mask,
        })
    }

    pub fn decode_tokens(&self, ids: &[i64]) -> String {
        self.tokenizer
            .decode(&ids.iter().map(|id| *id as u32).collect::<Vec<_>>(), true)
            .unwrap_or_default()
    }

    pub fn greedy_token(&self, logits: &[f32]) -> i64 {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (id, &score) in logits.iter().enumerate() {
            if !self.suppressed(id) && score > best.0 {
                best = (score, id);
            }
        }
        best.1 as i64
    }

    pub fn top_tokens(&self, logits: &[f32], n: usize) -> Vec<(i64, f32)> {
        let mut v: Vec<_> = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.suppressed(*i))
            .map(|(i, s)| (i as i64, *s))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(n);
        v
    }

    fn suppressed(&self, id: usize) -> bool {
        self.suppress_mask.get(id).copied().unwrap_or(0) != 0
    }
}

fn last_logits(logits: &[f32]) -> &[f32] {
    logits
        .get(logits.len().saturating_sub(VOCAB)..)
        .unwrap_or(logits)
}

fn causal_bias(n: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|i| (0..n).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect()
}

fn materialize(path: &std::path::Path, data: Option<&std::path::Path>) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("fcitx5-asr-model");
    fs::create_dir_all(&dir)?;
    let dst = dir.join(path.file_name().context("invalid model filename")?);
    if !dst.exists() {
        fs::copy(path, &dst)?;
    }
    let data_name = format!("{}.data", path.file_name().unwrap().to_string_lossy());
    if let Some(data) = data {
        let data_dst = dir.join(&data_name);
        if !data_dst.exists() {
            fs::copy(data, data_dst)?;
        }
    }
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::is_filler;

    #[test]
    fn filler_catches_ark_silence_prior() {
        for t in ["嗯。", "嗯", "嗯嗯", "啊", "uh", "Um...", "hmm"] {
            assert!(is_filler(t), "{t}");
        }
    }

    #[test]
    fn filler_keeps_real_utterances() {
        for t in ["你好", "你好吗？", "Hello.", "You are.", "来。", "嗯，你好"] {
            assert!(!is_filler(t), "{t}");
        }
    }
}
