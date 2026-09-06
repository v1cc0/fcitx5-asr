use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use std::path::PathBuf;
use ndarray::Array3;
use ndarray::Array2;
use ndarray::Array4;
use serde::Deserialize;
use std::collections::HashMap;
use ort::value::DynValue;

#[derive(Debug, Deserialize)]
pub struct ArkConfig {
    pub eos_token_ids: Vec<i64>,
    pub prompt_prefix_ids: Vec<i64>,
    pub prompt_suffix_ids: Vec<i64>,
    pub suppress_token_ids: Vec<i64>,
}

const REPO: &str = "OpenVoiceOS/ARK-ASR-0.6B-onnx";

pub struct ArkModel {
    pub encoder: Session,
    pub embed_tokens: Session,
    pub decoder: Session,
    pub vocab: PathBuf,
    pub config: PathBuf,
    pub settings: ArkConfig,
    vocab_map: HashMap<String, i64>,
}

pub type KvCache = Vec<DynValue>;

impl ArkModel {
    pub fn transcribe(&mut self, samples: &[f32], max_tokens: usize) -> Result<String> {
        let features = crate::audio::features::whisper_log_mel(samples);
        let (audio, frames) = self.encode_with_shape(&features)?;
        let (prompt, seq) = self.compose_prompt(&audio, frames)?;
        let (logits, mut cache) = self.decode_prompt_with_cache(&prompt, seq)?;
        let mut token = self.greedy_token(&logits[logits.len().saturating_sub(163_958)..]);
        let mut ids = Vec::new();
        for _ in 0..max_tokens {
            if self.settings.eos_token_ids.contains(&token) { break; }
            ids.push(token);
            let embed = self.embed(&[token])?;
            let logits = self.decode_next(&embed, seq + ids.len() - 1, &mut cache)?;
            token = self.greedy_token(&logits[logits.len().saturating_sub(163_958)..]);
        }
        Ok(self.decode_tokens(&ids).trim().to_owned())
    }

    pub fn embed(&mut self, ids: &[i64]) -> Result<Vec<f32>> {
        let input = ort::value::Tensor::from_array(Array2::from_shape_vec((1, ids.len()), ids.to_vec())?)?;
        let outputs = self.embed_tokens.run(ort::inputs!["input_ids" => input])?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    /// Allocate the decoder's 24-layer empty KV cache (past length zero).
    pub fn empty_kv() -> Vec<ort::value::Tensor<f32>> {
        (0..48).map(|_| ort::value::Tensor::from_array((vec![1usize, 2, 0, 64], Vec::<f32>::new())).unwrap()).collect()
    }

    pub fn decode_prompt(&mut self, embeds: &[f32], tokens: usize) -> Result<Vec<f32>> {
        let mut inputs = ort::inputs![
            "inputs_embeds" => ort::value::Tensor::from_array(Array3::from_shape_vec((1, tokens, 896), embeds.to_vec())?)?,
            "attn_bias" => ort::value::Tensor::from_array(Array4::from_shape_vec((1, 1, tokens, tokens), vec![0.0f32; tokens * tokens])?)?,
            "position_ids" => ort::value::Tensor::from_array(Array2::from_shape_vec((1, tokens), (0..tokens as i64).collect())?)?,
        ];
        let kv = Self::empty_kv();
        for layer in 0..24 {
            inputs.push((format!("past_key_values.{layer}.key").into(), kv[layer * 2].clone().upcast().into()));
            inputs.push((format!("past_key_values.{layer}.value").into(), kv[layer * 2 + 1].clone().upcast().into()));
        }
        let outputs = self.decoder.run(inputs)?;
        Ok(outputs[0].try_extract_tensor::<f32>()?.1.to_vec())
    }

    pub fn decode_prompt_with_cache(&mut self, embeds: &[f32], tokens: usize) -> Result<(Vec<f32>, KvCache)> {
        let mut inputs = ort::inputs![
            "inputs_embeds" => ort::value::Tensor::from_array(Array3::from_shape_vec((1, tokens, 896), embeds.to_vec())?)?,
            "attn_bias" => ort::value::Tensor::from_array(Array4::from_shape_vec((1, 1, tokens, tokens), vec![0.0f32; tokens * tokens])?)?,
            "position_ids" => ort::value::Tensor::from_array(Array2::from_shape_vec((1, tokens), (0..tokens as i64).collect())?)?,
        ];
        let kv = Self::empty_kv();
        for layer in 0..24 {
            inputs.push((format!("past_key_values.{layer}.key").into(), kv[layer * 2].clone().upcast().into()));
            inputs.push((format!("past_key_values.{layer}.value").into(), kv[layer * 2 + 1].clone().upcast().into()));
        }
        let mut outputs = self.decoder.run(inputs)?;
        let logits = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        let cache = (0..48).filter_map(|i| outputs.remove(format!("present.{}.{}", i / 2, if i % 2 == 0 { "key" } else { "value" }))).collect();
        Ok((logits, cache))
    }

    pub fn decode_next(&mut self, embed: &[f32], position: usize, cache: &mut KvCache) -> Result<Vec<f32>> {
        anyhow::ensure!(embed.len() == 896 && cache.len() == 48, "invalid decoder step inputs");
        let mut inputs = ort::inputs![
            "inputs_embeds" => ort::value::Tensor::from_array(Array3::from_shape_vec((1, 1, 896), embed.to_vec())?)?,
            "attn_bias" => ort::value::Tensor::from_array(Array4::from_shape_vec((1, 1, 1, position + 1), vec![0.0f32; position + 1])?)?,
            "position_ids" => ort::value::Tensor::from_array(Array2::from_shape_vec((1, 1), vec![position as i64])?)?,
        ];
        for layer in 0..24 {
            inputs.push((format!("past_key_values.{layer}.key").into(), (&cache[layer * 2]).into()));
            inputs.push((format!("past_key_values.{layer}.value").into(), (&cache[layer * 2 + 1]).into()));
        }
        let mut outputs = self.decoder.run(inputs)?;
        let logits = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        *cache = (0..48).filter_map(|i| outputs.remove(format!("present.{}.{}", i / 2, if i % 2 == 0 { "key" } else { "value" }))).collect();
        Ok(logits)
    }

    pub fn prompt_next_token(&mut self, embeds: &[f32], tokens: usize) -> Result<i64> {
        let logits = self.decode_prompt(embeds, tokens)?;
        let vocab = 163_958;
        let last = logits.get(logits.len().saturating_sub(vocab)..).unwrap_or(&logits);
        Ok(self.greedy_token(last))
    }

    pub fn encode(&mut self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        Ok(self.encode_with_shape(features)?.0)
    }

    pub fn encode_with_shape(&mut self, features: &[Vec<f32>]) -> Result<(Vec<f32>, usize)> {
        anyhow::ensure!(features.len() == 128 && !features[0].is_empty(), "expected [128][frames] features");
        let frames = features[0].len();
        let mut data = Vec::with_capacity(128 * frames);
        for row in features { anyhow::ensure!(row.len() == frames, "inconsistent feature frame count"); data.extend_from_slice(row); }
        let input = ort::value::Tensor::from_array(Array3::from_shape_vec((1, 128, frames), data)?)?;
        let outputs = self.encoder.run(ort::inputs!["input_features" => input])?;
        let (shape, embeds) = outputs[0].try_extract_tensor::<f32>()?;
        Ok((embeds.to_vec(), *shape.get(1).unwrap_or(&0) as usize))
    }

    pub fn compose_prompt(&mut self, audio_embeds: &[f32], audio_frames: usize) -> Result<(Vec<f32>, usize)> {
        anyhow::ensure!(audio_embeds.len() == audio_frames * 896, "invalid audio embedding shape");
        let mut ids = self.settings.prompt_prefix_ids.clone();
        ids.extend_from_slice(&self.settings.prompt_suffix_ids);
        let text = self.embed(&ids)?;
        let prefix_len = self.settings.prompt_prefix_ids.len() * 896;
        let mut out = Vec::with_capacity(text.len() + audio_embeds.len());
        out.extend_from_slice(&text[..prefix_len]);
        out.extend_from_slice(audio_embeds);
        out.extend_from_slice(&text[prefix_len..]);
        Ok((out, ids.len() + audio_frames))
    }

    pub fn download_and_load(int8: bool) -> Result<Self> {
        let api = Api::new().context("initializing Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(REPO.to_owned(), RepoType::Model, "main".to_owned()));
        let suffix = if int8 { ".int8.onnx" } else { ".onnx" };
        let encoder_path = repo.get(&format!("encoder{suffix}"))?;
        let decoder_path = repo.get(&format!("decoder{suffix}"))?;
        let embed_path = repo.get(&format!("embed_tokens{suffix}"))?;
        let vocab = repo.get("vocab.json")?;
        let config = repo.get("config.json")?;
        let settings: ArkConfig = serde_json::from_slice(&std::fs::read(&config)?)?;
        let vocab_map: HashMap<String, i64> = serde_json::from_slice(&std::fs::read(&vocab)?)?;
        ort::init().with_name("fcitx5-asr").commit();
        Ok(Self {
            encoder: Session::builder()?.commit_from_file(encoder_path)?,
            embed_tokens: Session::builder()?.commit_from_file(embed_path)?,
            decoder: Session::builder()?.commit_from_file(decoder_path)?,
            vocab,
            config,
            settings,
            vocab_map,
        })
    }

    pub fn token_for_id(&self, id: i64) -> Option<&str> {
        self.vocab_map.iter().find_map(|(token, value)| (*value == id).then_some(token.as_str()))
    }

    pub fn decode_tokens(&self, ids: &[i64]) -> String {
        ids.iter().filter_map(|id| self.token_for_id(*id)).map(|s| s.replace('Ġ', " ")).collect()
    }

    pub fn greedy_token(&self, logits: &[f32]) -> i64 {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (id, &score) in logits.iter().enumerate() {
            if !self.settings.suppress_token_ids.contains(&(id as i64)) && score > best.0 { best = (score, id); }
        }
        best.1 as i64
    }
}
