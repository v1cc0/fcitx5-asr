use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType, api::sync::Api};
use ndarray::Array2;
use ort::session::Session;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const REPO: &str = "AutoArk-AI/Audio8-ASR-0.1B-onnx-runtime";
const BUNDLE: &str = "model_bundle";
const AUDIO_HIDDEN: usize = 1024;
const HIDDEN: usize = 512;
const LAYERS: usize = 8;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const MAX_CACHE: usize = 512;
const MEL: usize = 128;
const FRAMES: usize = 3000;
const HOP: usize = 160;
const MERGE: usize = 4;
const AUDIO_TOKEN_ID: i64 = 151_646;
const PAD_TOKEN_ID: i64 = 151_643;
const EOS_TOKEN_ID: i64 = 151_645;
const EXTRA_BLOCK: [i64; 5] = [151_646, 151_647, 151_648, 151_649, 151_650];

/// Official OnnxCacheAsrEngine default.
pub const MAX_NEW_TOKENS: usize = 128;

pub struct ArkModel {
    pub audio: Session,
    pub prefill: Session,
    pub decode: Session,
    pub config: PathBuf,
    tokenizer: Tokenizer,
    token_embedding: Array2<f32>,
    proj_norm_w: Vec<f32>,
    proj_norm_b: Vec<f32>,
    proj_linear_w: Array2<f32>,
    proj_linear_b: Vec<f32>,
    extra_block: Vec<u8>,
}

pub struct Transcript {
    pub text: String,
}

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

fn ark_audio_token_count(sample_count: usize) -> usize {
    let mel_frames = sample_count / HOP;
    let downsampled = (mel_frames + 1) / 2;
    (downsampled / MERGE).max(1)
}

fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let dim = weight.len();
    for row in x.chunks_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..dim {
            row[i] = (row[i] - mean) * inv * weight[i] + bias[i];
        }
    }
}

fn adaptive_avg_pool_time(x: &[f32], in_t: usize, dim: usize, out_t: usize) -> Vec<f32> {
    if in_t == out_t {
        return x.to_vec();
    }
    let mut out = vec![0.0f32; out_t * dim];
    for o in 0..out_t {
        let start = o * in_t / out_t;
        let end = (((o + 1) * in_t + out_t - 1) / out_t)
            .max(start + 1)
            .min(in_t);
        let n = (end - start) as f32;
        for d in 0..dim {
            let mut s = 0.0;
            for t in start..end {
                s += x[t * dim + d];
            }
            out[o * dim + d] = s / n;
        }
    }
    out
}

fn cache_len() -> usize {
    KV_HEADS * MAX_CACHE * HEAD_DIM
}

fn write_cache_prefix(dst: &mut [f32], src: &[f32], src_time: usize) {
    for h in 0..KV_HEADS {
        for t in 0..src_time {
            let s = (h * src_time + t) * HEAD_DIM;
            let d = (h * MAX_CACHE + t) * HEAD_DIM;
            dst[d..d + HEAD_DIM].copy_from_slice(&src[s..s + HEAD_DIM]);
        }
    }
}

fn write_cache_pos(dst: &mut [f32], src: &[f32], pos: usize) {
    for h in 0..KV_HEADS {
        let s = h * HEAD_DIM;
        let d = (h * MAX_CACHE + pos) * HEAD_DIM;
        dst[d..d + HEAD_DIM].copy_from_slice(&src[s..s + HEAD_DIM]);
    }
}

impl ArkModel {
    pub fn transcribe(&mut self, samples: &[f32], max_tokens: usize) -> Result<Transcript> {
        let audio = self.audio_embeddings(samples)?;
        let n_audio = audio.len() / HIDDEN;
        let (ids, mut embeds) = self.prompt_embeddings(&audio, n_audio)?;
        let prompt_len = ids.len();
        anyhow::ensure!(
            prompt_len < MAX_CACHE,
            "prompt_len {prompt_len} >= cache {MAX_CACHE}"
        );
        let max_tokens = max_tokens.min(MAX_CACHE - prompt_len);
        let mut keys = vec![vec![0.0f32; cache_len()]; LAYERS];
        let mut values = vec![vec![0.0f32; cache_len()]; LAYERS];
        let mut logits = self.prefill(&embeds, &mut keys, &mut values, prompt_len)?;
        let first = self.greedy_token(&logits);
        println!(
            "ARK first token: {first} ({:?}), top={:?}",
            self.decode_tokens(&[first]),
            self.top_tokens(&logits, 5)
        );
        let mut generated = Vec::new();
        let mut last_token = None;
        let mut repeats = 0;
        let mut pos = prompt_len;
        for _ in 0..max_tokens {
            self.mask_logits(&mut logits);
            let token = self.greedy_token(&logits);
            if token == EOS_TOKEN_ID || token == PAD_TOKEN_ID {
                break;
            }
            generated.push(token);
            if last_token == Some(token) {
                repeats += 1;
                if repeats >= 4 {
                    break;
                }
            } else {
                repeats = 0;
                last_token = Some(token);
            }
            if let Some(n) = repeated_ngram(&generated) {
                generated.truncate(generated.len() - 2 * n);
                break;
            }
            embeds.clear();
            embeds.extend(self.token_embedding.row(token as usize).iter().copied());
            logits = self.decode_step(&embeds, &mut keys, &mut values, pos, pos + 1)?;
            pos += 1;
        }
        Ok(Transcript {
            text: normalize_prediction_text(&self.decode_tokens(&generated)),
        })
    }

    pub fn audio_embeddings(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let features = crate::audio::features::whisper_log_mel(samples);
        anyhow::ensure!(features.len() == MEL && features[0].len() == FRAMES);
        let mut data = Vec::with_capacity(MEL * FRAMES);
        for row in &features {
            data.extend_from_slice(row);
        }
        let n = samples.len().min(30 * 16_000).max(1);
        let feat_len = ((n + HOP - 1) / HOP).clamp(1, FRAMES) as i64;
        let input = ort::value::Tensor::from_array(ndarray::Array3::from_shape_vec(
            (1, MEL, FRAMES),
            data,
        )?)?;
        let lengths = ort::value::Tensor::from_array(ndarray::Array1::from_vec(vec![feat_len]))?;
        let outputs = self
            .audio
            .run(ort::inputs!["audios" => input, "audio_feature_lengths" => lengths])?;
        let hidden = outputs[0].try_extract_tensor::<f32>()?.1.to_vec();
        let valid: Vec<bool> = if let Ok((_, m)) = outputs[1].try_extract_tensor::<bool>() {
            m.to_vec()
        } else if let Ok((_, m)) = outputs[1].try_extract_tensor::<i64>() {
            m.iter().map(|x| *x != 0).collect()
        } else if let Ok((_, m)) = outputs[1].try_extract_tensor::<u8>() {
            m.iter().map(|x| *x != 0).collect()
        } else {
            anyhow::bail!("audio mask dtype unsupported");
        };
        anyhow::ensure!(hidden.len() % AUDIO_HIDDEN == 0);
        let tokens = hidden.len() / AUDIO_HIDDEN;
        let mut kept = Vec::new();
        let per = if valid.len() == tokens {
            1
        } else {
            AUDIO_HIDDEN
        };
        anyhow::ensure!(valid.len() == tokens || valid.len() == hidden.len());
        for t in 0..tokens {
            if valid[t * per] {
                kept.extend_from_slice(&hidden[t * AUDIO_HIDDEN..(t + 1) * AUDIO_HIDDEN]);
            }
        }
        let in_t = (kept.len() / AUDIO_HIDDEN).max(1);
        let want = ark_audio_token_count(n);
        let mut pooled = if in_t == want {
            kept
        } else {
            adaptive_avg_pool_time(&kept, in_t, AUDIO_HIDDEN, want)
        };
        layer_norm(&mut pooled, &self.proj_norm_w, &self.proj_norm_b);
        let tokens = pooled.len() / AUDIO_HIDDEN;
        let src = Array2::from_shape_vec((tokens, AUDIO_HIDDEN), pooled)?;
        let mut out = src.dot(&self.proj_linear_w.t());
        for mut row in out.rows_mut() {
            for (x, b) in row.iter_mut().zip(self.proj_linear_b.iter()) {
                *x += *b;
            }
        }
        Ok(out.iter().cloned().collect())
    }

    fn prompt_embeddings(&self, audio: &[f32], n_audio: usize) -> Result<(Vec<i64>, Vec<f32>)> {
        let audio_tok = "<|audio|>".repeat(n_audio);
        let prompt = format!(
            "<|user|><|begin_of_audio|>{audio_tok}<|end_of_audio|>Please transcribe this audio.<|assistant|>"
        );
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut embeds = Vec::with_capacity(ids.len() * HIDDEN);
        let mut audio_i = 0;
        for &id in &ids {
            if id == AUDIO_TOKEN_ID {
                anyhow::ensure!(audio_i < n_audio, "too many audio tokens");
                let s = audio_i * HIDDEN;
                embeds.extend_from_slice(&audio[s..s + HIDDEN]);
                audio_i += 1;
            } else {
                anyhow::ensure!((id as usize) < self.token_embedding.nrows());
                embeds.extend(self.token_embedding.row(id as usize).iter().copied());
            }
        }
        anyhow::ensure!(
            audio_i == n_audio,
            "prompt audio tokens {audio_i} != {n_audio}"
        );
        Ok((ids, embeds))
    }

    fn prefill(
        &mut self,
        embeds: &[f32],
        keys: &mut [Vec<f32>],
        values: &mut [Vec<f32>],
        prompt_len: usize,
    ) -> Result<Vec<f32>> {
        let input = ort::value::Tensor::from_array(ndarray::Array3::from_shape_vec(
            (1, prompt_len, HIDDEN),
            embeds.to_vec(),
        )?)?;
        let pos = ort::value::Tensor::from_array(ndarray::Array1::from_vec(
            (0..prompt_len as i64).collect(),
        ))?;
        let outputs = self
            .prefill
            .run(ort::inputs!["inputs_embeds" => input, "cache_position" => pos])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let seq = *shape.get(1).unwrap_or(&(prompt_len as i64)) as usize;
        let logits = last_token_logits(data, seq.max(1))?;
        for layer in 0..LAYERS {
            let k = outputs[1 + layer * 2].try_extract_tensor::<f32>()?.1;
            let v = outputs[2 + layer * 2].try_extract_tensor::<f32>()?.1;
            let kt = k.len() / (KV_HEADS * HEAD_DIM);
            write_cache_prefix(&mut keys[layer], k, kt);
            write_cache_prefix(&mut values[layer], v, kt);
        }
        Ok(logits)
    }

    fn decode_step(
        &mut self,
        embed: &[f32],
        keys: &mut [Vec<f32>],
        values: &mut [Vec<f32>],
        position: usize,
        valid_len: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(embed.len() == HIDDEN);
        let input = ort::value::Tensor::from_array(ndarray::Array3::from_shape_vec(
            (1, 1, HIDDEN),
            embed.to_vec(),
        )?)?;
        let mut attn = vec![0i64; MAX_CACHE];
        for x in attn.iter_mut().take(valid_len) {
            *x = 1;
        }
        let attn_t = ort::value::Tensor::from_array(Array2::from_shape_vec((1, MAX_CACHE), attn)?)?;
        let pos = ort::value::Tensor::from_array(ndarray::Array1::from_vec(vec![position as i64]))?;
        let mut inputs = ort::inputs!["inputs_embeds" => input, "attention_mask" => attn_t, "cache_position" => pos];
        for layer in 0..LAYERS {
            inputs.push((
                format!("cache_key_{layer}").into(),
                ort::value::Tensor::from_array((
                    vec![1usize, KV_HEADS, MAX_CACHE, HEAD_DIM],
                    keys[layer].clone(),
                ))?
                .upcast()
                .into(),
            ));
            inputs.push((
                format!("cache_value_{layer}").into(),
                ort::value::Tensor::from_array((
                    vec![1usize, KV_HEADS, MAX_CACHE, HEAD_DIM],
                    values[layer].clone(),
                ))?
                .upcast()
                .into(),
            ));
        }
        let outputs = self.decode.run(inputs)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let seq = *shape.get(1).unwrap_or(&1) as usize;
        let logits = last_token_logits(data, seq.max(1))?;
        for layer in 0..LAYERS {
            let k = outputs[1 + layer * 2].try_extract_tensor::<f32>()?.1;
            let v = outputs[2 + layer * 2].try_extract_tensor::<f32>()?.1;
            write_cache_pos(&mut keys[layer], k, position);
            write_cache_pos(&mut values[layer], v, position);
        }
        Ok(logits)
    }

    fn mask_logits(&self, logits: &mut [f32]) {
        for (id, flag) in self.extra_block.iter().enumerate() {
            if *flag != 0 && id < logits.len() {
                logits[id] = f32::NEG_INFINITY;
            }
        }
    }

    pub fn greedy_token(&self, logits: &[f32]) -> i64 {
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (id, &score) in logits.iter().enumerate() {
            if self.extra_block.get(id).copied().unwrap_or(0) == 0 && score > best.0 {
                best = (score, id);
            }
        }
        best.1 as i64
    }

    pub fn top_tokens(&self, logits: &[f32], n: usize) -> Vec<(i64, f32)> {
        let mut v: Vec<_> = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| self.extra_block.get(*i).copied().unwrap_or(0) == 0)
            .map(|(i, s)| (i as i64, *s))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1));
        v.truncate(n);
        v
    }

    pub fn decode_tokens(&self, ids: &[i64]) -> String {
        self.tokenizer
            .decode(&ids.iter().map(|id| *id as u32).collect::<Vec<_>>(), false)
            .unwrap_or_default()
    }

    pub fn download_and_load(int8: bool) -> Result<Self> {
        let api = Api::new().context("initializing Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(
            REPO.to_owned(),
            RepoType::Model,
            "main".to_owned(),
        ));
        let audio_name = if int8 {
            "audio_hidden_int8.onnx"
        } else {
            "audio_hidden.onnx"
        };
        let prefill_name = if int8 {
            "lm_cache_prefill_int8.onnx"
        } else {
            "lm_cache_prefill.onnx"
        };
        let decode_name = if int8 {
            "lm_cache_decode_int8.onnx"
        } else {
            "lm_cache_decode.onnx"
        };
        let audio_src = get(&repo, audio_name)?;
        let prefill_src = get(&repo, prefill_name)?;
        let decode_src = get(&repo, decode_name)?;
        let audio_path = materialize(&audio_src, get(&repo, &format!("{audio_name}.data")).ok())?;
        let prefill_path = materialize(
            &prefill_src,
            get(&repo, &format!("{prefill_name}.data")).ok(),
        )?;
        let decode_path =
            materialize(&decode_src, get(&repo, &format!("{decode_name}.data")).ok())?;
        let tok_path = get(&repo, "tokenizer.json")?;
        let emb_path = get(&repo, "weights/token_embedding.npy")?;
        let proj_path = get(&repo, "weights/audio_projector.npz")?;
        let meta = get(&repo, "metadata.json")?;
        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!(e))?;
        let (emb_shape, emb) = load_npy(&emb_path)?;
        anyhow::ensure!(
            emb_shape.len() == 2 && emb_shape[1] == HIDDEN,
            "token embedding shape {emb_shape:?}"
        );
        let token_embedding = Array2::from_shape_vec((emb_shape[0], emb_shape[1]), emb)?;
        let mut npz = npyz::npz::NpzArchive::open(&proj_path)?;
        let (nw_s, proj_norm_w) = load_npz_array(&mut npz, "norm_weight")?;
        let (_, proj_norm_b) = load_npz_array(&mut npz, "norm_bias")?;
        let (lw_s, lw) = load_npz_array(&mut npz, "linear_weight")?;
        let (_, proj_linear_b) = load_npz_array(&mut npz, "linear_bias")?;
        anyhow::ensure!(nw_s.last() == Some(&AUDIO_HIDDEN), "norm_weight {nw_s:?}");
        let proj_linear_w = if lw_s == [HIDDEN, AUDIO_HIDDEN] {
            Array2::from_shape_vec((HIDDEN, AUDIO_HIDDEN), lw)?
        } else if lw_s == [AUDIO_HIDDEN, HIDDEN] {
            Array2::from_shape_vec((AUDIO_HIDDEN, HIDDEN), lw)?
                .t()
                .to_owned()
        } else {
            anyhow::bail!("linear_weight shape {lw_s:?}");
        };
        let mut extra_block = vec![0u8; token_embedding.nrows().max(163_958)];
        for id in EXTRA_BLOCK {
            if let Some(slot) = extra_block.get_mut(id as usize) {
                *slot = 1;
            }
        }
        ort::init()
            .with_name("fcitx5-asr")
            .with_global_thread_pool(
                ort::environment::GlobalThreadPoolOptions::default()
                    .with_intra_threads(2)?
                    .with_inter_threads(1)?
                    .with_spin_control(false)?,
            )
            .commit();
        Ok(Self {
            audio: load_session(&audio_path)?,
            prefill: load_session(&prefill_path)?,
            decode: load_session(&decode_path)?,
            config: meta,
            tokenizer,
            token_embedding,
            proj_norm_w,
            proj_norm_b,
            proj_linear_w,
            proj_linear_b,
            extra_block,
        })
    }
}

fn last_token_logits(logits: &[f32], seq: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(
        seq > 0 && logits.len() % seq == 0,
        "logits {} not divisible by seq {seq}",
        logits.len()
    );
    let vocab = logits.len() / seq;
    Ok(logits[(seq - 1) * vocab..].to_vec())
}

fn load_session(path: &Path) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_memory_pattern(true)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(builder.commit_from_file(path)?)
}

fn get(repo: &hf_hub::api::sync::ApiRepo, name: &str) -> Result<PathBuf> {
    Ok(repo.get(&format!("{BUNDLE}/{name}"))?)
}

fn load_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let bytes = fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape = npy.shape().iter().map(|&d| d as usize).collect();
    Ok((shape, npy.into_vec::<f32>()?))
}

fn load_npz_array(
    npz: &mut npyz::npz::NpzArchive<std::io::BufReader<fs::File>>,
    name: &str,
) -> Result<(Vec<usize>, Vec<f32>)> {
    let arr = npz
        .by_name(name)?
        .with_context(|| format!("missing {name} in projector npz"))?;
    let shape = arr.shape().iter().map(|&d| d as usize).collect();
    Ok((shape, arr.into_vec::<f32>()?))
}

fn materialize(path: &Path, data: Option<PathBuf>) -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("fcitx5-asr-model");
    fs::create_dir_all(&dir)?;
    let dst = dir.join(path.file_name().context("invalid model filename")?);
    if !dst.exists() {
        fs::copy(path, &dst)?;
    }
    if let Some(data) = data {
        let data_dst = dir.join(format!(
            "{}.data",
            path.file_name().unwrap().to_string_lossy()
        ));
        if !data_dst.exists() {
            fs::copy(data, data_dst)?;
        }
    }
    Ok(dst)
}
pub fn normalize_prediction_text(text: &str) -> String {
    let mut t = text.to_string();
    let mut cut = t.len();
    for marker in ["<|user|>", "<|assistant|>", "<|im_end|>"] {
        if let Some(i) = t.find(marker) {
            cut = cut.min(i);
        }
    }
    t.truncate(cut);
    if let Some(i) = t.find("<|text|>") {
        t = t[i + "<|text|>".len()..].to_string();
    }
    t = strip_angle_tokens(&t);
    t = collapse_ws(&t);
    t = t
        .trim_start_matches(|c: char| {
            c.is_whitespace() || matches!(c, ',' | '.' | ';' | ':' | '!' | '?' | '-')
        })
        .to_string();
    collapse_repeated_phrase(t.trim())
}

fn strip_angle_tokens(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<|") {
        out.push_str(&rest[..start]);
        match rest[start..].find("|>") {
            Some(end) => rest = &rest[start + end + 2..],
            None => {
                rest = &rest[start..];
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

fn collapse_ws(text: &str) -> String {
    let mut out = String::new();
    let mut prev_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            prev_space = false;
            out.push(c);
        }
    }
    out
}

fn repeated_ngram(ids: &[i64]) -> Option<usize> {
    let len = ids.len();
    for n in 2..=48.min(len / 3) {
        if ids[len - n..] == ids[len - 2 * n..len - n]
            && ids[len - 2 * n..len - n] == ids[len - 3 * n..len - 2 * n]
        {
            return Some(n);
        }
    }
    None
}

fn is_sent_end(c: char) -> bool {
    matches!(c, '.' | '。' | '?' | '？' | '!' | '！')
}

fn collapse_repeated_phrase(text: &str) -> String {
    let t = text.trim();
    if t.is_empty() {
        return String::new();
    }
    collapse_duplicate_sentences(&collapse_tiled(t))
}

fn collapse_tiled(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n < 8 {
        return text.to_string();
    }
    for period in 4..=n / 3 {
        let last = chars[period - 1];
        if !(is_sent_end(last) || last == ' ') {
            continue;
        }
        let unit = &chars[..period];
        let mut copies = 0;
        let mut i = 0;
        while i + period <= n && chars[i..i + period] == *unit {
            copies += 1;
            i += period;
        }
        let tail_ok = i == n || unit.starts_with(&chars[i..]);
        if copies >= 3 && tail_ok {
            return unit.iter().collect::<String>().trim().to_string();
        }
    }
    text.to_string()
}

fn collapse_duplicate_sentences(text: &str) -> String {
    let mut parts = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if is_sent_end(c) {
            let piece = cur.trim().to_string();
            if !piece.is_empty() {
                parts.push(piece);
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    let mut out: Vec<String> = Vec::new();
    for p in parts {
        let core: String = p
            .chars()
            .filter(|c| !is_sent_end(*c) && !c.is_whitespace())
            .collect();
        let dup = out.last().is_some_and(|last| {
            let last_core: String = last
                .chars()
                .filter(|c| !is_sent_end(*c) && !c.is_whitespace())
                .collect();
            !core.is_empty() && last_core == core
        });
        if !dup {
            out.push(p);
        }
    }
    let mut s = String::new();
    for (i, p) in out.iter().enumerate() {
        if i > 0 {
            let prev_end = out[i - 1].chars().last().is_some_and(is_sent_end);
            let cjk = p.chars().next().is_some_and(|c| !c.is_ascii());
            if prev_end && !cjk && !p.starts_with(' ') {
                s.push(' ');
            }
        }
        s.push_str(p);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::{ark_audio_token_count, is_filler, normalize_prediction_text, repeated_ngram};

    #[test]
    fn audio_token_count_matches_official() {
        assert_eq!(ark_audio_token_count(16_000), 12);
        assert_eq!(ark_audio_token_count(480_000), 375);
    }

    #[test]
    fn repeated_ngram_catches_any_period() {
        let ids: Vec<i64> = [7, 8, 9, 13].iter().copied().cycle().take(12).collect();
        assert_eq!(repeated_ngram(&ids), Some(4));
        assert_eq!(repeated_ngram(&[1, 2, 3, 4, 5, 6]), None);
    }

    #[test]
    fn normalize_strips_turn_markers_and_repeats() {
        assert_eq!(
            normalize_prediction_text("Hello.<|im_end|> leftover"),
            "Hello."
        );
        let looped = "Okay. Let's talk. ".repeat(12) + "Okay. Let's";
        assert_eq!(normalize_prediction_text(&looped), "Okay. Let's talk.");
    }

    #[test]
    fn filler_catches_silence_prior() {
        for t in ["嗯。", "嗯", "uh", "Um..."] {
            assert!(is_filler(t), "{t}");
        }
    }

    #[test]
    fn filler_keeps_real_utterances() {
        for t in ["你好", "Hello.", "Are you listening to me?"] {
            assert!(!is_filler(t), "{t}");
        }
    }
}
