use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use ort::session::Session;
use std::path::PathBuf;

const REPO: &str = "OpenVoiceOS/ARK-ASR-0.6B-onnx";

pub struct ArkModel {
    pub encoder: Session,
    pub embed_tokens: Session,
    pub decoder: Session,
    pub vocab: PathBuf,
    pub config: PathBuf,
}

impl ArkModel {
    pub fn download_and_load(int8: bool) -> Result<Self> {
        let api = Api::new().context("initializing Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(REPO.to_owned(), RepoType::Model, "main".to_owned()));
        let suffix = if int8 { ".int8.onnx" } else { ".onnx" };
        let encoder_path = repo.get(&format!("encoder{suffix}"))?;
        let decoder_path = repo.get(&format!("decoder{suffix}"))?;
        let embed_path = repo.get(&format!("embed_tokens{suffix}"))?;
        let vocab = repo.get("vocab.json")?;
        let config = repo.get("config.json")?;
        ort::init().with_name("fcitx5-asr").commit();
        Ok(Self {
            encoder: Session::builder()?.commit_from_file(encoder_path)?,
            embed_tokens: Session::builder()?.commit_from_file(embed_path)?,
            decoder: Session::builder()?.commit_from_file(decoder_path)?,
            vocab,
            config,
        })
    }
}
