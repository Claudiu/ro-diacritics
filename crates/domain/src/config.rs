//! `config.json` as written by `src/diacritics/export/safetensors.py`.

use serde::Deserialize;

use crate::alphabet::Alphabet;

pub const FORMAT_VERSION: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelDims {
    pub vocab_size: usize,
    pub window: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_classes: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExportConfig {
    pub format_version: u32,
    pub model: ModelDims,
    pub alphabet: Alphabet,
    pub labels: Vec<String>,
    pub overlap: usize,
    pub threshold: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config.json: {0}")]
    Parse(String),

    #[error("unsupported export format {0}, expected {FORMAT_VERSION}")]
    Format(u32),

    #[error("config.json is inconsistent: {0}")]
    Invalid(&'static str),
}

impl ExportConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.format_version != FORMAT_VERSION {
            return Err(ConfigError::Format(self.format_version));
        }
        let m = &self.model;
        if m.vocab_size != self.alphabet.size() {
            return Err(ConfigError::Invalid("vocab_size differs from alphabet"));
        }
        if m.n_classes != self.labels.len() || m.n_classes != crate::Label::COUNT {
            return Err(ConfigError::Invalid("unexpected number of labels"));
        }
        if m.window == 0 || self.overlap >= m.window {
            return Err(ConfigError::Invalid("overlap must be smaller than window"));
        }
        if !m.d_model.is_multiple_of(m.n_heads) {
            return Err(ConfigError::Invalid("d_model must be divisible by n_heads"));
        }

        Ok(())
    }
}
