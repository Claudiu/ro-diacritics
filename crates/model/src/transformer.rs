//! Forward pass of the character encoder, parameter names matching
//! `src/diacritics/model/encoder.py` one to one.

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, LayerNormConfig, Linear};
use candle_nn::{ops::softmax_last_dim, VarBuilder};

use diacritics_domain::alphabet::PAD_ID;
use diacritics_domain::normalize::strip_text;
use diacritics_domain::{
    plan_windows, restore, ExportConfig, Label, Prediction, RestoreError, Restorer,
};

/// Longest input accepted by `restore`, in chars; one request can't eat the CPU.
pub const MAX_CHARS: usize = 20_000;

/// Values per fp16 scale in a 4-bit matrix; see `src/diacritics/export/safetensors.py`.
const GROUP: usize = 32;
const SCALE: &str = ".scale";

/// Added to attention scores of padding keys: effectively minus infinity after softmax.
const PAD_PENALTY: f64 = -1e9;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Config(#[from] diacritics_domain::config::ConfigError),

    #[error("config.json: {0}")]
    ConfigJson(#[from] serde_json::Error),

    #[error("weights: {0}")]
    Weights(#[from] candle_core::Error),
}

struct Attention {
    qkv: Linear,
    out: Linear,
    n_heads: usize,
}

impl Attention {
    /// `mask` is `(batch, 1, 1, t)`: 0 for keys that may be attended to, very negative
    /// for padding.
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let head = d / self.n_heads;
        let qkv = self.qkv.forward(x)?;
        let q = qkv.narrow(D::Minus1, 0, d)?;
        let k = qkv.narrow(D::Minus1, d, d)?;
        let v = qkv.narrow(D::Minus1, 2 * d, d)?;
        let split = |t_: Tensor| {
            t_.reshape((b, t, self.n_heads, head))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (split(q)?, split(k)?, split(v)?);

        let scale = 1.0 / (head as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let weights = softmax_last_dim(&scores.broadcast_add(mask)?)?;
        let y = weights.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;

        self.out.forward(&y)
    }
}

struct Block {
    ln1: LayerNorm,
    attn: Attention,
    ln2: LayerNorm,
    up: Linear,
    down: Linear,
}

impl Block {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let x = (x + self.attn.forward(&self.ln1.forward(x)?, mask)?)?;
        let h = self
            .down
            .forward(&self.up.forward(&self.ln2.forward(&x)?)?.gelu_erf()?)?;

        x + h
    }
}

pub struct Transformer {
    config: ExportConfig,
    device: Device,
    embed: Embedding,
    pos: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    head: Linear,
}

impl Transformer {
    /// Load the PyTorch export: `config.json` text and `model.safetensors` bytes.
    pub fn from_export(
        config_json: &str,
        weights: Vec<u8>,
        device: Device,
    ) -> Result<Self, ModelError> {
        let config: ExportConfig = serde_json::from_str(config_json)?;
        config.validate()?;
        let vb = VarBuilder::from_tensors(dequantize_all(&weights)?, DType::F32, &device);

        Self::build(config, vb, device)
    }

    /// Zero weights with the given config; for benchmarks and shape tests.
    pub fn zeros(config: ExportConfig, device: Device) -> Result<Self, ModelError> {
        config.validate()?;
        let vb = VarBuilder::zeros(DType::F32, &device);

        Self::build(config, vb, device)
    }

    fn build(config: ExportConfig, vb: VarBuilder, device: Device) -> Result<Self, ModelError> {
        let m = &config.model;
        let ln = LayerNormConfig::default();
        let blocks = (0..m.n_layers)
            .map(|i| {
                let vb = vb.pp(format!("blocks.{i}"));

                Ok(Block {
                    ln1: layer_norm(m.d_model, ln, vb.pp("ln1"))?,
                    attn: Attention {
                        qkv: linear(m.d_model, 3 * m.d_model, vb.pp("attn.qkv"))?,
                        out: linear(m.d_model, m.d_model, vb.pp("attn.out"))?,
                        n_heads: m.n_heads,
                    },
                    ln2: layer_norm(m.d_model, ln, vb.pp("ln2"))?,
                    up: linear(m.d_model, m.d_ff, vb.pp("ff.up"))?,
                    down: linear(m.d_ff, m.d_model, vb.pp("ff.down"))?,
                })
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self {
            embed: embedding(m.vocab_size, m.d_model, vb.pp("embed"))?,
            pos: embedding(m.window, m.d_model, vb.pp("pos"))?,
            blocks,
            ln_f: layer_norm(m.d_model, ln, vb.pp("ln_f"))?,
            head: linear(m.d_model, m.n_classes, vb.pp("head"))?,
            config,
            device,
        })
    }

    pub fn config(&self) -> &ExportConfig {
        &self.config
    }

    /// `(batch, t)` ids → `(batch, t, n_classes)` class probabilities. `PAD_ID` positions
    /// are masked out as keys, so padding never changes the output of real positions.
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let mask = ids
            .eq(PAD_ID)?
            .to_dtype(DType::F32)?
            .affine(PAD_PENALTY, 0.0)?
            .reshape((b, 1, 1, t))?;
        let positions = Tensor::arange(0u32, t as u32, &self.device)?;
        let mut x = self
            .embed
            .forward(ids)?
            .broadcast_add(&self.pos.forward(&positions)?)?;
        for block in &self.blocks {
            x = block.forward(&x, &mask)?;
        }
        let logits = self.head.forward(&self.ln_f.forward(&x)?)?;

        softmax_last_dim(&logits)
    }

    /// One prediction per char of diacritic-free `stripped`, windowed like the Python
    /// `Predictor` so both produce the same output.
    pub fn predict(&self, stripped: &str) -> Result<Vec<Prediction>, RestoreError> {
        let ids = self.config.alphabet.encode(stripped);
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let window = self.config.model.window;
        let windows = plan_windows(ids.len(), window, self.config.overlap);
        let width = ids.len().min(window);
        let mut batch = vec![PAD_ID; windows.len() * width];
        for (i, w) in windows.iter().enumerate() {
            batch[i * width..i * width + (w.end - w.start)].copy_from_slice(&ids[w.start..w.end]);
        }

        let input = Tensor::from_vec(batch, (windows.len(), width), &self.device)
            .map_err(|e| RestoreError::Inference(e.to_string()))?;
        let probs: Vec<Vec<Vec<f32>>> = self
            .forward(&input)
            .and_then(|p| p.to_vec3())
            .map_err(|e| RestoreError::Inference(e.to_string()))?;

        let mut out = Vec::with_capacity(ids.len());
        for (i, w) in windows.iter().enumerate() {
            for row in &probs[i][w.keep_from - w.start..w.keep_to - w.start] {
                let (index, confidence) =
                    row.iter().enumerate().fold((0, f32::MIN), |best, (j, &p)| {
                        if p > best.1 {
                            (j, p)
                        } else {
                            best
                        }
                    });
                let label =
                    Label::from_index(index).expect("n_classes validated against Label::COUNT");
                out.push(Prediction { label, confidence });
            }
        }

        Ok(out)
    }
}

impl Restorer for Transformer {
    /// Existing diacritics are stripped first, so the model decides every letter.
    fn restore(&self, text: &str) -> Result<String, RestoreError> {
        let count = text.chars().count();
        if count > MAX_CHARS {
            return Err(RestoreError::TooLong(count, MAX_CHARS));
        }

        let stripped = strip_text(text);
        let predictions = self.predict(&stripped)?;

        Ok(restore(&stripped, &predictions, self.config.threshold))
    }
}

/// f32 weights from the export: 4-bit matrices unpacked, fp16 vectors upcast.
fn dequantize_all(weights: &[u8]) -> candle_core::Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load_buffer(weights, &Device::Cpu)?;
    let mut out = HashMap::new();
    for (name, t) in &tensors {
        if name.ends_with(SCALE) {
            continue;
        }
        let t = if t.dtype() == DType::U8 {
            let scale = tensors
                .get(&format!("{name}{SCALE}"))
                .ok_or_else(|| candle_core::Error::Msg(format!("{name}: missing {SCALE}")))?;
            dequantize(t, scale)?
        } else {
            t.to_dtype(DType::F32)?
        };
        out.insert(name.clone(), t);
    }

    Ok(out)
}

/// `(rows, cols/2)` packed nibbles (even element low, stored as q + 8) → `(rows, cols)` f32.
fn dequantize(packed: &Tensor, scale: &Tensor) -> candle_core::Result<Tensor> {
    let (rows, half) = packed.dims2()?;
    let bytes = packed.flatten_all()?.to_vec1::<u8>()?;
    let scales = scale.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    if scales.len() * GROUP != bytes.len() * 2 {
        return Err(candle_core::Error::Msg(
            "scale count does not match weights".into(),
        ));
    }
    let values = bytes
        .iter()
        .flat_map(|b| [b & 15, b >> 4])
        .enumerate()
        .map(|(i, q)| (f32::from(q) - 8.0) * scales[i / GROUP])
        .collect();

    Tensor::from_vec(values, (rows, 2 * half), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::path::PathBuf;

    #[derive(Deserialize)]
    struct Case {
        text: String,
        predictions: Vec<(usize, f32)>,
        restored: String,
    }

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/fixtures/parity")
    }

    /// `uv run diacritics fixture test/fixtures/parity` writes a tiny random-weight
    /// export plus PyTorch's outputs; the forward pass here must reproduce them.
    #[test]
    fn matches_pytorch_on_the_parity_fixture() {
        let dir = fixture_dir();
        let config =
            std::fs::read_to_string(dir.join("config.json")).expect("run scripts/parity.sh first");
        let weights = std::fs::read(dir.join("model.safetensors")).unwrap();
        let cases: Vec<Case> =
            serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).unwrap())
                .unwrap();
        let model = Transformer::from_export(&config, weights, Device::Cpu).unwrap();

        for case in cases {
            let predictions = model.predict(&case.text).unwrap();
            assert_eq!(predictions.len(), case.predictions.len(), "{:?}", case.text);
            for (got, &(label, confidence)) in predictions.iter().zip(&case.predictions) {
                assert_eq!(got.label as usize, label, "{:?}", case.text);
                assert!(
                    (got.confidence - confidence).abs() < 1e-4,
                    "{:?}: {} vs {}",
                    case.text,
                    got.confidence,
                    confidence
                );
            }
            assert_eq!(model.restore(&case.text).unwrap(), case.restored);
        }
    }

    #[test]
    fn rejects_oversized_input() {
        let config = std::fs::read_to_string(fixture_dir().join("config.json"))
            .expect("run scripts/parity.sh first");
        let model =
            Transformer::zeros(serde_json::from_str(&config).unwrap(), Device::Cpu).unwrap();
        let text = "a".repeat(MAX_CHARS + 1);

        assert!(matches!(
            model.restore(&text),
            Err(RestoreError::TooLong(..))
        ));
        assert_eq!(model.restore("").unwrap(), "");
    }
}
