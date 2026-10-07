//! `new Diacritics(configJson, weightsBytes).restore(text)` from JavaScript.

use candle_core::Device;
use diacritics_domain::Restorer;
use diacritics_model::Transformer;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Diacritics {
    model: Transformer,
}

#[wasm_bindgen]
impl Diacritics {
    /// Load the export produced by `diacritics train`: `config.json` text and the bytes
    /// of `model.safetensors`.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str, weights: Vec<u8>) -> Result<Diacritics, JsError> {
        let model = Transformer::from_export(config_json, weights, Device::Cpu)
            .map_err(|e| JsError::new(&e.to_string()))?;

        Ok(Self { model })
    }

    /// Restore diacritics; existing ones are stripped first so the model decides every letter.
    pub fn restore(&self, text: &str) -> Result<String, JsError> {
        self.model
            .restore(text)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Longest accepted input, in characters.
    pub fn max_chars(&self) -> usize {
        diacritics_model::MAX_CHARS
    }
}
