//! Driver-free core: everything the model needs around it and nothing it runs on.
//!
//! `normalize`, `decode` and `windows` mirror `src/diacritics/domain/` in Python; the
//! fixtures under `test/fixtures/` are the shared contract.

pub mod alphabet;
pub mod config;
pub mod decode;
pub mod label;
pub mod normalize;
pub mod restorer;
pub mod windows;

pub use alphabet::Alphabet;
pub use config::ExportConfig;
pub use decode::{restore, Prediction};
pub use label::Label;
pub use restorer::{RestoreError, Restorer};
pub use windows::{plan_windows, Window};

#[cfg(test)]
pub(crate) mod fixtures {
    use std::path::PathBuf;

    pub fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/fixtures")
    }

    pub fn jsonl<T: serde::de::DeserializeOwned>(name: &str) -> Vec<T> {
        let text = std::fs::read_to_string(dir().join(name)).expect("fixture file");

        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("fixture line"))
            .collect()
    }
}
