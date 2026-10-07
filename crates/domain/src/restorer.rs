//! Port: anything that restores diacritics in text.

#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    #[error("input of {0} characters exceeds the limit of {1}")]
    TooLong(usize, usize),

    #[error("inference failed: {0}")]
    Inference(String),
}

/// Restores diacritics in diacritic-free (or partially marked) Romanian text.
/// Implementations must return exactly as many chars as they were given.
pub trait Restorer {
    fn restore(&self, text: &str) -> Result<String, RestoreError>;
}

/// Leaves text unchanged: the fake for tests and the floor for any comparison.
pub struct Identity;

impl Restorer for Identity {
    fn restore(&self, text: &str) -> Result<String, RestoreError> {
        Ok(text.to_owned())
    }
}
