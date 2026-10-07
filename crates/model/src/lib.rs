//! The candle adapter for [`diacritics_domain::Restorer`].

mod transformer;

pub use transformer::{ModelError, Transformer, MAX_CHARS};
