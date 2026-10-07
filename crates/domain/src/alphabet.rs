//! Character vocabulary, read from `config.json` as written by `Alphabet.to_dict()`.

use std::collections::HashMap;

use serde::Deserialize;

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
const RESERVED: u32 = 2;

#[derive(Debug, Clone, Deserialize)]
#[serde(from = "AlphabetJson")]
pub struct Alphabet {
    index: HashMap<char, u32>,
    size: usize,
}

#[derive(Deserialize)]
struct AlphabetJson {
    chars: String,
}

impl From<AlphabetJson> for Alphabet {
    fn from(json: AlphabetJson) -> Self {
        Self::new(json.chars.chars())
    }
}

impl Alphabet {
    pub fn new(chars: impl Iterator<Item = char>) -> Self {
        let index: HashMap<char, u32> = chars
            .enumerate()
            .map(|(i, ch)| (ch, i as u32 + RESERVED))
            .collect();
        let size = index.len() + RESERVED as usize;

        Self { index, size }
    }

    /// Number of ids, including pad and unknown.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| self.index.get(&ch).copied().unwrap_or(UNK_ID))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_known_and_unknown() {
        let alphabet = Alphabet::new(" a".chars());

        assert_eq!(alphabet.size(), 4);
        assert_eq!(alphabet.encode("a z"), vec![3, 2, UNK_ID]);
    }

    #[test]
    fn reads_python_layout() {
        let alphabet: Alphabet =
            serde_json::from_str(r#"{"pad_id": 0, "unk_id": 1, "chars": "ab"}"#).unwrap();

        assert_eq!(alphabet.encode("ba"), vec![3, 2]);
    }
}
