//! Character-level text rules shared with `src/diacritics/domain/normalize.py`.
//!
//! Every function maps one `char` to one `char`, so input, labels and output always
//! have the same length.

use crate::label::Label;

/// Legacy cedilla forms, still common in older texts, mapped to comma-below ones.
pub fn fix_cedilla(ch: char) -> char {
    match ch {
        '\u{015f}' => '\u{0219}',
        '\u{0163}' => '\u{021b}',
        '\u{015e}' => '\u{0218}',
        '\u{0162}' => '\u{021a}',
        other => other,
    }
}

/// The Romanian diacritic removed, case kept.
pub fn strip_diacritic(ch: char) -> char {
    match ch {
        'ă' | 'â' => 'a',
        'î' => 'i',
        'ș' => 's',
        'ț' => 't',
        'Ă' | 'Â' => 'A',
        'Î' => 'I',
        'Ș' => 'S',
        'Ț' => 'T',
        other => other,
    }
}

/// Whether a diacritic-free character could take a diacritic.
pub fn is_candidate(ch: char) -> bool {
    matches!(ch, 'a' | 'i' | 's' | 't' | 'A' | 'I' | 'S' | 'T')
}

/// The label a character of correct Romanian text carries.
pub fn label_of(ch: char) -> Label {
    match ch {
        'ă' | 'ș' | 'ț' | 'Ă' | 'Ș' | 'Ț' => Label::BreveComma,
        'â' | 'î' | 'Â' | 'Î' => Label::Circumflex,
        _ => Label::None,
    }
}

/// Cedilla forms fixed and diacritics removed, as the model expects its input.
pub fn strip_text(text: &str) -> String {
    text.chars()
        .map(|ch| strip_diacritic(fix_cedilla(ch)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Case {
        text: String,
        stripped: String,
        labels: Vec<u8>,
        candidates: Vec<bool>,
    }

    #[test]
    fn normalize_fixture() {
        for case in fixtures::jsonl::<Case>("normalize.jsonl") {
            let stripped = strip_text(&case.text);
            assert_eq!(stripped, case.stripped, "{:?}", case.text);

            let labels: Vec<u8> = case
                .text
                .chars()
                .map(|ch| label_of(fix_cedilla(ch)) as u8)
                .collect();
            assert_eq!(labels, case.labels, "{:?}", case.text);

            let candidates: Vec<bool> = stripped.chars().map(is_candidate).collect();
            assert_eq!(candidates, case.candidates, "{:?}", case.text);
        }
    }
}
