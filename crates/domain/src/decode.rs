//! Turn per-character predictions back into Romanian text.
//! Mirrors `src/diacritics/domain/decode.py`.

use crate::label::Label;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub label: Label,
    pub confidence: f32,
}

/// The diacritic for a (base letter, label) pair, or `None` when the pair is invalid.
fn apply(ch: char, label: Label) -> Option<char> {
    match (ch, label) {
        ('a', Label::BreveComma) => Some('ă'),
        ('s', Label::BreveComma) => Some('ș'),
        ('t', Label::BreveComma) => Some('ț'),
        ('a', Label::Circumflex) => Some('â'),
        ('i', Label::Circumflex) => Some('î'),
        ('A', Label::BreveComma) => Some('Ă'),
        ('S', Label::BreveComma) => Some('Ș'),
        ('T', Label::BreveComma) => Some('Ț'),
        ('A', Label::Circumflex) => Some('Â'),
        ('I', Label::Circumflex) => Some('Î'),
        _ => None,
    }
}

/// Apply `predictions` to diacritic-free `text`, one prediction per `char`.
///
/// A diacritic is added only when the prediction is valid for that letter and at least
/// `threshold` confident; a missed diacritic reads better than a wrong one.
///
/// # Panics
/// When `predictions.len()` differs from the number of chars in `text`; callers produce
/// one prediction per char by construction.
pub fn restore(text: &str, predictions: &[Prediction], threshold: f32) -> String {
    let count = text.chars().count();
    assert_eq!(predictions.len(), count, "one prediction per character");

    text.chars()
        .zip(predictions)
        .map(|(ch, p)| {
            if p.confidence < threshold {
                return ch;
            }

            apply(ch, p.label).unwrap_or(ch)
        })
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
        predictions: Vec<(usize, f32)>,
        threshold: f32,
        restored: String,
    }

    #[test]
    fn decode_fixture() {
        for case in fixtures::jsonl::<Case>("decode.jsonl") {
            let predictions: Vec<Prediction> = case
                .predictions
                .iter()
                .map(|&(label, confidence)| Prediction {
                    label: Label::from_index(label).expect("label"),
                    confidence,
                })
                .collect();

            assert_eq!(
                restore(&case.text, &predictions, case.threshold),
                case.restored,
                "{:?}",
                case.text
            );
        }
    }
}
