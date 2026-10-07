/// What to do with one character of diacritic-free input.
///
/// The base letter is known from the input, so one class covers several letters:
/// `BreveComma` turns a→ă, s→ș, t→ț; `Circumflex` turns a→â, i→î.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Label {
    None = 0,
    BreveComma = 1,
    Circumflex = 2,
}

impl Label {
    pub const COUNT: usize = 3;

    /// Class index as the model and `config.json` order them.
    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::None),
            1 => Some(Self::BreveComma),
            2 => Some(Self::Circumflex),
            _ => None,
        }
    }
}
