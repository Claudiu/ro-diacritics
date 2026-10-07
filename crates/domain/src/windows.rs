//! Split a long sequence into fixed windows with overlap and decide which window owns
//! each position's prediction. Mirrors `src/diacritics/domain/windows.py`.

/// A slice `start..end` of the input. Predictions for `keep_from..keep_to` (absolute
/// positions) come from this window; the rest is context shared with its neighbours.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub start: usize,
    pub end: usize,
    pub keep_from: usize,
    pub keep_to: usize,
}

/// Cover `length` positions with windows of size `window`, consecutive ones sharing
/// `overlap` positions. The owned ranges partition the input: a position belongs to the
/// window where it sits furthest from an edge, so the model sees context on both sides.
///
/// # Panics
/// When `window == 0` or `overlap >= window`; both come from a validated config.
pub fn plan_windows(length: usize, window: usize, overlap: usize) -> Vec<Window> {
    assert!(window > 0, "window must be positive");
    assert!(overlap < window, "overlap must be smaller than window");
    if length == 0 {
        return Vec::new();
    }
    if length <= window {
        return vec![Window {
            start: 0,
            end: length,
            keep_from: 0,
            keep_to: length,
        }];
    }

    let stride = window - overlap;
    let mut starts: Vec<usize> = (0..length - window).step_by(stride).collect();
    starts.push(length - window);

    let half = overlap / 2;
    let mut owners = Vec::with_capacity(starts.len() + 1);
    owners.push(0);
    owners.extend(starts[1..].iter().map(|start| start + half));
    owners.push(length);

    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| Window {
            start,
            end: start + window,
            keep_from: owners[i],
            keep_to: owners[i + 1],
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
        length: usize,
        window: usize,
        overlap: usize,
        windows: Vec<(usize, usize, usize, usize)>,
    }

    #[test]
    fn windows_fixture() {
        for case in fixtures::jsonl::<Case>("windows.jsonl") {
            let expected: Vec<Window> = case
                .windows
                .iter()
                .map(|&(start, end, keep_from, keep_to)| Window {
                    start,
                    end,
                    keep_from,
                    keep_to,
                })
                .collect();

            assert_eq!(
                plan_windows(case.length, case.window, case.overlap),
                expected,
                "{}/{}/{}",
                case.length,
                case.window,
                case.overlap
            );
        }
    }

    #[test]
    fn owned_ranges_partition_the_input() {
        for &(window, overlap) in &[(8, 0), (8, 2), (8, 5), (16, 4), (3, 1)] {
            for length in 0..60 {
                let owned: Vec<usize> = plan_windows(length, window, overlap)
                    .iter()
                    .flat_map(|w| w.keep_from..w.keep_to)
                    .collect();

                assert_eq!(owned, (0..length).collect::<Vec<_>>());
            }
        }
    }
}
