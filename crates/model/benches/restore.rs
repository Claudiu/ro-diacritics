//! Native inference speed on the production model shape (zero weights, same compute).

use candle_core::Device;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use diacritics_domain::Restorer;
use diacritics_model::Transformer;

const CONFIG: &str = r#"{
  "format_version": 1,
  "model": {"vocab_size": 256, "window": 256, "d_model": 192, "n_layers": 4, "n_heads": 4, "d_ff": 512, "dropout": 0.1, "n_classes": 3},
  "alphabet": {"pad_id": 0, "unk_id": 1, "chars": "ALPHABET"},
  "labels": ["none", "breve_comma", "circumflex"],
  "overlap": 32,
  "threshold": 0.5
}"#;

fn production_config() -> String {
    let chars: String = (0..256u32)
        .filter_map(|i| char::from_u32(0x20 + i))
        .filter(|c| *c != '"' && *c != '\\')
        .take(254)
        .collect();

    CONFIG.replace("ALPHABET", &chars)
}

fn bench_restore(c: &mut Criterion) {
    let config: diacritics_domain::ExportConfig =
        serde_json::from_str(&production_config()).unwrap();
    let model = Transformer::zeros(config, Device::Cpu).unwrap();
    let sentence = "Langa casa mea e casa ta si e o casa foarte frumoasa, asa ca stam acasa. ";
    let paragraph = sentence.repeat(12);

    let mut group = c.benchmark_group("restore");
    for (name, text) in [("sentence", sentence.to_owned()), ("paragraph", paragraph)] {
        group.throughput(Throughput::Elements(text.chars().count() as u64));
        group.bench_function(name, |b| b.iter(|| model.restore(&text).unwrap()));
    }
    group.finish();
}

criterion_group!(benches, bench_restore);
criterion_main!(benches);
