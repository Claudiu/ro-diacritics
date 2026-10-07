//! `diacritics restore --model artifacts/export < input.txt`

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use candle_core::Device;
use clap::{Parser, Subcommand};
use diacritics_domain::Restorer;
use diacritics_model::Transformer;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Restore diacritics line by line from stdin to stdout.
    Restore {
        /// Directory holding config.json and model.safetensors.
        #[arg(long, default_value = "artifacts/export")]
        model: PathBuf,
    },
}

fn load(dir: &Path) -> anyhow::Result<Transformer> {
    let config = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("read {}/config.json", dir.display()))?;
    let weights = std::fs::read(dir.join("model.safetensors"))
        .with_context(|| format!("read {}/model.safetensors", dir.display()))?;

    Transformer::from_export(&config, weights, Device::Cpu).context("load model")
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Restore { model } => {
            let model = load(&model)?;
            let stdin = io::stdin().lock();
            let mut stdout = io::stdout().lock();
            for line in stdin.lines() {
                let line = line.context("read stdin")?;
                writeln!(stdout, "{}", model.restore(&line)?)?;
            }
        }
    }

    Ok(())
}
