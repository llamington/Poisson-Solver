use std::process::ExitCode;

use clap::Parser;

// mod util;
mod util;

/// Program for solving a 3D discrete form of the Poisson Equation
#[derive(Parser, Debug)]
struct Args {
    /// Size of one edge of the volume
    #[arg(short = 'n', long = "size", default_value_t = 7)]
    size: usize,

    /// Number of iterations
    #[arg(short, long, default_value_t = 300)]
    iterations: u16,

    /// Number of threads
    #[arg(short, long)]
    threads: Option<u8>,
}

fn main() -> ExitCode {
    let Args {
        size,
        iterations,
        threads,
    } = Args::parse();

    if size % 2 == 0 {
        eprintln!("Expected an odd size");
        return ExitCode::FAILURE;
    }
    let mut source: Vec<f32> = vec![0.0; size * size * size];
    source[util::tensor_idx!(size / 2, size / 2, size / 2, size)] = 1.0;
    return ExitCode::SUCCESS;
}
