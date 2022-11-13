use clap::Parser;
use std::process::ExitCode;
use util::tensor_idx;
mod poisson_solver;
mod util;

const DELTA: f32 = 1.0;

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
    threads: Option<usize>,
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
    let threads = match threads {
        None => {
            std::thread::available_parallelism()
                .expect("Parallelism data unavailable. Please manually input a thread count")
                .get()
                - 3
        }
        Some(t) => t,
    };
    // Initialise volume
    let mut source: Vec<f32> = vec![0.0; size * size * size];
    source[util::tensor_idx!(size / 2, size / 2, size / 2, size)] = 1.0;

    let mut poisson_solver =
        poisson_solver::PoissonSolver::new(size, &source, iterations, threads, DELTA);
    let solution = poisson_solver.solve();

    // Print the central slice of the solution
    for j in 0..size {
        for k in 0..size {
            print!("{:.5} ", solution[tensor_idx!(size / 2, j, k, size)])
        }
        println!();
    }

    return ExitCode::SUCCESS;
}
