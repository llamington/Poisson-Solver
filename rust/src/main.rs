use clap::Parser;

/// Program for solving a 3D discrete form of the Poisson Equation
#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Size of one edge of the volume
    #[arg(short = 'n', long = "size", default_value_t = 7)]
    size: u16,

    /// Number of iterations
    #[arg(short, long, default_value_t = 300)]
    interations: u16,

    /// Number of threads
    #[arg(short, long)]
    threads: u8,
}

fn main() {
    let args = Args::parse();
}
