pub struct PoissonSolver<'a> {
    size: usize,
    iterations: u16,
    source: &'a Vec<f32>,
    threads: u8,
    delta: f32,
    curr: *mut Vec<f32>,
    next: *mut Vec<f32>,
}

impl<'a> PoissonSolver<'a> {
    pub fn new(
        size: usize,
        source: &'a Vec<f32>,
        iterations: u16,
        threads: u8,
        delta: f32,
    ) -> Self {
        PoissonSolver {
            size,
            iterations,
            source,
            threads,
            delta,
            curr: &mut vec![0.0; size * size * size],
            next: &mut vec![0.0; size * size * size],
        }
    }
    pub fn solve() -> *mut Vec<f32> {}
}
