use crate::util::tensor_idx;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp;

pub struct PoissonSolver<'a> {
    size: usize,
    iterations: u16,
    source: &'a Vec<f32>,
    threads: usize,
    delta: f32,
    curr: Box<Vec<f32>>,
    next: Box<Vec<f32>>,
    slice_width: usize,
}

impl<'a> PoissonSolver<'a> {
    pub fn new(
        size: usize,
        source: &'a Vec<f32>,
        iterations: u16,
        threads: usize,
        delta: f32,
    ) -> Self {
        PoissonSolver {
            size,
            iterations,
            source,
            threads,
            delta,
            curr: Box::new(vec![0.0; size * size * size]),
            next: Box::new(vec![0.0; size * size * size]),
            slice_width: (size + threads - 1) / threads,
        }
    }
    pub fn solve(&mut self) -> &Box<Vec<f32>> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .unwrap();
        thread_pool.install(|| {
            for _iter in 0..self.iterations {
                let chunked_next = self
                    .next
                    .par_chunks_mut(self.slice_width * self.size * self.size);

                chunked_next.enumerate().for_each(|(job_id, next_chunk)| {
                    let start_i = job_id * self.slice_width;
                    let end_i = cmp::min((job_id + 1) * self.slice_width, self.size);
                    for i in start_i..end_i {
                        for j in 0..self.size {
                            for k in 0..self.size {
                                let mut v: f32 = 0.0;

                                if i == 0 {
                                    v += 2.0 * self.curr[tensor_idx!(1, j, k, self.size)];
                                } else if i == self.size - 1 {
                                    v += 2.0
                                        * self.curr[tensor_idx!(self.size - 2, j, k, self.size)];
                                } else {
                                    v += self.curr[tensor_idx!(i - 1, j, k, self.size)];
                                    v += self.curr[tensor_idx!(i + 1, j, k, self.size)];
                                }

                                if j == 0 {
                                    v += 2.0 * self.curr[tensor_idx!(i, 1, k, self.size)];
                                } else if j == self.size - 1 {
                                    v += 2.0
                                        * self.curr[tensor_idx!(i, self.size - 2, k, self.size)];
                                } else {
                                    v += self.curr[tensor_idx!(i, j - 1, k, self.size)];
                                    v += self.curr[tensor_idx!(i, j + 1, k, self.size)];
                                }

                                if k == 0 {
                                    v += 2.0 * self.curr[tensor_idx!(i, j, 1, self.size)];
                                } else if k == self.size - 1 {
                                    v += 2.0
                                        * self.curr[tensor_idx!(i, j, self.size - 2, self.size)];
                                } else {
                                    v += self.curr[tensor_idx!(i, j, k - 1, self.size)];
                                    v += self.curr[tensor_idx!(i, j, k + 1, self.size)];
                                }

                                v -= self.delta
                                    * self.delta
                                    * self.source[tensor_idx!(i, j, k, self.size)];
                                v /= 6.0;
                                next_chunk[tensor_idx!((i - start_i), j, k, self.size)] = v;
                            }
                        }
                    }
                });

                std::mem::swap(&mut self.curr, &mut self.next);
            }
        });

        return &self.curr;
    }
}
