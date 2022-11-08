macro_rules! tensor_idx {
    ($i:expr, $j:expr, $k:expr, $n:expr) => {{
        $n * ($n * $i + $j) + $k
    }};
}
pub(crate) use tensor_idx;
