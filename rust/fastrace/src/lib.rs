//! fastrace: fused kernel for the factor-probit share transform.
//!
//! Computes the min-wins factor race of the `thurstone`/`raceutil` line:
//!   p_i = E_f [ integral g_i(x|f) * prod_{j!=i} S_j(x|f) dx ],
//! with S the Gaussian survival, by the shared-survival-field identity
//! (build the field product once per node, divide one out per alternative).
//!
//! Design notes.
//! - Parallel over factor nodes (rayon): nodes are independent.
//! - x-tiled: per node, lattice columns are processed in tiles so the
//!   per-tile logS/logg blocks (N x TILE) stay in cache; two passes per
//!   tile (accumulate field, then distribute) instead of materializing
//!   N x L temporaries as NumPy must.
//! - log-domain throughout: log_ndtr directly (never 1 - Phi), analytic
//!   log-density. Mirrors the tail-stability fix in the Python reference.
//! - Not yet implemented (stacks on top): the FFT cross-correlation form
//!   of the per-node location->probability curves, which would let
//!   calibration Newton steps sample frozen curves by interpolation
//!   instead of re-integrating (the `winning` interpolation trick,
//!   factor generalization).

use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const TILE: usize = 256;
const LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;

/// log(Phi(z)). libm::erfc is exact to double precision until it
/// underflows (z < about -37.5); beyond that use the asymptotic series
/// log Phi(z) = -z^2/2 - log(-z) - log sqrt(2 pi) + log(1 - 1/z^2 + 3/z^4 - 15/z^6).
fn log_ndtr(z: f64) -> f64 {
    if z > 6.0 {
        -0.5 * libm::erfc(z * std::f64::consts::FRAC_1_SQRT_2)
    } else if z > -37.0 {
        (0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)).ln()
    } else {
        let z2 = z * z;
        -0.5 * z2 - LN_SQRT_2PI - (-z).ln()
            + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2) - 15.0 / (z2 * z2 * z2)).ln_1p_free()
    }
}

trait Ln1pFree {
    fn ln_1p_free(self) -> f64;
}
impl Ln1pFree for f64 {
    #[inline]
    fn ln_1p_free(self) -> f64 {
        self.ln()
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
) -> (Array1<f64>, f64) {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let log_norm: Vec<f64> = sd.iter().map(|s| s.ln() + LN_SQRT_2PI).collect();

    // conditional locations m[qi*n + i] and the global interval
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    let mut m_all = vec![0.0f64; q * n];
    for qi in 0..q {
        for i in 0..n {
            let mut mi = mu[i];
            for r in 0..v.ncols() {
                mi += v[[i, r]] * f_nodes[[qi, r]];
            }
            m_all[qi * n + i] = mi;
            lo = lo.min(mi);
            hi = hi.max(mi);
        }
    }
    lo -= 8.0 * sd_max;
    hi += 8.0 * sd_max;
    let dx = (hi - lo) / (points - 1) as f64;

    let p: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            let mut acc = vec![0.0f64; n];
            let mut logs = vec![0.0f64; n * TILE];
            let mut logg = vec![0.0f64; n * TILE];
            let mut field = vec![0.0f64; TILE];
            let mut t0 = 0;
            while t0 < points {
                let tl = TILE.min(points - t0);
                field[..tl].fill(0.0);
                for i in 0..n {
                    let inv_sd = 1.0 / sd[i];
                    let ln_i = log_norm[i];
                    let mi = m[i];
                    let row_s = &mut logs[i * TILE..i * TILE + tl];
                    let row_g = &mut logg[i * TILE..i * TILE + tl];
                    for t in 0..tl {
                        let x = lo + (t0 + t) as f64 * dx;
                        let z = (x - mi) * inv_sd;
                        let ls = log_ndtr(-z);
                        row_s[t] = ls;
                        row_g[t] = -0.5 * z * z - ln_i;
                        field[t] += ls;
                    }
                }
                for i in 0..n {
                    let row_s = &logs[i * TILE..i * TILE + tl];
                    let row_g = &logg[i * TILE..i * TILE + tl];
                    let mut s = 0.0f64;
                    for t in 0..tl {
                        let e = row_g[t] + field[t] - row_s[t];
                        if e > -745.0 {
                            s += e.exp();
                        }
                    }
                    acc[i] += wq * s * dx;
                }
                t0 += tl;
            }
            acc
        })
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    let total: f64 = p.iter().sum();
    let out: Array1<f64> = Array1::from_iter(p.into_iter().map(|x| x / total));
    (out, total)
}

/// Min-wins factor-race win probabilities plus the pre-normalization total.
/// Mirrors raceutil.win_probabilities_factor(mu, V, D, F, W, points,
/// return_total=True).
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=1501))]
fn win_probabilities_factor<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let (p, total) = py.allow_threads(|| {
        forward_kernel(
            mu.as_array(),
            v.as_array(),
            d.as_array(),
            f.as_array(),
            w.as_array(),
            points,
        )
    });
    Ok((p.into_pyarray_bound(py), total))
}

#[pymodule]
fn fastrace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(win_probabilities_factor, m)?)?;
    Ok(())
}
