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

use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
            + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2) - 15.0 / (z2 * z2 * z2)).ln()
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
) -> (Array1<f64>, Array1<f64>, f64) {
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
            let mut acc = vec![0.0f64; 2 * n];
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
                    let inv_sd = 1.0 / sd[i];
                    let mi = m[i];
                    let mut s = 0.0f64;
                    let mut sl = 0.0f64;
                    for t in 0..tl {
                        let e = row_g[t] + field[t] - row_s[t];
                        if e > -745.0 {
                            let v = e.exp();
                            s += v;
                            let x = lo + (t0 + t) as f64 * dx;
                            sl += (x - mi) * inv_sd * inv_sd * v;
                        }
                    }
                    acc[i] += wq * s * dx;
                    acc[n + i] += wq * sl * dx;
                }
                t0 += tl;
            }
            acc
        })
        .reduce(
            || vec![0.0f64; 2 * n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    let total: f64 = p[..n].iter().sum();
    let p_norm: Array1<f64> = Array1::from_iter(p[..n].iter().map(|x| x / total));
    let slopes: Array1<f64> = Array1::from_iter(p[n..].iter().cloned());
    (p_norm, slopes, total)
}

/// Min-wins factor-race win probabilities (normalized), raw own-location
/// slopes of the unnormalized map (the inversion preconditioner), and the
/// pre-normalization total. slope_i = d p_raw_i / d mu_i.
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=1501))]
fn forward_and_slopes<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points)
    });
    Ok((p.into_pyarray_bound(py), sl.into_pyarray_bound(py), total))
}

/// Back-compatible forward-only entry point: (normalized p, total).
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
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, _sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points)
    });
    Ok((p.into_pyarray_bound(py), total))
}


#[allow(clippy::too_many_arguments)]
fn jvp_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    h: ArrayView1<f64>,
    points: usize,
    grid_form: bool,
) -> Array1<f64> {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let log_norm: Vec<f64> = sd.iter().map(|s| s.ln() + LN_SQRT_2PI).collect();
    let hvec: Vec<f64> = h.iter().cloned().collect();

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

    let out: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            let mut acc = vec![0.0f64; n];
            let mut logs = vec![0.0f64; n * TILE];
            let mut logg = vec![0.0f64; n * TILE];
            let mut haz = vec![0.0f64; n * TILE];
            let mut field = vec![0.0f64; TILE];
            let mut asum = vec![0.0f64; TILE];   // A = sum_j h_j haz_j
            let mut lsum = vec![0.0f64; TILE];   // Lambda = sum_j haz_j
            let mut t0 = 0;
            while t0 < points {
                let tl = TILE.min(points - t0);
                field[..tl].fill(0.0);
                asum[..tl].fill(0.0);
                lsum[..tl].fill(0.0);
                for i in 0..n {
                    let inv_sd = 1.0 / sd[i];
                    let ln_i = log_norm[i];
                    let mi = m[i];
                    let hi_ = hvec[i];
                    for t in 0..tl {
                        let x = lo + (t0 + t) as f64 * dx;
                        let z = (x - mi) * inv_sd;
                        let ls = log_ndtr(-z);
                        let lg = -0.5 * z * z - ln_i;
                        let hz = (lg - ls).exp();
                        logs[i * TILE + t] = ls;
                        logg[i * TILE + t] = lg;
                        haz[i * TILE + t] = hz;
                        field[t] += ls;
                        asum[t] += hi_ * hz;
                        lsum[t] += hz;
                    }
                }
                for i in 0..n {
                    let hi_ = hvec[i];
                    let mi = m[i];
                    let inv_d = 1.0 / d[i];
                    let mut s = 0.0f64;
                    for t in 0..tl {
                        let e = logg[i * TILE + t] + field[t] - logs[i * TILE + t];
                        if e > -745.0 {
                            let g_r = e.exp();
                            let term = if grid_form {
                                let x = lo + (t0 + t) as f64 * dx;
                                hi_ * (x - mi) * inv_d + asum[t]
                                    - hi_ * haz[i * TILE + t]
                            } else {
                                asum[t] - hi_ * lsum[t]
                            };
                            s += g_r * term;
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
    Array1::from_vec(out)
}

/// Jacobian-vector product of the min-wins map. form "ibp" is the
/// continuum weighted-Laplacian derivative; form "grid" is the exact
/// derivative of the frozen-grid rectangle sum. Mirrors
/// raceutil.jacobian_vector_product.
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, h, points=3001, form="ibp"))]
#[allow(clippy::too_many_arguments)]
fn jacobian_vector_product<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    h: PyReadonlyArray1<f64>,
    points: usize,
    form: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let h_o: Array1<f64> = h.as_array().to_owned();
    let grid = form == "grid";
    let out = py.allow_threads(|| {
        jvp_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                   w_o.view(), h_o.view(), points, grid)
    });
    Ok(out.into_pyarray_bound(py))
}


fn cheb_nodes(a: f64, b: f64, r: usize) -> Vec<f64> {
    (0..r)
        .map(|k| 0.5 * (a + b)
            + 0.5 * (b - a)
                * ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * r) as f64).cos())
        .collect()
}

fn bary_weights(nodes: &[f64]) -> Vec<f64> {
    let r = nodes.len();
    (0..r)
        .map(|j| {
            let mut w = 1.0;
            for k in 0..r {
                if k != j {
                    w /= nodes[j] - nodes[k];
                }
            }
            w
        })
        .collect()
}

/// Barycentric Lagrange interpolation row for query point q.
fn bary_row(nodes: &[f64], wts: &[f64], q: f64) -> Vec<f64> {
    let r = nodes.len();
    for j in 0..r {
        if (q - nodes[j]).abs() < 1e-14 {
            let mut row = vec![0.0; r];
            row[j] = 1.0;
            return row;
        }
    }
    let mut row: Vec<f64> = (0..r).map(|j| wts[j] / (q - nodes[j])).collect();
    let sum: f64 = row.iter().sum();
    for x in row.iter_mut() {
        *x /= sum;
    }
    row
}

#[allow(clippy::too_many_arguments)]
fn separated_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
    rm: usize,
    rs_req: usize,
) -> (Array1<f64>, f64) {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_min = sd.iter().cloned().fold(f64::MAX, f64::min);
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let rs = if sd_max - sd_min < 1e-12 { 1 } else { rs_req };

    let mut m_all = vec![0.0f64; q * n];
    let mut m_lo = f64::MAX;
    let mut m_hi = f64::MIN;
    for qi in 0..q {
        for i in 0..n {
            let mut mi = mu[i];
            for r in 0..v.ncols() {
                mi += v[[i, r]] * f_nodes[[qi, r]];
            }
            m_all[qi * n + i] = mi;
            m_lo = m_lo.min(mi);
            m_hi = m_hi.max(mi);
        }
    }
    let lo = m_lo - 8.0 * sd_max;
    let hi = m_hi + 8.0 * sd_max;
    let dx = (hi - lo) / (points - 1) as f64;

    let mn = cheb_nodes(m_lo, m_hi, rm);
    let sn = if rs == 1 {
        vec![0.5 * (sd_min + sd_max)]
    } else {
        cheb_nodes(sd_min, sd_max, rs)
    };
    let wm = bary_weights(&mn);
    let ws = bary_weights(&sn);
    let r_tot = rm * rs;

    // kernel tables at Chebyshev nodes: (r_tot, points)
    let mut logs_c = vec![0.0f64; r_tot * points];
    let mut haz_c = vec![0.0f64; r_tot * points];
    for cm in 0..rm {
        for cs in 0..rs {
            let c = cm * rs + cs;
            let inv_sd = 1.0 / sn[cs];
            let ln_c = sn[cs].ln() + LN_SQRT_2PI;
            for t in 0..points {
                let x = lo + t as f64 * dx;
                let z = (x - mn[cm]) * inv_sd;
                let ls = log_ndtr(-z);
                logs_c[c * points + t] = ls;
                haz_c[c * points + t] = (-0.5 * z * z - ln_c - ls).exp();
            }
        }
    }
    // per-runner sigma rows (fixed across nodes)
    let ts_rows: Vec<Vec<f64>> = (0..n).map(|i| bary_row(&sn, &ws, sd[i])).collect();

    let p: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            // Tm rows and the aggregation matrix A[cm][cs] = sum_i Tm_i Ts_i
            let tm_rows: Vec<Vec<f64>> =
                (0..n).map(|i| bary_row(&mn, &wm, m[i])).collect();
            let mut amat = vec![0.0f64; r_tot];
            for i in 0..n {
                for (cm, tmv) in tm_rows[i].iter().enumerate() {
                    for (cs, tsv) in ts_rows[i].iter().enumerate() {
                        amat[cm * rs + cs] += tmv * tsv;
                    }
                }
            }
            // field(x) = sum_c amat_c logS_c(x); weights = exp(field) dx
            // b_c = sum_x haz_c(x) * weights(x)
            let mut b = vec![0.0f64; r_tot];
            for t in 0..points {
                let mut field = 0.0;
                for c in 0..r_tot {
                    field += amat[c] * logs_c[c * points + t];
                }
                if field > -745.0 {
                    let wt = field.exp() * dx;
                    for c in 0..r_tot {
                        b[c] += haz_c[c * points + t] * wt;
                    }
                }
            }
            // p_i = sum_c T_i(c) b_c
            let mut acc = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0;
                for (cm, tmv) in tm_rows[i].iter().enumerate() {
                    for (cs, tsv) in ts_rows[i].iter().enumerate() {
                        s += tmv * tsv * b[cm * rs + cs];
                    }
                }
                acc[i] = wq * s;
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
    let out = Array1::from_iter(p.into_iter().map(|x| x / total));
    (out, total)
}

/// Chebyshev-separated forward pass: O(Q r (N + L)) per the exp20
/// prototype, with exponential convergence in (rm, rs). Returns
/// (normalized shares, pre-normalization total).
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=1501, rm=48, rs=14))]
#[allow(clippy::too_many_arguments)]
fn win_probabilities_factor_separated<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
    rm: usize,
    rs: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, total) = py.allow_threads(|| {
        separated_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                         w_o.view(), points, rm, rs)
    });
    Ok((p.into_pyarray_bound(py), total))
}

#[pymodule]
fn fastrace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_and_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_vector_product, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor_separated, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor, m)?)?;
    Ok(())
}
