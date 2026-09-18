# The mutation-selection operator: structure, spectrum and algorithmic consequences

Structural statements about the operator are numbered **S1** to **S12**, each with the test or
record that checks it (section 10).

---

## 1. The object

Genotypes are length-`L` binary strings, `s` in `{0,1}^L`, indexed as in
`quasarstack/io/conventions.py`: site `i` is bit `i` is qubit `i`, little-endian throughout.
A landscape assigns each genotype a Malthusian fitness `f(s)`. Mutation flips each site
independently at rate `mu`.

The Crow-Kimura (parallel mutation-selection) dynamics on the vector of genotype frequencies
`x` is

    dx/dt  =  W x  -  <f> x,        W  =  diag(f)  +  mu * sum_i (X_i - I)

where `X_i` is the bit flip on site `i` and `<f> = f . x` keeps the total normalised. The
nonlinear term is a scalar multiple of `x`, so it changes the norm and nothing else. The
equilibrium is the dominant eigenvector of the linear operator `W`, and the equilibrium mean
fitness is its eigenvalue.

Two normalisations coexist. Biological distributions satisfy `sum_s p(s) = 1` (L1); quantum
states satisfy `sum_s |psi(s)|^2 = 1` (L2). `conventions.py` holds both and the conversion,
and section 8 describes the effect of measurement.

---

## 2. It is not a Markov generator

**S1. The columns of `W` sum to `f(s)`, not to zero.**

The mutation part contributes `L` off-diagonal entries of `mu` to each column and a diagonal
`-mu L`, summing to zero. The selection part adds `f(s)` on the diagonal, so

    sum_{s'} W_{s',s}  =  f(s) .

`W` conserves probability only when `f` is identically zero. It is a non-conservative linear
operator, and results stated for stochastic matrices or their generators do not transfer
without further argument.

---

## 3. Perron-Frobenius: existence, uniqueness, positivity

**S2. For `mu > 0` the dominant eigenvalue of `W` is simple and its eigenvector is strictly
positive.**

Choose `c = mu L + max(0, -min_s f(s))`. Then `W + c I` has non-negative entries: the
off-diagonals are `mu >= 0`, and the shift lifts every diagonal entry to at least zero. Its
graph is the `L`-dimensional hypercube under single-site flips, which is connected, so
`W + c I` is irreducible. Perron-Frobenius gives a simple largest eigenvalue with a strictly
positive eigenvector. A uniform shift changes eigenvalues by `c` and eigenvectors not at all,
so the same holds for `W`.

**S3. At `mu = 0` the operator is reducible.** `W = diag(f)` has every genotype as its own
component. The dominant eigenvector is a point mass at the fitness optimum, and if the optimum
is degenerate it is not unique. The `mu -> 0` limit of the quasispecies is well defined; the
value at `mu = 0` need not be. `crow_kimura._site_ratio` raises when `a = 0` and `mu = 0`
leave a site undetermined.

---

## 4. Symmetry and reversibility

**S4. With site-independent, direction-independent mutation, `W` is a real symmetric
matrix.** `diag(f)` and each `X_i` are symmetric, so their sum is. The spectrum is real and
the eigenvectors are orthogonal.

**S5. `W` is reversible with respect to the uniform measure.** Detailed balance,
`pi_s W_{s,s'} = pi_{s'} W_{s',s}`, holds with `pi` uniform by symmetry, so the reversibility
defect is identically zero.

**S6. Asymmetric per-site mutation stays reversible.** With wild-type-to-mutant rate `mu_f`
and reverse rate `mu_b`, the same at every site, `pi_s = (mu_f / mu_b)^{|s|}` satisfies
detailed balance on every edge, since a single flip changes the Hamming weight `|s|` by one.
The measure is a product measure; the operator is reversible though not symmetric.

**S7. Selection does not affect reversibility.** Detailed balance constrains only
off-diagonal entries, and `diag(f)` is diagonal, so no landscape can make `W` nonreversible.
Across six landscapes, including one with fitness entries spanning fifteen orders of
magnitude, and three mutation models, the defect is unchanged to `1e-12`.

**S8. Direction-specific context-dependent mutation breaks reversibility.** If the rate at
site `i` depends on neighbouring sites differently in the two directions, Kolmogorov's cycle
criterion fails on the hypercube and no `pi` exists. The constructed case in
`spectral/perron.py` has defect `0.60`, against exactly `0` for every symmetric or
context-free case.

Consequently, constructions whose speedup relies on nonreversibility, such as that of
Claudon, Piquemal and Monmarche for stationary distributions, do not apply to this
generator for any fitness landscape.

---

## 5. Stoquasticity

**S9. `H = -W` is stoquastic, and the quasispecies is its ground state with non-negative
amplitudes.** The off-diagonal entries of `H` are `-mu <= 0`, so its ground state can be
chosen entrywise non-negative, which is the Perron vector of `W` by S2.

Imaginary-time evolution under `H` therefore converges to the quasispecies from any initial
state with non-zero overlap, without sign cancellation, and amplitudes can be read as square
roots of probabilities without phase ambiguity.

---

## 6. Pauli structure

Write `z_i = 1 - 2 s_i`, so `z_i = +1` is wild type at site `i`. In Pauli operators,

    W  =  sum_{A subset of sites} c_A * Z_A   +   mu * sum_i X_i   -   mu L * I

where `Z_A` is the product of `Z` over `A` and the `c_A` are the Walsh-Hadamard coefficients
of `f`. The transverse part is always `L` terms; the cost lies in the diagonal.

**S10. The number of Pauli terms is set by the polynomial degree of the landscape in the
`z` variables.**

| landscape | form | terms |
|---|---|---|
| additive | `sum_i a_i z_i` | `L` |
| additive plus pairwise | `sum a_i z_i + sum b_ij z_i z_j` | up to `L(L+1)/2` |
| single peak as a projector | `height * prod_i (1 + z_i) / 2^L` | `2^L` |

The single peak is a product over all sites, so its Walsh expansion is dense; written as a
low-degree function of Hamming distance it is sparse. At `L = 12` the two forms need `4108`
and `27` terms, a ratio of 152.1, and the ratio grows with `L`: 1.8, 4.7, 13.9, 45.0 and
152.1 at `L = 4, 6, 8, 10, 12`. The package therefore uses the spin convention `a_i Z_i`,
`b_ij Z_i Z_j`.

**S11. Permutation symmetry does not make a landscape cheap.** A landscape quadratic in
Hamming distance has support only to Pauli weight two (28 terms at `L = 6`), while one
exponential in Hamming distance is dense (37 terms at `L = 5`, all subsets). Low polynomial
degree is the relevant property.

---

## 7. Spectrum

### 7.1 Additive landscapes factorise

If `f = sum_i a_i z_i` then

    W  =  sum_i ( a_i Z_i  +  mu X_i  -  mu I )

is a sum of `L` commuting single-site operators with eigenvalues `-mu +/- sqrt(a_i^2 + mu^2)`,
so the full spectrum is every sum of independent choices.

**S12a.** `lambda_1 = sum_i ( -mu + sqrt(a_i^2 + mu^2) )`.

**S12b.** The gap is `Delta = 2 min_i sqrt(a_i^2 + mu^2)`, independent of `L`. The second
eigenvector flips the cheapest single site; with equal sites `lambda_2` is `L`-fold
degenerate. This agrees with dense diagonalisation to `2.9e-14` over 28 configurations.

Because the gap never closes, every eigenvector-extraction method converges in `O(1)`
iterations on this family at every size.

### 7.2 Permutation-symmetric landscapes

When `f` depends on `s` only through its Hamming weight, `W` commutes with the symmetric
group and the symmetric sector is `L + 1` dimensional. In the binomially symmetrised basis
it is tridiagonal with

    diagonal_d = f_d - mu L ,      offdiagonal_d = mu * sqrt( (d+1)(L-d) ) .

The Perron vector lies in this sector; `lambda_2` need not, so the sector gap is an upper
bound on the full gap. For the single peak the two coincide at `L = 4, 6, 8` across four
mutation rates, which `symmetric_sector_holds_lambda2` tests per landscape. Imaginary time
started from a symmetric state sees the sector gap; a general start sees the full gap.

### 7.3 Above the error threshold

When selection is weak, `W` is dominated by `mu sum_i (X_i - I)`, whose eigenvalues are
`mu(L - 2k) - mu L` for `k = 0..L`, with gap `2 mu`. The single-peak gap approaches this from
below as `L` grows, agreeing to twelve digits at `mu = 0.5`, `L = 128`.

### 7.4 The threshold

For the single peak of height `h`, the gap minimised over `mu` sits at `mu* L -> h`, with a
`1/L` correction of coefficient about 1. `mu* L / h` agrees to five digits between `h = 1.0`
and `h = 2.5` at every `L` from 8 to 1024. At that point the gap closes exponentially in `L`;
the rate is reported in `results/wp1/g_1.json`.

---

## 8. Amplitudes, probabilities and the decode

The circuit prepares the quasispecies in amplitudes and measurement returns their squares.
Comparing measured frequencies directly with the biological distribution compares `p^2` with
`p`, which gives cosine `0.9865` at a total variation of `0.224`. Taking the square root and
renormalising in L1 brings total variation to `0.0036` at 100k shots.

The square root is steep near zero and amplifies the sampling floor: a component that should
be zero but lands at `1e-5` decodes to `3e-3`. Shot requirements therefore scale against the
decoded distribution.

---

## 9. Runtime and the gap

Each eigenvector-extraction method pays for a small gap in its own currency:

| method | cost to reach accuracy `eps` |
|---|---|
| imaginary-time evolution | `tau ~ ln(1/eps) / Delta` |
| power iteration | `~ ln(1/eps) / ln(lambda_1/lambda_2)` |
| QSVT eigenvector filtering | polynomial degree `~ (1/Delta) ln(1/eps)` |
| varQITE | the above, with a gradient variance that decays exponentially in `L` |

- Additive and class-invariant landscapes: gap `Theta(1)`, independent of `L`, and a
  polynomial-time classical algorithm is known.
- Away from the threshold: gap `Theta(1)`; above it, exactly `2 mu`.
- At the threshold: gap exponentially small in `L`, so every method in the table costs
  exponential time there.

Computing the quasispecies is polynomial for the class-invariant landscapes and open in
general. The gap map localises any hardness at the error threshold, the only place the gap
closes. A quantum advantage there would require a cost scaling as `1/sqrt(Delta)` where a
classical method needs `1/Delta`, on a family with no known polynomial classical algorithm.

---

## 10. Claim-to-check index

| | claim | checked by |
|---|---|---|
| S1 | columns sum to `f`, not zero | `test_perron.py`, `results/wp0/wp0_reversibility.json` |
| S2 | simple dominant eigenvalue, positive eigenvector, `mu > 0` | `test_perron.py`, `test_analytic.py` |
| S3 | `mu = 0` is reducible and may be degenerate | `test_analytic.py` |
| S4 | `W` symmetric under symmetric mutation | `test_perron.py` |
| S5 | reversible, defect exactly zero | `results/wp0/wp0_reversibility.json` |
| S6 | asymmetric per-site mutation still reversible, product measure | `results/wp0/wp0_reversibility.json` |
| S7 | selection cannot affect reversibility | `test_perron.py`, 18 landscape-mutation pairs |
| S8 | context-dependent directional mutation breaks it, defect `0.60` | `results/wp0/wp0_reversibility.json` |
| S9 | `-W` stoquastic, ground state non-negative | `results/wp_r/g_r_2.json` |
| S10 | Pauli count set by polynomial degree; ratio 152.1 at `L = 12` | `results/wp_r/g_r_10.json` |
| S11 | permutation symmetry is not what makes a landscape cheap | `results/wp_r/g_r_10.json` |
| S12 | additive gap `2 min sqrt(a^2 + mu^2)`, `L`-independent | `test_gap.py` |
