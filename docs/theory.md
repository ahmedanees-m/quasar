# The mutation-selection operator: structure, spectrum, and what follows for algorithms

WP1 tasks T1.1 and T1.5. The execution plan attaches a warning to T1.1 that this document
takes literally: *"Be careful and exact here. Do not overclaim nonreversibility; derive what
is actually true for each landscape family."*

Every structural claim below is numbered **S1** to **S12** and carries the test or artefact
that checks it. A claim with no check next to it does not belong here.

---

## 1. The object

Genotypes are length-`L` binary strings, `s` in `{0,1}^L`, indexed as in
`quasarstack/io/conventions.py`: site `i` is bit `i` is qubit `i`, little-endian throughout.
A landscape assigns each genotype a Malthusian fitness `f(s)`. Mutation flips each site
independently at rate `mu`.

The Crow-Kimura (parallel mutation-selection) dynamics on the vector of genotype frequencies
`x` is

    dx/dt  =  W x  -  <f> x,        W  =  diag(f)  +  mu * sum_i (X_i - I)

where `X_i` is the bit-flip on site `i` and `<f> = f . x` keeps the total normalised. The
nonlinear term is a scalar multiple of `x`, so it changes the norm and nothing else. The
equilibrium is therefore the dominant eigenvector of the **linear** operator `W`, and the
equilibrium mean fitness is its eigenvalue. Everything below is about `W`.

Two normalisations coexist and are the most common source of silent error in this project.
The biology wants `sum_s p(s) = 1` (L1). A quantum state carries `sum_s |psi(s)|^2 = 1` (L2).
`conventions.py` holds both and the conversion; section 8 returns to what measurement does
to the difference.

---

## 2. It is not a Markov generator

**S1. The columns of `W` sum to `f(s)`, not to zero.**

The mutation part contributes `L` off-diagonal entries of `mu` to each column and a diagonal
`-mu L`, summing to zero. The selection part adds `f(s)` on the diagonal. So

    sum_{s'} W_{s',s}  =  f(s) .

`W` conserves probability only when `f` is identically zero. It is a non-conservative linear
operator, not a Markov generator, and results stated for stochastic matrices or their
generators do not transfer without an argument.

*Checked by* `tests/unit/test_perron.py`, and this is one of the two independent reasons
recorded in docs/notes.md that the Claudon-Piquemal-Monmarche construction does not apply here.

---

## 3. Perron-Frobenius: existence, uniqueness, positivity

**S2. For `mu > 0` the dominant eigenvalue of `W` is simple and its eigenvector is strictly
positive.**

Choose `c = mu L + max(0, -min_s f(s))`. Then `W + c I` has non-negative entries: the
off-diagonals are `mu >= 0` already, and the shift lifts every diagonal entry to at least
zero. Its graph is the `L`-dimensional hypercube under single-site flips, which is connected,
so `W + c I` is irreducible. Perron-Frobenius then gives a simple largest eigenvalue with a
strictly positive eigenvector. A uniform shift changes eigenvalues by `c` and eigenvectors
not at all, so the same holds for `W`.

**S3. At `mu = 0` all of this fails, and it fails in a way worth naming.** `W = diag(f)` is
reducible: every genotype is its own component. The dominant eigenvector is a point mass at
the fitness optimum, and if the optimum is degenerate the eigenvector is not unique. The
`mu -> 0` limit of the quasispecies is well defined; the value *at* `mu = 0` need not be.
`crow_kimura._site_ratio` raises rather than returning a wrong answer when `a = 0` and
`mu = 0` leave a site undetermined.

*Checked by* `tests/unit/test_analytic.py` and `tests/unit/test_perron.py`.

---

## 4. `W` is symmetric, and this is the fact that decided Route B

**S4. With site-independent, direction-independent mutation, `W` is a real symmetric
matrix.**

`diag(f)` is symmetric. Each `X_i` is symmetric. A sum of symmetric matrices is symmetric.
Therefore the spectrum is real, the eigenvectors are orthogonal, and no Jordan structure
exists.

**S5. `W` is reversible with respect to the uniform measure, and its reversibility defect is
exactly zero.**

Detailed balance asks for a positive `pi` with `pi_s W_{s,s'} = pi_{s'} W_{s',s}`. Symmetry
gives it immediately with `pi` uniform. There is nothing to prove and nothing to measure:
the defect is zero identically, not numerically.

**S6. Asymmetric per-site mutation stays reversible.** Let wild-type-to-mutant run at `mu_f`
and the reverse at `mu_b`, both positive and the same at every site. Then
`pi_s = (mu_f / mu_b)^{|s|}`, where `|s|` is the Hamming weight, satisfies detailed balance
on every edge: a single flip changes `|s|` by one and the ratio absorbs it. The measure is a
product measure and the operator is reversible, though no longer symmetric.

**S7. Selection cannot affect reversibility at all.** Detailed balance constrains only
off-diagonal entries. `diag(f)` is diagonal. So no landscape, however rugged, however
epistatic, can make `W` nonreversible. Ruggedness is not a route to nonreversibility.

This closes off the tempting repair to docs/notes.md. If ruggedness could buy nonreversibility,
Route B's stated foundation could be recovered by choosing a harder landscape family, and the
project could carry on as planned. It cannot: reversibility is fixed entirely by the mutation
model. Measured across six landscapes, including one with fitness entries spanning fifteen
orders of magnitude, crossed with all three mutation models. The defect is unchanged to
`1e-12` in every case, including the context-dependent model where it is `0.60` rather than
zero, so the claim is that selection leaves the defect alone rather than merely leaving zero
alone.

**S8. What does break reversibility is direction-specific context-dependent mutation.** If
the rate at site `i` depends on the states of neighbouring sites, and depends on them
differently in the two directions, Kolmogorov's cycle criterion fails on the hypercube and no
`pi` exists. Measured defect `0.60` for the constructed case in `spectral/perron.py`, against
exactly `0` for every symmetric and every asymmetric-but-context-free case.

*Checked by* `results/wp0/prior_art_iv_4.json`, twenty operators, and
`tests/unit/test_perron.py`.

**Why this matters more than it looks.** Execution plan v4 rests Route B on a speedup that
comes *from* nonreversibility. Our operator is reversible by construction, and S7 says the
biology cannot be adjusted to change that. Combined with S1, the stated foundation of the
project's novelty core does not hold. This is in docs/notes.md, and it is open for both PIs.

---

## 5. Stoquasticity: why there is no sign problem

**S9. `H = -W` is stoquastic, and the quasispecies is its ground state with non-negative
amplitudes.**

The off-diagonal entries of `H` are `-mu <= 0`. A Hamiltonian whose off-diagonals in the
computational basis are all non-positive is stoquastic by definition. Its ground state can be
chosen entrywise non-negative, which is exactly the Perron vector of `W` by S2.

Two consequences the project uses constantly. Imaginary-time evolution under `H` converges to
the quasispecies from any initial state with non-zero overlap, with no sign cancellation.
And the amplitudes can be read as the square roots of biological probabilities without an
ambiguity about phase, which is what makes the decode in section 8 well posed.

*Checked by* `results/wp_r/g_r_2.json`, cosine `1.000000` over 40 configurations.

---

## 6. Pauli structure, and why the spin convention is a factor of 152

Write `z_i = 1 - 2 s_i`, so `z_i = +1` is wild type at site `i`. In Pauli operators,

    W  =  sum_{A subset of sites} c_A * Z_A   +   mu * sum_i X_i   -   mu L * I

where `Z_A` is the product of `Z` over `A` and the `c_A` are the Walsh-Hadamard coefficients
of `f`. The transverse part is always `L` terms. The cost is entirely in the diagonal.

**S10. The number of Pauli terms is set by the polynomial degree of the landscape in the
`z` variables, not by its symmetry.**

| landscape | form | terms |
|---|---|---|
| additive | `sum_i a_i z_i` | `L` |
| additive plus pairwise | `sum a_i z_i + sum b_ij z_i z_j` | up to `L(L+1)/2` |
| single peak as a projector | `height * prod_i (1 + z_i) / 2^L` | `2^L` |

The single peak is a product over all sites, so its Walsh expansion is dense. Written instead
as a low-degree function of Hamming distance it is sparse. At `L = 12` the two forms need
`4108` and `27` terms, a ratio of **152.1**, and the ratio grows with `L`: 1.8, 4.7, 13.9,
45.0, 152.1 at `L = 4, 6, 8, 10, 12`.

This is why `notes.md` fixes the spin convention `a_i Z_i`, `b_ij Z_i Z_j` and forbids the
projector form. The rule reads like style and is a factor of 152 at the size we care about.

**S11. Permutation symmetry is not what makes a landscape cheap.** Two permutation-symmetric
families behave completely differently: a landscape quadratic in Hamming distance has support
only to Pauli weight two (28 terms at `L = 6`), while one exponential in Hamming distance is
dense (37 terms at `L = 5`, all subsets). Low polynomial degree is the property that matters.

*Checked by* `results/wp_r/g_r_10.json`.

---

## 7. Spectrum

### 7.1 Additive landscapes factorise completely

If `f = sum_i a_i z_i` then

    W  =  sum_i ( a_i Z_i  +  mu X_i  -  mu I )

is a sum of `L` commuting single-site operators. Each has eigenvalues
`-mu +/- sqrt(a_i^2 + mu^2)`, so the full spectrum is every sum of independent choices.

**S12a.** `lambda_1 = sum_i ( -mu + sqrt(a_i^2 + mu^2) )`.

**S12b. The gap is `Delta = 2 min_i sqrt(a_i^2 + mu^2)`, exactly, and it does not depend on
`L`.** The second eigenvector flips the cheapest single site. With equal sites there are `L`
equally cheap ways to do it, so `lambda_2` is `L`-fold degenerate.

Verified against dense diagonalisation to `2.9e-14` over 28 configurations.

**This makes the additive family a ruler, not a target.** Its gap never closes, so every
eigenvector-extraction method converges in `O(1)` iterations at every size, quantum or
classical. No advantage claim can be built on it, and any that appears to be is measuring
something else.

*Checked by* `tests/unit/test_gap.py`.

### 7.2 Permutation-symmetric landscapes reduce, but only within one sector

When `f` depends on `s` only through its Hamming weight, `W` commutes with the symmetric
group and the symmetric sector is `L + 1` dimensional. In the binomially symmetrised basis it
is tridiagonal with

    diagonal_d = f_d - mu L ,      offdiagonal_d = mu * sqrt( (d+1)(L-d) ) .

The Perron vector lives in this sector, being strictly positive. **`lambda_2` need not.** The
reduction is `L + 1` numbers out of `2^L`, so a gap computed there is an upper bound on the
true gap, and the two coincide only when the second eigenvector happens to be symmetric. For
the single peak it does coincide, checked at `L = 4, 6, 8` across four mutation rates. This
is measured by `symmetric_sector_holds_lambda2` rather than assumed, because the answer is
family-dependent and the failure mode is a gap map that is quietly too optimistic.

The practical statement: **which gap governs convergence depends on what the initial state
overlaps.** Imaginary time started from a symmetric state stays in the symmetric sector and
sees the sector gap. A general start sees the full gap.

### 7.3 Above the error threshold the gap is the mutation gap

Once selection has lost, `W` is dominated by `mu sum_i (X_i - I)`, whose eigenvalues are
`mu(L - 2k) - mu L` for `k = 0..L`, so its gap is exactly `2 mu`. The single-peak gap
approaches this from below as `L` grows: at `mu = 0.5` and `L = 128` it agrees to twelve
digits.

### 7.4 The threshold, and where the hardness lives

For the single peak of height `h`, the gap minimised over `mu` sits at

    mu* L  ->  h,   with a 1/L correction of coefficient about 1.

The collapse across heights is exact: `mu* L / h` agrees to five digits between `h = 1.0` and
`h = 2.5` at every `L` tested from 8 to 1024.

At that point the gap **closes exponentially in `L`**. Exploratory measurement puts the base
near `0.71`, meaning the gap at the threshold falls by roughly a factor of 2 for every two
extra sites. The number is registered and measured properly by check G-1; it is quoted here as
exploratory and should be read from the artefact, not from this sentence.

---

## 8. Amplitudes, probabilities, and the decode

The circuit prepares the quasispecies in **amplitudes**. Measurement returns their squares.
Comparing measured frequencies directly against the biological distribution compares `p^2`
against `p`, and cosine similarity barely notices: `0.9865` cosine at a total variation of
`0.224`. Taking the square root and renormalising in L1 inverts it and brings total variation
to `0.0036` at 100k shots.

The cost is real and belongs in every shot-count estimate. The square root is steep near
zero, so it amplifies the sampling floor along with the signal: a component that should be
zero but lands at `1e-5` decodes to `3e-3`. **Shot requirements scale against the decoded
distribution, not the raw one.** Any claim about resolving a quasispecies tail has to be
argued at the decoded noise floor.

*Checked by* `results/wp_r/g_r_8.json` and `quasarstack/io/conventions.py`.

---

## 9. What the gap implies for runtime (T1.5)

Every eigenvector-extraction method pays for a small gap, in its own currency:

| method | cost to reach accuracy `eps` |
|---|---|
| imaginary-time evolution | `tau ~ ln(1/eps) / Delta` |
| power iteration | `~ ln(1/eps) / ln(lambda_1/lambda_2)` |
| QSVT eigenvector filtering | polynomial degree `~ (1/Delta) ln(1/eps)` |
| varQITE | the above, *and* an optimisation whose gradient variance decays exponentially in `L` |

So the gap map is the shared denominator, and the structure above says where the denominator
is dangerous:

- **Additive / Dixit-Vishnoi class: gap `Theta(1)`, independent of `L`.** Easy for everything.
  Also the class for which a polynomial-time classical algorithm is known. Both facts point
  the same way and neither supports an advantage claim.
- **Away from the threshold, generally: gap `Theta(1)`.** Above it, exactly `2 mu`.
- **At the threshold: gap exponentially small in `L`.** Every method in the table costs
  exponential time there.

**The honest complexity statement.** Computing the quasispecies is polynomial for the
Dixit-Vishnoi class. In general it is open. What the gap map establishes is not a hardness
result but a *localisation*: whatever hardness exists is concentrated at the error threshold,
because that is the only place the gap closes. A quantum advantage, if there is one, has to
be argued there, and an exponentially small gap is a cost for the quantum methods too. It is
not evidence for them.

**What would count as an advantage, stated in advance.** A quantum method that reaches the
quasispecies at the threshold with cost scaling as `1/sqrt(Delta)` where a classical method
needs `1/Delta`, on a family with no known polynomial classical algorithm. Nothing measured
so far demonstrates that, and Route B's stated route to it does not apply (S1, S7). This
paragraph exists so that the eventual result is compared against a target written before the
measurement rather than after it.

---

## 10. Claim-to-check index

| | claim | checked by |
|---|---|---|
| S1 | columns sum to `f`, not zero | `test_perron.py`, `results/wp0/prior_art_iv_4.json` |
| S2 | simple dominant eigenvalue, positive eigenvector, `mu > 0` | `test_perron.py`, `test_analytic.py` |
| S3 | `mu = 0` is reducible and may be degenerate | `test_analytic.py` |
| S4 | `W` symmetric under symmetric mutation | `test_perron.py` |
| S5 | reversible, defect exactly zero | `results/wp0/prior_art_iv_4.json` |
| S6 | asymmetric per-site mutation still reversible, product measure | `results/wp0/prior_art_iv_4.json` |
| S7 | selection cannot affect reversibility | `test_perron.py`, 18 landscape-mutation pairs |
| S8 | context-dependent directional mutation breaks it, defect `0.60` | `results/wp0/prior_art_iv_4.json` |
| S9 | `-W` stoquastic, ground state non-negative | `results/wp_r/g_r_2.json` |
| S10 | Pauli count set by polynomial degree; ratio 152.1 at `L = 12` | `results/wp_r/g_r_10.json` |
| S11 | permutation symmetry is not what makes a landscape cheap | `results/wp_r/g_r_10.json` |
| S12 | additive gap `2 min sqrt(a^2 + mu^2)`, `L`-independent | `test_gap.py` |
