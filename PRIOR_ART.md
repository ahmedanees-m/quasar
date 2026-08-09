# PRIOR_ART.md — the four-literature dossier

This project sits at the intersection of four literatures. Missing any one of them is the
failure mode that has already forced two repositions of this project. This file is a living
document; the adversarial red-team (T9.3) checks it explicitly for gaps.

**Verification status.** Each entry carries a status flag:

- `verified` — the paper has been read in this project and the summary below reflects it.
- `to-verify` — the entry was carried over from the planning documents and must be checked
  against the source before it is cited in the manuscript.

Nothing may be cited in the manuscript while still marked `to-verify`. This is WP0 task
T0.2 and T0.3, and it gates WP4 onward via G-0.

Entry format: citation, what it establishes, what it leaves open, how QUASAR relates.

---

## Literature I — the evolution to spin-chain correspondence

The mathematical bridge the project stands on. This literature is mature. Nothing here is
novel to us, and the manuscript says so in the introduction rather than in a limitations
paragraph.

### I.1 Leuthäusser (1986/87) — `to-verify`

- **Establishes.** Eigen's quasispecies model maps onto a two-dimensional Ising system.
- **Leaves open.** A statistical-physics equivalence, not an algorithm. No treatment of
  computational cost, and no circuit.
- **Relation.** The origin of the correspondence. Cited as the founding result.

### I.2 Baake, Baake & Wagner, *Phys. Rev. Lett.* 78:559 (1997), with erratum PRL 79:1782 — `to-verify`

- **Establishes.** The Crow–Kimura parallel mutation–selection model is exactly a quantum
  spin chain. The equation governing genotype frequencies is a Schrödinger equation in
  imaginary time with mutation entering as a transverse field.
- **Leaves open.** Exactness of the mapping, not tractability. No quantum-algorithmic
  analysis: no spectral gap treatment, no oracle model, no resource scaling.
- **Relation.** The specific correspondence QUASAR implements. The erratum must be checked
  before any convention is copied from the original.

### I.3 Saakian & Hu, *Phys. Rev. E* 69:021913 and 69:046121 (2004) — `to-verify`

- **Establishes.** Exact dynamics and closed-form quasispecies distributions for the Eigen
  model as a quantum spin chain, via the Suzuki–Trotter formalism.
- **Leaves open.** Closed forms exist for structured landscapes. Rugged and
  broken-symmetry landscapes are not covered.
- **Relation.** A source of the analytic oracle used as the ruler in `quasarstack/analytic/`.

### I.4 Jain & Krug (2005), arXiv q-bio/0508008 — `to-verify`

- **Establishes.** Mutation–selection with pairwise epistasis written explicitly as a
  transverse-field Ising Hamiltonian.
- **Leaves open.** The Hamiltonian form, not its simulation cost.
- **Relation.** The epistatic term `b_ij Z_i Z_j` in the Hamiltonian compiler follows this.

### I.5 Park & Deem (2006) — `to-verify`

- **Establishes.** Named in the honest-reframe document as part of the correspondence
  literature. Content to be confirmed.
- **Relation.** To be determined on reading.

---

## Literature II — classical algorithms for quasispecies

The most dangerous literature for this project, because it contains results that solve the
target problem outright. The manuscript leads with these rather than burying them.

### II.1 Dixit, Srivastava & Vishnoi, *J. Comput. Biol.* (2012), arXiv:1203.1287 — `to-verify`

- **Establishes.** A deterministic polynomial-time algorithm for the stationary
  distribution of finite-population molecular evolution, for the class of fitness landscapes
  it covers. Framed explicitly for mutagenic-drug design, which is the application QUASAR
  points at. For single-peak and permutation-symmetric landscapes the steady state is the
  noisy-hypercube matrix with an explicit spectrum.
- **Leaves open.** The applicability class. The result assumes structured landscapes.
  Rugged, broken-symmetry, strong-epistasis landscapes are not covered.
- **Relation.** Baseline B, and the single most important entry in this file. Its
  applicability boundary *defines* the candidate quantum-relevant regime. Where it applies
  it is expected to win outright, and the paper says so. It is invisible to the physics
  benchmarking literature, which is precisely why including it differentiates this work.

### II.2 Dalmau (2014, 2018) — `to-verify`

- **Establishes.** Closed-form Wright–Fisher quasispecies distributions on sharp-peak and
  class-dependent landscapes.
- **Leaves open.** Closed forms for structured landscapes only.
- **Relation.** A second analytic check on the finite-population baseline, and a further
  reason no advantage claim is available on the easy landscapes.

### II.3 Cerf & Dalmau (2016; monograph 2022) — `to-verify`

- **Establishes.** Exact formulas for the quasispecies distribution in the Wright–Fisher
  framework.
- **Relation.** As above.

---

## Literature III — quantum imaginary-time evolution methods

Saturated. The execution plan demotes the varQITE-versus-Motta comparison from a
contribution to a methods subsection because of this literature. The manuscript makes no
novelty claim here.

### III.1 Motta et al. (2020) — `to-verify`

- **Establishes.** QITE and QLanczos: imaginary-time evolution reproduced by unitaries whose
  generators are found by a linear solve from measured Pauli expectations. No variational
  optimisation, therefore no barren plateaus.
- **Leaves open.** Generator support and circuit depth grow as correlations spread.
- **Relation.** Route A fallback, `quasarstack/ite/qite_motta.py`.

### III.2 McArdle et al. (2019) — `to-verify`

- **Establishes.** Variational imaginary-time evolution by the McLachlan variational
  principle: solve `A theta_dot = C` with A and C from parameter-shift circuit evaluations.
  Circuit depth is constant in imaginary time.
- **Leaves open.** Barren plateaus. The planning documents record a ceiling near L = 10 to
  12 with gradient variance decaying as roughly 0.42^L.
- **Relation.** Route A primary, `quasarstack/ite/varqite.py`.

### III.3 Nishi, Kosugi & Matsushita (2020) — `to-verify`

- **Establishes.** Probabilistic implementation of imaginary-time evolution.
- **Relation.** Alternative ITE route; cited to show the method space is well populated.

### III.4 Probabilistic ITE (PITE), 2024 — `to-verify`

- **Establishes.** Continued development of probabilistic ITE.
- **Relation.** Evidence that ITE methods on TFIM are an active and crowded area, which is
  the reason this project does not claim them as a contribution.

### III.5 Nonlocal-approximation QITE, *npj Quantum Information* — `to-verify`

- **Relation.** Same as III.4.

### III.6 Quasiprobabilistic ITE — `to-verify`

- **Relation.** Same as III.4.

### III.7 RL-designed ITE ansaetze (2026) — `to-verify`

- **Relation.** Same as III.4. The most recent entry; confirms the area is still moving.

---

## Literature IV — quantum algorithms for classical stochastic processes

The literature v3 of the plan ignored entirely, and the reason v4 exists. This is where
Route B connects to provable-complexity results rather than heuristics.

### IV.1 Quantum-enhanced simulation of stochastic processes, *PRX* (2021) — `to-verify`

- **Establishes.** Memory and time advantages for quantum simulation of classical
  stochastic processes in specific scenarios.
- **Leaves open.** The advantages are scenario-specific, not general.
- **Relation.** Establishes that the question "does quantum help for a classical stochastic
  process" has a real, non-trivial answer space.

### IV.2 Aghamohammadi, Mahoney & Crutchfield — `to-verify`

- **Establishes.** Extreme quantum advantage in *memory* for generating classical
  spin-chain configurations.
- **Leaves open.** Memory advantage, not time advantage, and not eigenvector extraction.
- **Relation.** A precedent for advantage in a related setting; cited to keep the scope
  claim honest in both directions.

### IV.3 Orfi & Sels (Flatiron, 2024) — `to-verify`

- **Establishes.** No worst-case speedup for quantum-enhanced Markov chain Monte Carlo.
- **Relation.** A negative result that constrains what Route B may claim. Cited directly
  against any temptation to overclaim.

### IV.4 Claudon, Piquemal & Monmarche — "Quantum speedup for nonreversible Markov chains", arXiv:2501.05868, *Nature Communications* 16:10732 (2025) — ✅ `verified` 2026-08-09

- **Establishes.** Two quantum methods for sampling the stationary distribution of a
  **row-stochastic Markov kernel**. A generalised quantum singular value transform of the
  *curved* discriminant, which generalises the reversible-case technique and needs the time
  reversal, hence knowledge of the stationary distribution; and a generalised quantum
  eigenvalue transform of the *flat* discriminant, which does not, and instead requires a
  weaker property the authors call reversibility on pi-average. Reversibility is defined by
  detailed balance. **The headline beyond-quadratic, up-to-exponential speedup comes
  specifically from the chain being nonreversible**; reversible chains get the previously
  known quadratic acceleration. Complexity scales as the square root of the product of the
  reversibilisation time and the mixing time, with the pseudo-spectral gap governing mixing.

- **Leaves open.** Everything outside the row-stochastic class. The paper does not treat
  non-conservative operators.

- **Relation to QUASAR — verified by measurement, and it is not what execution plan v4
  assumed.** Artefact: `results/wp0/prior_art_iv_4.json`, twenty operators.

  Plan v4 section 0 argues that "the quasispecies is the Perron eigenvector of a
  non-conservative linear operator" and that extracting a dominant eigenvector "is precisely
  what QSVT eigenvalue transforms do". The first half is correct. The inference is not, for
  two independent reasons.

  1. **Class mismatch.** Their theorems are stated for row-stochastic kernels. The
     mutation-selection generator is not one: its columns sum to the fitness, not to zero or
     one. Measured, `max_abs_column_sum` is non-zero for every landscape tested. Converting
     it to a stochastic matrix is possible in principle by a Doob h-transform, but that
     transform is built from the Perron vector, which is the object being computed. The
     conversion is circular.

  2. **Property mismatch, and this is the decisive one.** Their speedup is bought by
     nonreversibility. **The mutation-selection generator is reversible.** With the
     symmetric mutation the project implements it is not merely reversible but symmetric,
     with a reversibility defect of exactly zero and a uniform symmetrising measure.

  Two further measured results sharpen the scope:

  - **Asymmetric per-site mutation does not help.** Different forward and backward rates
    make the generator non-symmetric but leave it reversible, defect 1e-15, because
    independent per-site flips are a product of two-state birth-death processes and those
    are reversible whatever the rates.
  - **Selection cannot create nonreversibility at all.** Detailed balance constrains only
    off-diagonal entries and selection is diagonal, so the reversibility defect is identical
    across flat, additive, epistatic and single-peak landscapes. Ruggedness, the project's
    main axis, is irrelevant to this property.

  **The one route into their class that was found.** Direction-specific context-dependent
  mutation, where the forward rate at a site depends on a neighbour but the back-mutation
  rate does not, gives a reversibility defect of 0.60. This is biologically faithful, since
  CpG hypermutation and APOBEC motif preference raise C to T without raising T to C. The
  same context factor applied to *both* directions leaves the chain reversible, because the
  rate then factorises into direction times context and that product cancels out of
  Kolmogorov's cycle condition. Both were measured; the control is in the artefact.

- **Consequence for Route B.** See `DECISIONS.md` ADR-0010. Route B is not dead, but it
  cannot be built on this reference as planned, and the plan's framing needs correcting
  before WP2 starts.

---

## Literature V — classical tensor-network simulation (the benchmark frontier)

Not one of the four named literatures, but it sets the bar the boundary map is measured
against, so it belongs in this dossier.

### V.1 Flatiron CCQ / Tindall and related work on 3D tensor networks — `to-verify`

- **Establishes.** Custom tensor-network methods at hundreds of spins have repeatedly
  overturned quantum-advantage claims for transverse-field Ising dynamics.
- **Relation.** Sets the scope limit stated in T6.5. This project benchmarks against a
  standard, well-tuned MPS implementation at accessible L, not against bespoke
  state-of-the-art tooling at hundreds of spins, and the manuscript scopes the claim to what
  was actually tested.

### V.2 MPO-based spin-glass methods — `to-verify`

- **Relation.** As above.

---

## Gap log

Where the red-team or a reviewer identifies a literature this file misses, it is recorded
here with the date, so the history of what was missed and when is visible.

*(No entries yet.)*
