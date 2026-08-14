# PRIOR_ART.md: the four-literature dossier

This project sits at the intersection of four literatures. Missing any one of them is the
failure mode that has already forced two repositions of this project. This file is a living
document; the adversarial red-team (T9.3) checks it explicitly for gaps.

**Verification status.** Each entry carries a status flag:

- `verified`: the paper has been read in this project and the summary below reflects it.
- `to-verify`: the entry was carried over from the planning documents and must be checked
  against the source before it is cited in the manuscript.

Nothing may be cited in the manuscript while still marked `to-verify`. This is WP0 task
T0.2 and T0.3, and it gates WP4 onward via G-0.

Entry format: citation, what it establishes, what it leaves open, how QUASAR relates.

---

## Literature group I: the evolution to spin-chain correspondence

The mathematical bridge the project stands on. This literature is mature. Nothing here is
novel here, and the manuscript says so in the introduction rather than in a limitations
paragraph.

### I.1 Leuthäusser, *J. Chem. Phys.* 84(3):1884 (1986); *J. Stat. Phys.* 48:343 (1987): `verified 2026-08-15, bibliographic`

- **Establishes.** "An exact correspondence between Eigen's evolution model and a
  two-dimensional Ising system". For point mutations, Eigen's model maps onto a
  **two-dimensional** Ising system with nearest-neighbour interaction in one direction. The
  1987 paper, "Statistical mechanics of Eigen's evolution model", is the extended treatment.
- **Leaves open.** A statistical-physics equivalence, not an algorithm. No computational cost,
  no circuit.
- **Relation.** The origin of the correspondence, cited as the founding result. Note the
  dimensionality: this is a two-dimensional Ising system, whereas the chain QUASAR works with
  is the one-dimensional quantum chain of I.2. The two are related by the usual quantum to
  classical correspondence and are not the same object, so the citation must not be used to
  support a statement about the quantum chain.

### I.2 Baake, Baake & Wagner, *Phys. Rev. Lett.* 78:559 (1997), erratum PRL 79:1782: `verified 2026-08-15, bibliographic`

- **Establishes.** "Ising quantum chain is equivalent to a model of biological evolution".
  A sequence-space model of mutation and selection is equivalent to an Ising quantum chain,
  with three representative fitness landscapes solved exactly by statistical mechanics.
- **Leaves open.** Exactness of the mapping, not tractability. No spectral gap treatment, no
  oracle model, no resource scaling.
- **Relation.** The specific correspondence QUASAR implements. Erratum PRL 79:1782 confirmed
  to exist and must be consulted before any convention is copied from the original.

### I.3 Saakian & Hu, *Phys. Rev. E* 69:021913 and 69:046121 (2004), arXiv:cond-mat/0402212: `verified 2026-08-15, bibliographic`

- **Establishes.** "Eigen model as a quantum spin chain: exact dynamics" maps the Eigen model
  onto a one-dimensional quantum spin model and derives exact relaxation behaviour using the
  Suzuki-Trotter formalism. The companion paper is "Solvable biological evolution model with a
  parallel mutation-selection scheme".
- **Leaves open.** Closed forms for structured landscapes. Rugged and broken-symmetry
  landscapes are not covered.
- **Relation.** A source of the analytic oracle in `quasarstack/analytic/`. **One difference
  matters:** this work uses a **non-Hermitian** Hamiltonian, whereas QUASAR works with the
  stoquastic Hermitian `H = -W` whose Perron vector is sign-definite. Results are not
  transferable term by term without checking which operator is meant.

### I.4 Jain & Krug, arXiv:q-bio/0508008 (2005): `verified 2026-08-15, corrected`

**The previous entry in this file was wrong and the correction matters.** It recorded this as
writing "mutation-selection with pairwise epistasis explicitly as a transverse-field Ising
Hamiltonian", and credited the epistatic `b_ij Z_i Z_j` term in the Hamiltonian compiler to it.

Read: the paper is **"Adaptation in simple and complex fitness landscapes"**, an introductory
review published in *Structural Approaches to Sequence Evolution* (Springer, 2007, pp. 299-340).
It reviews deterministic mutation-selection models, the error threshold, rugged landscapes and
evolutionary dynamics. It **does not** present the pairwise-epistatic transverse-field Ising
Hamiltonian the entry attributed to it.

- **Relation, corrected.** Cite as a review of quasispecies theory and rugged landscapes. It is
  **not** the source for the `b_ij Z_i Z_j` construction, which is standard Ising notation and
  needs either a correct primary citation or no attribution at all. Nothing in the code changes:
  the compiler's epistatic term is validated against exact diagonalisation by G-R.2 rather than
  taken on authority from a reference.

### I.5 Park & Deem, *J. Stat. Phys.*, arXiv:q-bio/0607012 (2006): `verified 2026-08-15, bibliographic`

- **Establishes.** "Schwinger Boson Formulation and Solution of the Crow-Kimura and Eigen Models
  of Quasispecies Theory". Spin coherent-state functional integrals by the Schwinger boson
  method, giving long-time behaviour for arbitrary replication and degradation functions and the
  phase transitions as a function of mutation rate.
- **Leaves open.** Analytic solution technique, not computational complexity.
- **Relation.** Part of the correspondence literature. Supports the statement that the
  mapping is well established and not claimed as novel here.

---

## Literature II: classical algorithms for quasispecies

The most dangerous literature for this project, because it contains results that solve the
target problem outright. The manuscript leads with these rather than burying them.

### II.1 Dixit, Srivastava & Vishnoi, arXiv:1203.1287 (2012): `verified 2026-08-14`

Read in full from the arXiv PDF. This entry was the one substantive prior-art risk in the file,
because if this paper's efficient class were larger than Baseline B's, WP7 would contain cells it
believes are classically hard which are not. **It is not larger. It is a strict subset**, and the
attribution the execution plan carried was wrong in a way a referee in this field would catch.

- **Exact title.** *A Finite Population Model of Molecular Evolution: Theory and Computation.*
  Note "Finite Population": the paper's own model is the RSM process, a Wright-Fisher-style
  finite-population chain, **not** the infinite-population quasispecies that Baseline B computes.

- **Their efficient class, defined verbatim.** "The fitness landscape is said to be
  class-invariant if `a_sigma` depends only on the Hamming weight of `sigma`." That is exactly
  permutation symmetry.

- **Their algorithm, Theorem 3.3 verbatim.** "For any class invariant fitness landscape A, there
  is an algorithm running in time `T = O(N^O(L^2))` which computes the steady state of the RSM
  process with population size N and the genome length L." This is for the **finite-population**
  model, and `N^O(L^2)` is not polynomial in L.

- **What they attribute to whom, and this is the correction.** For the infinite-population
  quasispecies, the paper says: "In the case of class-invariant fitness landscapes, it is known
  **[SS82]** that one only needs to find the leading eigenvector of an `(L + 1) x (L + 1)`
  matrix." `[SS82]` is Swetina and Schuster 1982. **The `(L+1)`-dimensional reduction that
  Baseline B implements is Swetina-Schuster, not Dixit-Srivastava-Vishnoi**, and this paper says
  so itself.

- **Class relation, decided by measurement rather than by reading.** Baseline B covers
  `{additive} union {permutation symmetric}`. Additive with distinct per-site coefficients
  depends on *which* sites carry mutations, not only how many, so it is **outside** class
  invariance while Baseline B still solves it in `O(L)` as a product state. Checked directly:

  | landscape | DSV class-invariant | Baseline B applies |
  |---|---|---|
  | additive, distinct `a_i` | no | yes, `O(L)` |
  | additive, uniform `a_i` | yes | yes |
  | single peak, symmetric | yes | yes |
  | house of cards | no | no |

- **Consequence for WP7.** None adverse. The boundary map marks *more* cells classically easy
  than this paper alone would justify, so the error, if any, runs against the quantum method
  rather than in its favour. The exposure the review raised does not exist.

- **Relation, corrected.** Cite **Swetina and Schuster (1982)** for Baseline B's
  permutation-symmetric reduction. Cite this paper for the finite-population RSM model, its
  convergence-to-quasispecies result, and its mixing-time condition (Theorem 3.4), which is
  relevant to WP4 rather than to Baseline B. Do **not** describe Baseline B as
  "Dixit-Srivastava-Vishnoi": the plan's label was wrong.

### II.1a Swetina & Schuster, *Biophys. Chem.* 16:329-345 (1982): `verified 2026-08-14 by citation`

- **Establishes.** For class-invariant landscapes the quasispecies is the leading eigenvector of
  an `(L+1) x (L+1)` matrix rather than a `2^L x 2^L` one. This is the reduction Baseline B's
  permutation-symmetric branch implements.
- **Status.** Verified indirectly: arXiv:1203.1287 attributes the reduction to it explicitly and
  by reference number. The primary source has not been read, so the entry is marked by citation
  rather than read in full, and the manuscript should not attribute anything to it beyond the
  `(L+1)` reduction until someone reads it.
- **Relation.** The correct citation for Baseline B, `quasarstack/classical/exact_class.py`.

### II.2 Dalmau (2014, 2018): `to-verify`

- **Establishes.** Closed-form Wright–Fisher quasispecies distributions on sharp-peak and
  class-dependent landscapes.
- **Leaves open.** Closed forms for structured landscapes only.
- **Relation.** A second analytic check on the finite-population baseline, and a further
  reason no advantage claim is available on the easy landscapes.

### II.3 Cerf & Dalmau (2016; monograph 2022): `to-verify`

- **Establishes.** Exact formulas for the quasispecies distribution in the Wright–Fisher
  framework.
- **Relation.** As above.

---

## Literature III: quantum imaginary-time evolution methods

Saturated. The execution plan demotes the varQITE-versus-Motta comparison from a
contribution to a methods subsection because of this literature. The manuscript makes no
novelty claim here.

### III.1 Motta et al. (2020): `to-verify`

- **Establishes.** QITE and QLanczos: imaginary-time evolution reproduced by unitaries whose
  generators are found by a linear solve from measured Pauli expectations. No variational
  optimisation, therefore no barren plateaus.
- **Leaves open.** Generator support and circuit depth grow as correlations spread.
- **Relation.** Route A fallback, `quasarstack/ite/qite_motta.py`.

### III.2 McArdle et al. (2019): `to-verify`

- **Establishes.** Variational imaginary-time evolution by the McLachlan variational
  principle: solve `A theta_dot = C` with A and C from parameter-shift circuit evaluations.
  Circuit depth is constant in imaginary time.
- **Leaves open.** Barren plateaus. The planning documents record a ceiling near L = 10 to
  12 with gradient variance decaying as roughly 0.42^L.
- **Relation.** Route A primary, `quasarstack/ite/varqite.py`.

### III.3 Nishi, Kosugi & Matsushita (2020): `to-verify`

- **Establishes.** Probabilistic implementation of imaginary-time evolution.
- **Relation.** Alternative ITE route; cited to show the method space is well populated.

### III.4 Probabilistic ITE (PITE), 2024: `to-verify`

- **Establishes.** Continued development of probabilistic ITE.
- **Relation.** Evidence that ITE methods on TFIM are an active and crowded area, which is
  the reason this project does not claim them as a contribution.

### III.5 Nonlocal-approximation QITE, *npj Quantum Information*: `to-verify`

- **Relation.** Same as III.4.

### III.6 Quasiprobabilistic ITE: `to-verify`

- **Relation.** Same as III.4.

### III.7 RL-designed ITE ansaetze (2026): `to-verify`

- **Relation.** Same as III.4. The most recent entry; confirms the area is still moving.

---

## Literature IV: quantum algorithms for classical stochastic processes

The literature v3 of the plan ignored entirely, and the reason v4 exists. This is where
Route B connects to provable-complexity results rather than heuristics.

### IV.1 Quantum-enhanced simulation of stochastic processes, *PRX* (2021): `to-verify`

- **Establishes.** Memory and time advantages for quantum simulation of classical
  stochastic processes in specific scenarios.
- **Leaves open.** The advantages are scenario-specific, not general.
- **Relation.** Establishes that the question "does quantum help for a classical stochastic
  process" has a real, non-trivial answer space.

### IV.2 Aghamohammadi, Mahoney & Crutchfield: `to-verify`

- **Establishes.** Extreme quantum advantage in *memory* for generating classical
  spin-chain configurations.
- **Leaves open.** Memory advantage, not time advantage, and not eigenvector extraction.
- **Relation.** A precedent for advantage in a related setting; cited to keep the scope
  claim accurate in both directions.

### IV.3 Orfi & Sels (Flatiron, 2024): `to-verify`

- **Establishes.** No worst-case speedup for quantum-enhanced Markov chain Monte Carlo.
- **Relation.** A negative result that constrains what Route B may claim. Cited directly
  against any temptation to overclaim.

### IV.4 Claudon, Piquemal & Monmarche: "Quantum speedup for nonreversible Markov chains", arXiv:2501.05868, *Nature Communications* 16:10732 (2025), ✅ `verified` 2026-08-09

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

- **Relation to QUASAR: verified by measurement, and it is not what execution plan v4
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

## Literature V: classical tensor-network simulation (the benchmark frontier)

Not one of the four named literatures, but it sets the bar the boundary map is measured
against, so it belongs in this dossier.

### V.1 Tindall, Fishman, Stoudenmire & Sels, *PRX Quantum* 5:010308 (2024), arXiv:2306.14887: `verified 2026-08-15, bibliographic`

- **Establishes.** "Efficient Tensor Network Simulation of IBM's Eagle Kicked Ising
  Experiment". A tensor network whose geometry follows the device lattice, contracted
  approximately by belief propagation, reproduces the 127-qubit kicked-Ising results at
  greater accuracy and precision than the processor and than several other classical methods,
  and extends to long times in the thermodynamic limit. The authors attribute the accuracy to
  the tree-like correlation structure of the wavefunction.
- **Relation.** This is the scope limit, and it is a real one. QUASAR benchmarks against a
  standard matrix-product implementation at the sizes actually tested, **not** against
  lattice-geometry tensor networks with belief-propagation contraction at hundreds of spins.
  A null measured here therefore says nothing about what the state of the art could do, and
  the manuscript must scope the comparison to what was run. The direction of the bias is worth
  stating: bespoke methods of this kind have repeatedly closed claimed advantage gaps, so the
  classical side of any comparison scoped as ours is a lower bound on classical capability.

### V.2 MPO-based spin-glass methods: `unresolved, no citation`

- **Status.** This entry names a body of work without identifying a paper, so there is nothing
  to verify. It is left open rather than marked verified, because marking a placeholder as
  checked is worse than leaving it visibly incomplete.
- **Needed.** Either a specific citation, or removal. The scope limit it was intended to
  support is already carried by V.1.

---

## Gap log

Where the red-team or a reviewer identifies a literature this file misses, it is recorded
here with the date, so the history of what was missed and when is visible.

*(No entries yet.)*
