# References

Notes on the literature this work sits on, grouped by the four areas it draws from: the
correspondence between mutation-selection dynamics and spin chains, classical algorithms for
the quasispecies, quantum imaginary-time evolution, quantum algorithms for classical stochastic
processes, and classical tensor-network simulation. Each entry gives the citation, what the
paper establishes, what it leaves open, and how this project relates to it.

Six entries changed when I went back and read them rather than trusting how they had been
recorded in the planning documents:

| Entry | What changed |
|---|---|
| II.1 | The efficient class is a strict subset of Baseline B's, so no boundary cell is misclassified. The `(L+1)` reduction is Swetina-Schuster 1982, not this paper, which cites it for exactly that |
| I.4 | Credited with a pairwise-epistatic Ising Hamiltonian it does not contain; it is an introductory review |
| III.3 and III.5 | The same paper listed twice, and described as probabilistic ITE when it is a nonlocal approximation. Merged |
| III.4 | Dated to a year matching no paper in the line; the line runs 2021 to 2025 |
| IV.1 | Wrong title |
| II.3 | Treats the Moran model, not Wright-Fisher as implied |

Two of those, II.1 and I.4, were misattributions that would have reached a referee. Both came
from a reference recorded out of a planning document rather than out of the paper.

Dates for when each entry was last checked are at the end.

---

## Literature group I: the evolution to spin-chain correspondence

The mathematical bridge the project stands on. This literature is mature. Nothing here is
novel here, and the manuscript says so in the introduction rather than in a limitations
paragraph.

### I.1 Leuthäusser, *J. Chem. Phys.* 84(3):1884 (1986); *J. Stat. Phys.* 48:343 (1987)

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

### I.2 Baake, Baake & Wagner, *Phys. Rev. Lett.* 78:559 (1997), erratum PRL 79:1782

- **Establishes.** "Ising quantum chain is equivalent to a model of biological evolution".
  A sequence-space model of mutation and selection is equivalent to an Ising quantum chain,
  with three representative fitness landscapes solved exactly by statistical mechanics.
- **Leaves open.** Exactness of the mapping, not tractability. No spectral gap treatment, no
  oracle model, no resource scaling.
- **Relation.** The specific correspondence QUASAR implements. Erratum PRL 79:1782 confirmed
  to exist and must be consulted before any convention is copied from the original.

### I.3 Saakian & Hu, *Phys. Rev. E* 69:021913 and 69:046121 (2004), arXiv:cond-mat/0402212

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

### I.4 Jain & Krug, arXiv:q-bio/0508008 (2005)

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

### I.5 Park & Deem, *J. Stat. Phys.*, arXiv:q-bio/0607012 (2006)

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

### II.1 Dixit, Srivastava & Vishnoi, arXiv:1203.1287 (2012)

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

### II.1a Swetina & Schuster, *Biophys. Chem.* 16:329-345 (1982)

- **Establishes.** For class-invariant landscapes the quasispecies is the leading eigenvector of
  an `(L+1) x (L+1)` matrix rather than a `2^L x 2^L` one. This is the reduction Baseline B's
  permutation-symmetric branch implements.
- **Status.** Verified indirectly: arXiv:1203.1287 attributes the reduction to it explicitly and
  by reference number. The primary source has not been read, so the entry is marked by citation
  rather than read in full, and the manuscript should not attribute anything to it beyond the
  `(L+1)` reduction until someone reads it.
- **Relation.** The correct citation for Baseline B, `quasarstack/classical/exact_class.py`.

### II.2 Dalmau, arXiv:1403.6951 (2014) and arXiv:1712.00279 (2017)

- **Establishes.** "The distribution of the quasispecies for the Wright-Fisher model on the
  sharp peak landscape" and "The Wright-Fisher model for class-dependent fitness landscapes".
  Exact distributions for the finite-population Wright-Fisher quasispecies in the long-chain
  regime, on the two landscape classes named.
- **Leaves open.** Closed forms for structured landscapes only.
- **Relation.** A second analytic check on the finite-population baseline, and a further reason
  no advantage claim is available on the easy landscapes. A companion result for the
  Galton-Watson process, arXiv:1411.4488, exists and is not used here.

### II.3 Cerf & Dalmau, *Stoch. Proc. Appl.* 126:1681 (2016); monograph, Springer (2022)

- **Establishes.** The 2016 paper is "The distribution of the quasispecies for a Moran model on
  the sharp peak landscape", so it treats the **Moran** model rather than Wright-Fisher; the
  entry previously implied otherwise. The monograph, *The Quasispecies Equation and Classical
  Population Models* (Probability Theory and Stochastic Modelling 102, Springer 2022), carries
  full proofs for the Wright-Fisher model and exact formulas in the long-chain regime, on the
  sharp peak and on class-dependent landscapes.
- **Relation.** As II.2. Cite the monograph for the Wright-Fisher results and the 2016 paper
  only for the Moran model.

---

## Literature III: quantum imaginary-time evolution methods

Saturated. The execution plan demotes the varQITE against Motta comparison from a contribution
to a methods subsection because of this literature. No novelty is claimed here.

### III.1 Motta et al., *Nature Physics* 16:205 (2020), arXiv:1901.07653

- **Establishes.** "Determining eigenstates and thermal states on a quantum computer using
  quantum imaginary time evolution". Introduces QITE and quantum Lanczos as analogues of the
  classical algorithms, positioned explicitly against phase estimation, which needs deep
  circuits and ancillas, and against variational algorithms, which add high-dimensional
  classical optimisation.
- **Leaves open.** Generator support and circuit depth grow as correlations spread.
- **Relation.** Route A fallback, `quasarstack/ite/qite_motta.py`.

### III.2 McArdle et al., *npj Quantum Information* 5:75 (2019), arXiv:1804.03023

- **Establishes.** "Variational ansatz-based quantum simulation of imaginary time evolution".
  A hybrid variational algorithm for imaginary-time evolution on shallow circuits, applied to
  molecular ground states.
- **Relation.** Route A primary, `quasarstack/ite/varqite.py`.
- **Correction.** The previous entry attributed a gradient-variance decay of "roughly 0.42^L"
  to the planning documents. **G-R.9 measured it in this project** and the base is 0.535 to
  0.556 across six landscape and statistic combinations, 0.549 for the check statistic. The
  0.42 figure is superseded by measurement and should not be cited.

### III.3 Nishi, Kosugi & Matsushita, *npj Quantum Information* 7:85 (2021), arXiv:2005.12715

**Two errors, both found on reading.** The entry described this as "probabilistic
implementation of imaginary-time evolution", which is a different line of work by an
overlapping group, entry III.4. And the former entry III.5, "Nonlocal-approximation QITE, npj
Quantum Information", was **the same paper listed a second time**, so the dossier counted one
result as two.

- **Establishes.** "Implementation of quantum imaginary-time evolution method on NISQ devices:
  Nonlocal approximation". QITE suffers deep circuits on NISQ hardware; the paper introduces
  two approximations under a nonlocality condition, extended LA and nonlocal approximation, to
  reduce depth.
- **Relation.** Alternative ITE route, cited to show the method space is populated. The
  duplicate is removed rather than left inflating the count.

### III.4 Kosugi, Nishiya, Nishi & Matsushita, probabilistic imaginary-time evolution, *Phys. Rev. Research* 4:033121 (2022), arXiv:2111.12471

- **Establishes.** A distinct non-variational line from an overlapping group, using measurement
  to realise a non-unitary operation with a single ancilla and forward and backward real-time
  evolution as black boxes. Origin: Kosugi, Nishiya, Nishi & Matsushita, *Phys. Rev. Research*
  4:033121 (2022), arXiv:2111.12471. Continuations include amplitude-amplification
  acceleration, arXiv:2212.13816; optimal scheduling, *Phys. Rev. Research* 5:043048 (2023);
  and device implementations, arXiv:2504.04958 (2025).
- **Correction.** The entry was dated "2024", which matches no paper in the line. The line runs
  from 2021 to 2025.
- **Relation.** Evidence that ITE methods are an active and crowded area, which is why this
  project claims no contribution there.

### III.5 Ray et al., quasiprobabilistic imaginary-time evolution, *Quantum Inf. Comput.* 26(1):89 (2026), arXiv:2505.06343

- **Establishes.** "Quasiprobabilistic imaginary-time evolution on quantum computers".
  Decomposes a Trotterised imaginary-time evolution into a probabilistic linear combination of
  operations, in the manner of probabilistic error cancellation. Needs no ancillas and is
  noise-resilient without further mitigation. Demonstrated on an 8-qubit Heisenberg chain in
  simulation and on 2 qubits of hardware.
- **Relation.** As III.4. Renumbered after the duplicate above was removed.

### III.6 Suzuki & Watabe, automated ITE circuit design by deep reinforcement learning, arXiv:2604.07951 (2026)

- **Establishes.** Double deep Q-networks design variational imaginary-time evolution circuits
  as a multi-objective problem over energy and circuit complexity, reporting roughly 37% fewer
  checks and 43% less depth than a hardware-efficient ansatz on Max-Cut, and reaching the
  full-CI limit for molecular hydrogen on a shallower circuit.
- **Relation.** As III.4. The most recent entry, and it confirms the area is still moving.

---

## Literature IV: quantum algorithms for classical stochastic processes

Where Route B connects to provable-complexity results rather than to heuristics.

### IV.1 Korzekwa & Lostaglio, *Phys. Rev. X* 11:021019 (2021), arXiv:2005.02403

- **Establishes.** Three scenarios in which memory or time advantages arise when simulating
  classical stochastic processes by quantum dynamics, including quantum memoryless dynamics
  simulating classical processes that provably require memory.
- **Correction.** The entry titled this "Quantum-enhanced simulation of stochastic processes".
  The actual title is above.
- **Leaves open.** The advantages are scenario-specific, not general.
- **Relation.** Establishes that the question of whether quantum helps for a classical
  stochastic process has a real and non-trivial answer space.

### IV.2 Aghamohammadi, Mahoney & Crutchfield, *Sci. Rep.* 7:6735 (2017), arXiv:1609.03650

- **Establishes.** "Extreme Quantum Advantage when Simulating Classical Systems with Long-Range
  Interaction". For the Dyson one-dimensional Ising chain the advantage grows without bound
  with interaction range: the most memory-efficient classical algorithm known requires infinite
  memory where a quantum simulator requires finite memory.
- **Leaves open.** **Memory** advantage, not time, and not eigenvector extraction. The related
  rare-event sampling result is *Phys. Rev. X* 8:011025 (2018).
- **Relation.** A precedent for advantage in a related setting, cited to keep the scope claim
  accurate in both directions.

### IV.3 Orfi & Sels, *Phys. Rev. A* 110:052414 (2024), arXiv:2403.03087

- **Establishes.** "Bounding the speedup of the quantum-enhanced Markov-chain Monte Carlo
  algorithm". No speedup over classical sampling on a worst-case unstructured sampling problem,
  by an upper bound on the Markov gap that rules out a speedup for **any unital** quantum
  proposal.
- **Relation.** A negative result that constrains what Route B may claim, cited directly
  against any temptation to overclaim. Note the scope of the bound: worst case, unstructured,
  unital proposals. It does not forbid advantage on structured instances, and the manuscript
  should not present it as more than it is.

### IV.4 Claudon, Piquemal & Monmarché, *Nature Communications* 16:10732 (2025), arXiv:2501.05868

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

- **Consequence for Route B.** See `notes.md`. Route B is not dead, but it
  cannot be built on this reference as planned, and the plan's framing needs correcting
  before WP2 starts.

---

## Literature V: classical tensor-network simulation (the benchmark frontier)

Not one of the four named literatures, but it sets the bar the boundary map is measured
against, so it belongs in this dossier.

### V.1 Tindall, Fishman, Stoudenmire & Sels, *PRX Quantum* 5:010308 (2024), arXiv:2306.14887

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

### V.2 MPO-based spin-glass methods

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

---

## When each entry was last checked

`bibliographic` means title, authorship, venue and the claim summarised here were
confirmed against the source or its abstract. `read in full` means the paper was read
end to end. One entry names a body of work without identifying a paper, so there is
nothing to check against.

| Entry | Checked |
|---|---|
| I.1 | verified 2026-08-15, bibliographic |
| I.2 | verified 2026-08-15, bibliographic |
| I.3 | verified 2026-08-15, bibliographic |
| I.4 | verified 2026-08-15, corrected |
| I.5 | verified 2026-08-15, bibliographic |
| II.1 | verified 2026-08-14 |
| II.1a | verified 2026-08-14 by citation |
| II.2 | verified 2026-08-15, bibliographic |
| II.3 | verified 2026-08-15, bibliographic |
| III.1 | verified 2026-08-15, bibliographic |
| III.2 | verified 2026-08-15, bibliographic |
| III.3 | verified 2026-08-15, corrected and merged |
| III.4 | verified 2026-08-15, corrected |
| III.5 | verified 2026-08-15 |
| III.6 | verified 2026-08-15 |
| IV.1 | verified 2026-08-15, title corrected |
| IV.2 | verified 2026-08-15, bibliographic |
| IV.3 | verified 2026-08-15, bibliographic |
| IV.4 | verified 2026-08-09 |
| V.1 | verified 2026-08-15, bibliographic |
| V.2 | unresolved, no citation |
