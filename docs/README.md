# docs/

| File | Content |
|---|---|
| `installation.md` | Building the image on the VM, and the authoring setup on the laptop |
| `quickstart.md` | First gate run, start to finish |
| `theory.md` | The correspondence, the dictionary, the Hamiltonian, and the WP1 derivations |
| `methods_route_a.md` | Imaginary-time evolution: varQITE and Motta-QITE |
| `methods_route_b.md` | QSVT Perron-vector extraction: block encoding, phase factors, resources |
| `baselines.md` | The three classical baselines and the honest role of each |
| `validation.md` | The gate system, the conventions, and the failure modes each one locks |
| `compute_layout.md` | Laptop, VM, and Drive archive: what runs where and what moves how |
| `reproducing_the_paper.md` | Figure by figure, the exact command that regenerates it |

Every derivation stated in `docs/theory.md` is referenced from the docstring of the code
that relies on it. A property asserted in code but not derived here fails gate G-1.3.
