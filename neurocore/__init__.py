"""neurocore — the cross-cutting core shared by every thesis problem direction.

Scope is deliberately minimal (see THESIS-PLAN.md). It owns four things plus their support:

  signals.py      the driver / signal bank (DA/ACh/NE/5-HT, the tonic-vs-per-sample distinction and
                  the standardization rules, the loss_fn hook, the head-free novelty drivers)
  gates.py        the rank-K linear gate Gamma = 1 + sum_k m_k P_k, the gain forms and their
                  init-parity semantics, neuron and synapse granularity, gate_l1
  projections.py  the pt5 fixed/learned projection builders (disjoint / shared / learned) and the
                  eval-time-task-id label that must travel with the fixed ones
  ledger.py       the TSV ledger, --resume, ledger-sourced paired deltas, table generation
  controls.py     the `free` dead-gate / RNG-matched-baseline rule, the double-zero-init saddle
                  guards, the task-decodability probe, per-layer |g| reporting
  cost.py         first-class accounting: buffer bytes, extra params vs backbone (with the
                  head-vs-backbone ratio capacity-confound guard), fwd/bwd per step at train and
                  at inference
  buffers.py      the reservoir buffer (what "stored samples" means, for the three memory regimes)

Everything else — the Split-MNIST loops, the pt5 loop machinery, the rejected pt2/pt3 targets, and
the pt6/pt7 study-specific mechanisms — stays frozen in prototype/ and results/ and is extracted
only when a second problem actually calls for it (promotion by future use).

The archived Split-MNIST path is FROZEN and reproduces its ledger numbers unchanged; this package is
a copy-forward of the cross-cutting parts, not a cut. `verify_anchors.py` checks that the two paths
agree bit-exact.
"""
__all__ = ["buffers", "controls", "cost", "gates", "ledger", "projections", "signals", "task_selection", "tuned", "utils"]
