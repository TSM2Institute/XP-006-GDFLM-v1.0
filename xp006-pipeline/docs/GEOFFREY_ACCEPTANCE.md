# XP-006 GDFLM v1.0 — Formal Acceptance

**Pipeline:** XP-006 Gradient-Dominant Field Lensing Model v1.0
**Model tested:** K(x) = ρ̂ + |∇T̂| + |∇ρ̂| under equal weighting (α=β=γ=1.0) and zero free parameters
**Verdict submitted to Geoffrey:** FAIL (2 of 4 pre-registered criteria met)
**Verdict review:** Gate 4 PASS (Claude + Grok unanimous, 19 April 2026)
**Manifest SHA-256:** `2faf4b6c5e1c2ee40000d6f4a3b3811df151aff8f582036b2d75a97b708ccfc4`

---

## Geoffrey E. Thwaites — Formal Response (verbatim, 19 April 2026)

> Graham,
>
> **ACCEPT** — verdict stands as FAIL (2/4).
>
> The result is accepted as a valid negative outcome for the specific model tested:
> K(x) = ρ̂ + |∇T̂| + |∇ρ̂| under equal weighting and zero free parameters.
>
> **Clarification (important for record):**
>
> This result excludes this specific additive gas + gradient formulation only. It does not establish that a non-gas mass tracer is required, nor does it test the TSM2.1 density-gradient propagation (refractive) lensing model.
>
> As defined in the core equations, lensing in TSM2.1 is governed by propagation through a density-dependent refractive index:
>
> n(r) = n₀ + k_TSM · ρ_H(r)
> θ_bend = ∫ ∇n(r) dr
>
> This is a path-integral propagation problem, not a static additive field reconstruction. XP-006 therefore does not challenge the TSM2.1 lensing mechanism.
>
> **Positive result to retain:**
>
> The +48% χ²/dof improvement confirms that gradient terms (|∇T|, |∇ρ|) contain real physical signal and should be retained in future formulations.
>
> **Programme position:**
>
> Record XP-006 as FAIL (model class excluded: zero-parameter additive gas/gradient reconstruction).
>
> **Next step (directional, not a modification of this test):**
>
> Proceed to a propagation-based test aligned with the TSM2.1 lensing equation rather than further additive field variants.
>
> No post-hoc adjustment to XP-006.
>
> Regards,
> Geoff

---

## Formal Scope of the XP-006 Falsification

Per Geoffrey's acceptance, the falsification boundary is:

**Excluded by this result:** The specific class of lensing models where κ is reconstructed as an equal-weighted, zero-parameter, additive sum of min-max-normalised gas-derived fields (gas density + gas density gradient + gas temperature gradient) on the Bullet Cluster.

**Not tested by this result:**
- The TSM2.1 density-gradient propagation (refractive) lensing model, which is a path-integral computation θ_bend = ∫∇n(r) dr through a refractive medium, mathematically and physically distinct from additive static field reconstruction.
- SKIN-a-CAT v1.2's k_TSM × N_HI formulation, which uses a different lensing mechanism entirely.
- Non-equal-weighted variants of the additive formulation (α, β, γ as fitted parameters).
- GDFLM on relaxed (non-merging) clusters where gas and mass are spatially co-located.

## Retained Positive Result

The χ²/dof improvement of +48.33% over the X-ray surface brightness baseline is recorded as a genuine physical finding: gradient terms |∇T| and |∇ρ| carry lensing-relevant information beyond raw gas density. This signal should be retained in future formulations.

## Programme Record Entry

**XP-006 GDFLM v1.0 — FAIL (2/4). Model class excluded: zero-parameter additive gas/gradient reconstruction on the Bullet Cluster.**

## Next Direction (Programme-Level, Not XP-006 Modification)

Per Geoffrey: proceed to a propagation-based test aligned with the TSM2.1 lensing equation. This would require designing a new pipeline from scratch, not modifying XP-006.
