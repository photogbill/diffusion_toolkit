# ATK Diffusion Toolkit

Diffusion-model and learned-RF tools for ATK, in their own engine repo with
an ATK adapter. **All rights reserved** (see `ATK_DIFFUSION_PLAN.md` D1).
Design only as of 2026-10-08; nothing is built.

## The documents

| File | What it is |
|---|---|
| `ATK_DIFFUSION_PLAN.md` | The plan of record: principles, the RF data foundation (receiver profiles, the sample-rate law, `rf_data\`, the cabled loop, environment profiles, products), every track with its first experiment, the build order, evaluation, licenses, the papers mapped, decisions, change log. |
| `DETECTION_DESIGN.md` | The AI signal detector in detail: three proposers, the cut and classifier, cyclostationary processing, the right-click signal cut, low-SNR escalation and the three classical filters, training, CPU deployment, licenses, classes, labels, evaluation, build steps, where it sits on screen, decisions. |

## Where the rest lives

- **ATK's side** of all of this — sub-tabs, the right-click, the subsystem
  airlock, `rf_data` in the recorder, the hour log, map layers from
  products, Palimpsest's persona hooks — is in
  `D:\Analyst_Toolkit\ATK\FUTURE_PLANS.md`, section dated 2026-10-08.
- **Palimpsest** (the memory that develops; the host-side diffusion loop
  shared with this repo is its plan §8.6):
  `D:\Analyst_Toolkit\Palimpsest\PALIMPSEST_PLAN.md`.
- **Athanor** (the instruments; spikes S7 and S8 for Palimpsest):
  `D:\Analyst_Toolkit\Athanor\ATHANOR_PLAN.md`.
- **Passive radar** inputs are noted in the plan (§4.G) and belong in
  `atkpr`.
- **Data** never lives here: `D:\Analyst_Toolkit\rf_data\` (plan §3).
- **Papers**: `D:\Analyst_Toolkit\new papers\` (+ `additional\`, `additional\more\`), mapped in the plan §9.
"# diffusion_toolkit" 
