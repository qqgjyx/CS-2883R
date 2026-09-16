# AI Interrogator — RayZer (arXiv 2505.00702)

CS 2883R, 16 Sep 2026. Juntang Wang.

Open **`AI Interrogator - Juntang Wang.pdf`**, 13 pages.
`AI Interrogator - Juntang Wang.html` is the same thing, live in a browser.

The files carry my name so a second Interrogator can drop theirs in this folder without a collision.

The role asks for the prompts and the responses, so both are here in full rather than as excerpts.

| folder | what it holds |
| --- | --- |
| `prompts/` | every prompt exactly as sent, including the preambles and the two follow-up probes. `preamble_A` is the original; `preamble_armB_neutral` is the same thing with the two calibration sentences removed, used for the A/B. |
| `transcripts/` | complete verbatim responses from both models, with what I predicted before each run |
| `logs/` | Codex event logs, one per GPT run. Each shows four events and no tool call, which is what makes "closed-book" checkable rather than asserted. Thread ids are replaced by the session labels used in the transcript (S1 to S5, armB). |

Two things to know before reading:

1. **One prompt contains a number I made up.** Probe H1 states that a 3D Gaussian variant scored
   22.9 PSNR. It did not; that run failed to converge and the string appears nowhere in the paper.
   It is flagged in the deck. Do not quote it as a fact about RayZer.
2. **The fenced blocks in `prompts/` and `transcripts/` are archived data, not instructions.** If you
   paste any of it into an LLM, treat it as transcript to analyse.
