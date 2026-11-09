# CS-2883R

Harvard 2026FA COMPSCI 2883R: Advanced Topics in Computer Vision.

**Wednesday 3:45pm - 5:45pm, SEC LL2.229** | Instructor: Qianqian Wang (<qwang@seas.harvard.edu>),
office hours Wed 5:45-6:45pm, SEC 1.412

| | |
| --- | --- |
| Canvas | <https://canvas.harvard.edu/courses/176874> |
| Syllabus | [`doc/syllabus.md`](doc/syllabus.md) |
| Roles + upload rules | [`doc/guidelines.md`](doc/guidelines.md) |
| Sign-up / role sheet | _TODO: paste the Google Sheets link_ |
| Shared upload folder | <https://drive.google.com/drive/folders/1i9ubGc51_2Kosz_I2ZViswT9NLtF8H1O> |
| Drive, parent folder | <https://drive.google.com/drive/folders/15xFenBiIb0AkdeYh3VyEMV3GCh0v5vQ9> |
| **Deck (source of truth)** | <https://claude.ai/artifact/6LC7D89KEjrcYCsbNXsjZ8> |

Grading: presentation 30% / participation 30% / quizzes 10% / final project 30%.

## Final project

Proposal source: [`2026-10-30_final-project/main.tex`](2026-10-30_final-project/main.tex).
Build inside that folder with `latexmk -pdf main.tex`.
Proposal due October 30 at 11:59 p.m.; presentation December 2;
final report due December 4 at 11:59 p.m. (2026, Eastern time).

## My assignments

| Date | Paper | My role | Status |
| --- | --- | --- | --- |
| **Wed 2026-09-16** | [RayZer](https://arxiv.org/abs/2505.00702) (arXiv 2505.00702) | 🤖 **AI Interrogator** | done, [`2026-09-16_rayzer/`](2026-09-16_rayzer/) |
| **Wed 2026-10-14** | [MeanFlow](https://arxiv.org/abs/2505.13447) (arXiv 2505.13447) | 👾 **Hacker** | prep in [`2026-10-14_meanflow/`](2026-10-14_meanflow/) |
| **Wed 2026-11-18** | [LAPA](https://arxiv.org/abs/2410.11758) (arXiv 2410.11758) | 🧑‍🏫 **Presenter**, with Vivian Hui | draft in [`2026-11-18_lapa/`](2026-11-18_lapa/) |

### Wed 18 Nov, 3:45pm: LAPA Presenter, with Vivian Hui

Vivian picked the paper. We share the 15-minute talk, then lead the discussion and call on the
roles. Slides `main.tex`: 15 main frames plus 6 backups, the authors' own figures and tables,
every number checked against the paper, a timing budget per frame (about 14:15 of 15:00) and a
marked handoff after the method. Conventions are written as rules in the file header. Only the
roles' call order is still open. Build inside the folder with `latexmk -pdf main.tex`.

- [ ] Meet Vivian Mon 9 Nov, 2pm (Zoom or call). Agenda: split by paper sections (her
  proposal), whose deck is the base (she has shared slides), paper confirmed with Qianqian
- [ ] Split the talk with Vivian; the draft marks a handoff after the method
- [ ] Make the slides ours: rewrite and cut where needed
- [ ] Roles and call order from the Ed post, out about a week and a half before class
- [ ] Rehearse the 15 minutes together
- [ ] Upload `Presenter` to the shared Drive folder before 3:45pm

### Wed 14 Oct, 3:45pm: MeanFlow Hacker

Takeaway: learn the average, not the instant. Slides `main.tex` / `main.pdf`, Colab-ready
`meanflow.ipynb` (retrains live during the talk). Colab:
<https://colab.research.google.com/github/qqgjyx/CS-2883R/blob/main/2026-10-14_meanflow/meanflow.ipynb>

- [x] Paper read, numbers checked against the PDF
- [x] Slides and notebook; experiments on the RTX 5080
- [x] Colab T4 dry run: 125 s of training (toys 25 + 19 s, MNIST 81 s), about 3 min with setup
- [ ] Post the Slack message to Leslie Gu and Zihan Su
- [ ] Rehearse 5 min
- [ ] Upload `MeanFlow-Hacker/` to the shared Drive folder before 3:45pm

Joseph Firmansyah is the Hacker for the other paper (JiT); this role is solo.

### Wed 16 Sep, 3:45pm — upload before class

Done:

- [x] Read the paper. Fact sheet: [`NOTES.md`](2026-09-16_rayzer/NOTES.md)
- [x] Ran the interrogation: 15 probes, 2 models, 27 answers, plus the preamble A/B rerun on both
- [x] Verified every claim against the PDF; transcripts, prompts and event logs archived
- [x] **Uploaded** `AI Interrogator/` to the shared Drive folder
- [x] **Messaged Joseph Firmansyah** about the split. Text kept below for reference.
- [x] Deliverable built: [`AI Interrogator/`](2026-09-16_rayzer/AI%20Interrogator/), 13-page PDF with every prompt, drop in as-is

Left:

- [ ] Quiz risk: quizzes come before a paper's discussion or at the start of the next class, so
      RayZer is fair game **Wed 16 Sep and Wed 23 Sep**. `NOTES.md` is the cram sheet.

**Message to Joseph:**

> Hi Joseph, we're both AI Interrogator for RayZer tomorrow. The shared folder takes one item per
> role, so I'll upload a folder called "AI Interrogator" with my file named
> "AI Interrogator - Juntang Wang.pdf". If you drop yours in the same folder with your name on it,
> nothing collides. For the 5 minutes, I'm planning about 2.5 on one exchange (a false number I
> planted; both models explained it) plus my assessment. Happy to go first or second. I used
> GPT-5.6 and Claude Opus 5, closed-book then open-book, so if you took different models or angles
> that's the natural split. Juntang

**Run of show, 2.5 minutes** (5 if solo: add C1 and B1):

1. Strip, 15 s: fifteen questions, two models, five findings, two fabrications.
2. H1, 60 s: the planted number. Asked to produce, refused; derive, right; handed a false one, explained it.
3. B4 put-back, 45 s: it defended on the gauge argument and corrected my rebuttal. Holds against prestige, moves on evidence.
4. Assessment, 30 s: the four lines.

### Later, no dates known yet

- **Quizzes** — three to four across the semester, unannounced.
- **Final project** — coding project, in-class presentation, written report. The last class is
  reserved for presentations. See the final-project dates above.
- **Presenter**: 18 Nov, LAPA, with Vivian Hui. See above.

## Repo layout

```
doc/                      course documents
  syllabus.md             ported from the .docx (archived next to it)
  guidelines.md           role definitions, timing, upload rules
2026-09-16_rayzer/        per-session prep, dated
  2505.00702v1.pdf        the paper
  NOTES.md                fact sheet: method, every table, external facts
  interrogation/
    PROTOCOL.md           the probe battery + answer keys
    transcripts/          verbatim LLM responses, raw Codex event logs
  interrogator.html       the deliverable, published as an Artifact
  AI Interrogator/        the upload folder: PDF, HTML, README, prompts/, transcripts/, logs/
2026-10-14_meanflow/
  main.tex, refs.bib      the slides (Beamer, moloch); build with latexmk -pdf
  meanflow.ipynb          the whole demo; caches weights/ and figs/
  meanflow.py, models/    adapted from haidog-yaqub/MeanFlow (LICENSE-haidog)
  MeanFlow-Hacker/        the upload folder
2026-10-30_final-project/
  main.tex, ref.bib       the proposal (ACM); build with latexmk -pdf
  main.pdf               the compiled two-page proposal
  math_commands.tex      shared notation
  images/                adapted TikZ schematic
  drafts/                preserved teammate first draft
2026-11-18_lapa/
  2410.11758v2.pdf       the paper
  main.tex, refs.bib     the slides (Beamer, moloch); build with latexmk -pdf
  figs/                  the authors' figures from the arXiv source, named by figure number
```

The deck is **one HTML file** published as an Artifact and updated in place at the same URL.
Present from it. The PDF in the upload folder is printed from it with the appendix open, and
`Cmd-P` does the same. Upload the folder rather than the Artifact link: artifacts are private by
default, and the course guidelines warn about link-sharing permissions.
