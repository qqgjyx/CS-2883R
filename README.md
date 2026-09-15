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
| Shared upload folder | _TODO: paste the Drive folder link_ |
| **Deck (source of truth)** | <https://claude.ai/artifact/6LC7D89KEjrcYCsbNXsjZ8> |

Grading: presentation 30% / participation 30% / quizzes 10% / final project 30%.

## My assignments

| Date | Paper | My role | Status |
| --- | --- | --- | --- |
| **Wed 2026-09-16** | [RayZer](https://arxiv.org/abs/2505.00702) (arXiv 2505.00702) | 🤖 **AI Interrogator** | prep in [`2026-09-16_rayzer/`](2026-09-16_rayzer/) |

Role is shared with Joseph Firmansyah (row 7 of the sheet); Juntang is row 8. The shared folder
takes **one file per role**, so the two of us need to agree on a split and a single upload.

### Wed 16 Sep, 3:45pm — upload before class

Done:

- [x] Read the paper. Fact sheet: [`NOTES.md`](2026-09-16_rayzer/NOTES.md)
- [x] Ran the interrogation: 15 probes, 2 models, 27 answers, plus the preamble A/B rerun on both
- [x] Verified every claim against the PDF; transcripts, prompts and event logs archived
- [x] Deliverable built: [`AI Interrogator/`](2026-09-16_rayzer/AI%20Interrogator/), 13-page PDF with every prompt, drop in as-is

Left:

- [ ] **Message Joseph Firmansyah** (Canvas Inbox or Ed; no email on file). Text below.
- [ ] **Upload before 3:45pm Wed.** Paste the shared folder link at the top of this file, open it,
      drag `2026-09-16_rayzer/AI Interrogator` in. If Joseph's `AI Interrogator` folder is already
      there, drop the files into it instead. Finder: `open -R "2026-09-16_rayzer/AI Interrogator"`.
- [ ] Optional: rewrite the four assessment lines in your own words. Drafts are in the page, editable.
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
  reserved for presentations. No date in the syllabus; worth asking.
- **If you presenter a paper later** — confirm the paper with Qianqian, fill the role sheet and post
  to Ed, all **at least one week before**. Nothing on the sheet assigns you one yet.

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
```

The deck is **one HTML file** published as an Artifact and updated in place at the same URL.
Present from it. The PDF in the upload folder is printed from it with the appendix open, and
`Cmd-P` does the same. Upload the folder rather than the Artifact link: artifacts are private by
default, and the course guidelines warn about link-sharing permissions.
