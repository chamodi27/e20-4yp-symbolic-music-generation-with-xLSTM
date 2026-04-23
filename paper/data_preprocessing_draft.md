# Draft: Data Preprocessing Section — FINAL
**Paper:** Evaluating xLSTM for Long-Form Symbolic Music Generation  
**Section:** §III-A → `\subsubsection{Pre-processing}`  
**Status:** FINAL — 2026-04-16  
**Source of numbers:** `lmd_preprocessed_3/meta/` (split files), `lmd_preprocessed_3/README.md` (REMIGEN token stats)

---

## Confirmed Dataset Statistics

| Split | Songs | Total Tokens | Avg Tokens/Song |
|-------|------:|------------:|----------------:|
| Train | 19,521 | 265.2M | 13,575 |
| Validation | 2,440 | 33.2M | 13,611 |
| Test | 2,441 | 33.1M | 13,572 |
| **Total** | **24,402** | **331.5M** | — |

---

## LaTeX: `\subsubsection{Pre-processing}` (replaces line 161 placeholder)

```latex
\subsubsection{Pre-processing}
\label{sec:preprocessing}

We follow the MuseFormer preprocessing methodology~\cite{yu2022museformer} to obtain a
clean and structurally consistent MIDI corpus from the LMD.
Starting from the 29,940-file curated list provided by MuseFormer (itself a filtered
subset of the full LMD), we apply a five-stage pipeline that produces
\textbf{24,402 MIDI files} ready for tokenisation.
Our dataset is therefore a strict subset of the MuseFormer split, obtained by
independently re-executing the same preprocessing rules.

\paragraph{Stage~1 --- MIDI Parsing.}
Every candidate MIDI file is parsed using the \texttt{miditoolkit} Python library.
Per-file metadata (time signatures, tempo range, pitch range, note count, track count,
and empty-bar statistics) is recorded in a manifest CSV that drives all downstream
stages.
Files that fail to parse are discarded.

\paragraph{Stage~2 --- Melody Track Detection.}
We use \texttt{midi-miner}~\cite{midiminer}, a machine-learning-based track-role
classifier, to assign roles (\emph{melody}, \emph{bass}, \emph{drum}, or
\emph{other}) to each instrument track.
Files for which no melody track is detected are excluded.

\paragraph{Stage~3 --- Instrument Compression to Six Tracks.}
Following MuseFormer, each file is reduced to exactly six canonical instrument
channels: \texttt{square\_synth} (melody, GM~program~80),
\texttt{piano} (programs~0--7), \texttt{guitar} (programs~24--31),
\texttt{string} (programs~40--51, 88--95), \texttt{bass} (programs~32--39),
and \texttt{drum} (percussion channel).
When multiple source tracks compete for the same target channel, the top-$K$
tracks by note count are merged ($K{=}2$).
Per-track cleanup is then applied:
melody and bass are enforced to be monophonic (one note per onset);
piano, guitar, and string channels have polyphony capped (piano~$\leq 10$
simultaneous notes; guitar/string~$\leq 6$); and exact duplicate notes are removed.

\paragraph{Stage~4 --- MIDI Normalisation.}
Each compressed MIDI file is routed through MuseScore Studio~4~\cite{musescore}
in headless mode to quantise note timings to a consistent grid, enforce a uniform
tempo and time-signature representation, and resolve common encoding edge cases
(e.g.,~overlapping notes and sustain-pedal artefacts).

\paragraph{Stage~5 --- Quality Filtering and Pitch Normalisation.}
We apply the filtering rules from MuseFormer~\cite{yu2022museformer} to remove
low-quality and musically degenerate content; these rules are summarised in
Table~\ref{tab:filter_rules}.
Following filtering, all surviving files are transposed to a canonical key using the
Krumhansl--Kessler pitch-class correlation method~\cite{krumhansl1990cognitive}:
major-key pieces are transposed to C~major and minor-key pieces to A~minor, with the
shift constrained to $\pm 6$~semitones and octave adjustments applied as needed to
keep all notes within MIDI~21--108 (A0--C8).
```

---

## LaTeX: Quality Filtering Table (insert after Stage 5 paragraph, before `\subsubsection{Token Encoding}`)

```latex
\begin{table}[htbp]
\centering
\caption{Quality filtering rules applied in Stage~5 (following MuseFormer~\cite{yu2022museformer})}
\label{tab:filter_rules}
\begin{tabular}{ll}
\toprule
\textbf{Criterion} & \textbf{Rule} \\
\midrule
Time signature     & Only 4/4 retained \\
Minimum tracks     & $\geq 2$ non-empty instrument tracks \\
Melody presence    & \texttt{square\_synth} track required \\
Tempo range        & 24--200 BPM \\
Pitch range        & MIDI 21--108 (A0--C8) \\
Note duration      & $\leq 16$ beats per note \\
Empty bars         & $\leq 3$ consecutive empty bars \\
Degenerate content & Dropped if all notes share the same pitch or duration \\
Duplicates         & Removed via content fingerprint \\
\bottomrule
\end{tabular}
\end{table}
```

---

## LaTeX: Updated `tab:dataset_stats` (replace lines 168–184)

```latex
\begin{table}[htbp]
\centering
\caption{Dataset Statistics After REMIGEN Encoding}
\label{tab:dataset_stats}
\begin{tabular}{lrrr}
\toprule
\textbf{Split} & \textbf{Songs} & \textbf{Total Tokens} & \textbf{Avg.\ Tokens/Song} \\
\midrule
Train      & 19,521 & 265.2M & 13,575 \\
Validation &  2,440 &  33.2M & 13,611 \\
Test       &  2,441 &  33.1M & 13,572 \\
\midrule
\textbf{Total} & \textbf{24,402} & \textbf{331.5M} & \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Fragment to update at line 339 (perplexity table footnote)

**Change:** "our dataset retains 45,043"  
**To:** "our preprocessed dataset comprises 24,402 files after independently applying the same filtering and normalisation steps"

---

## New bibliography entries needed

```bibtex
\bibitem{midiminer}
S.~Lv, Z.~Jiang, and C.~Jin, ``MIDI Track Separation via Machine Learning,''
GitHub repository, 2022. [Online]. Available:
\url{https://github.com/ruiguo-erasmus/midi-miner}

\bibitem{musescore}
MuseScore, ``MuseScore Studio 4,'' 2024. [Online]. Available:
\url{https://musescore.org}

\bibitem{krumhansl1990cognitive}
C.~L.~Krumhansl, \emph{Cognitive Foundations of Musical Pitch}.
Oxford University Press, 1990.
```

---

## Remaining TODO

- [ ] Verify midi-miner citation (check repo README for preferred citation / paper)
