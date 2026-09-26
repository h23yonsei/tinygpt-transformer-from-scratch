# TinyGPT: Building a Transformer Language Model Step by Step

[![checks](https://github.com/h23yonsei/tinygpt-transformer-from-scratch/actions/workflows/checks.yml/badge.svg)](https://github.com/h23yonsei/tinygpt-transformer-from-scratch/actions/workflows/checks.yml)

A character-level language model built up in six stages — from a Bigram counting model to
a Multi-Head Transformer — with each stage introduced to fix a concrete limitation of the
one before it, and then a final implementation of my own on a different dataset, which reaches a
held-out loss of 1.36 at its best epoch.

Notebooks 01–06 are the course materials from ECO4126 at Yonsei University, Spring 2026, which
follow Andrej Karpathy's [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture
and his [`makemore`](https://github.com/karpathy/makemore) framework; the overview below
summarizes them. `notebook_06_h23yonsei.ipynb` is my own work: the same Transformer
structure on a different dataset, with six modifications described in
[section 3](#3-tinygpt-final-implementation--personal-project).

---

## Contents

1. [Overview](#1-overview)
2. [Project structure](#2-project-structure)
3. [TinyGPT final implementation — personal project](#3-tinygpt-final-implementation--personal-project)
4. [Results across all stages](#4-results-across-all-stages)
5. [Running and checking the notebooks](#5-running-and-checking-the-notebooks)
6. [Credits](#6-credits)

---

## 1. Overview

Each course notebook is built around one structural limitation of the one before it, so the
architecture grows for a reason rather than arriving fully formed. All of them share Karpathy's
`makemore` scaffolding: a character-level tokenizer, an autoregressive sampling loop, and a context
length (`block_size`) that grows from 1 to 64 across the stages.

| Notebook | Model | What it adds | What it still cannot do |
| --- | --- | --- | --- |
| 01 | Bigram | Tokenizer and sampling loop; an `nn.Embedding` lookup, which computes the same as a one-hot vector times a matrix | Sees only the previous character, and no two characters share anything |
| 02 | MLP with embeddings | Dense embeddings, in which similar characters learn to sit close together | Its input size is fixed at `block_size × emb_dim`, so the context cannot grow |
| 03 | The same MLP on Shakespeare | Only the data and the hyperparameters change; the architecture carries over | The same fixed context; the vocabulary is rebuilt for the new text |
| 04 | Positional-embedding LM | Token plus position embeddings, predicting the next character at every position in one pass | Positions never exchange information, so the loss stalls |
| 05 | Single-head self-attention | Weights computed from the input on every pass, with a causal mask so the model cannot read ahead | One head and one layer |
| 06 | TinyGPT | Four heads, a feed-forward network, residual connections, Pre-LN LayerNorm, dropout, four stacked blocks | (the base of my own notebook, section 3) |

Notebooks 01–02 train on a list of names, 03–06 on Tiny Shakespeare.

---

## 2. Project structure

```text
├── notebook_01.ipynb              # Stage 1: Bigram                  (course material)
├── notebook_02.ipynb              # Stage 2: MLP + Embedding         (course material)
├── notebook_03.ipynb              # Stage 3: Domain shift            (course material)
├── notebook_04.ipynb              # Stage 4: Positional embedding    (course material)
├── notebook_05.ipynb              # Stage 5: Single-Head Attention   (course material)
├── notebook_06.ipynb              # Stage 6: Multi-Head Transformer  (course material)
├── notebook_06_h23yonsei.ipynb    # Final: my TinyGPT, Oz dataset    (my work)
├── input.txt                      # The Wonderful Wizard of Oz, training data
├── requirements.txt               # Python dependencies (PyTorch, NumPy, Jupyter tooling)
├── tools/
│   └── check_notebooks.py         # Checks the stored losses against section 4
└── README.md
```

---

## 3. TinyGPT final implementation — personal project

`notebook_06_h23yonsei.ipynb` takes the Transformer structure from Notebook 06 and is my
own implementation, with the training data swapped from Shakespeare to
**The Wonderful Wizard of Oz**.

### 3.1. Training data: preparing `input.txt`

| Item | Detail |
| --- | --- |
| Source | [Project Gutenberg — *The Wonderful Wizard of Oz* by L. Frank Baum](https://www.gutenberg.org/ebooks/55) |
| Format | Plain text UTF-8 (`.txt`) |
| Final size | 4,582 lines / ~210 KB |

The raw Project Gutenberg `.txt` wraps the novel in legal notices, license text, and
metadata blocks. Training on it as-is would teach the model the copyright-boilerplate
pattern along with the prose, so two sections were removed by hand:

1. **Header** — everything from the first line through the
   `*** START OF THE PROJECT GUTENBERG EBOOK ***` separator
2. **Footer** — the entire license and copyright block after the
   `*** END OF THE PROJECT GUTENBERG EBOOK ***` separator

After editing, `input.txt` contains only the novel body: it starts at the title
"The Wonderful Wizard of Oz" and ends on the last line of Chapter XXIV
("And oh, Aunt Em! I'm so glad to be at home again!").

### 3.2. Architecture hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `vocab_size` | 68 | Unique characters in the Oz text |
| `emb_dim` | 128 | Token + position embedding dimension |
| `block_size` | 64 | Context window length |
| `n_heads` | 4 | Number of attention heads |
| `head_size` | 32 | Dimension per head (= 128 / 4) |
| FFN hidden | 512 | Feed-forward intermediate dimension (= 4 × 128) |
| `n_blocks` | 4 | Stacked Transformer blocks |
| `dropout` | 0.1 | Regularization rate |
| Optimizer | AdamW | `lr=3e-4`; everything else left at the PyTorch defaults |
| Batch size | 64 | |
| `max_steps` | 300 | Max training steps per epoch |
| Epochs | 100 | |

### 3.3. Architecture flow

```text
Input (B, T)
    ↓
Token Embedding (68 → 128)  +  Position Embedding (64 → 128)
    ↓
[Transformer Block × 4]
  each block:
    Pre-LayerNorm → Multi-Head Attention (4 heads) → residual
    Pre-LayerNorm → Feed-Forward (128→512→128) → residual
    ↓
Final LayerNorm
    ↓
LM Head Linear (128 → 68)
    ↓
Output logits (B, T, 68)
```

### 3.4. What I changed from the Notebook 06 baseline

Six modifications on top of the Notebook 06 structure:

| # | Change | Location | Reason |
| --- | --- | --- | --- |
| 1 | `ReLU` → `GELU` | `FeedForward` | The standard activation since GPT-2. Handles negative inputs smoothly, improving gradient flow |
| 2 | Weight tying | `TinyGPT.__init__` | `lm_head.weight = token_embedding.weight`, so the two share one matrix: 68 × 128 fewer parameters |
| 3 | Gradient clipping | `train_one_epoch` | `clip_grad_norm_(max_norm=1.0)` to prevent gradient explosion |
| 4 | Train / val split | Dataset setup | The first 80 % of the text trains, the last 20 % is held out, and the context windows are built inside each part so no window straddles the boundary **(introduced in this version)** |
| 5 | Cosine LR scheduler | Training loop | `CosineAnnealingLR` to stabilize convergence late in training |
| 6 | Temperature + top-k sampling | `sample_gpt` | Make generation diversity and quality tunable |

### 3.5. Training convergence

Trained on a Colab T4, 100 epochs of 300 steps each.

| Epoch | Train loss | Val loss | Note |
| --- | --- | --- | --- |
| 0 | 6.4474 | 2.4425 | Mean over the 300 steps of the first epoch; a uniform guess over 68 characters would be ln 68 = 4.22, and the validation loss after the epoch is already 2.44 |
| 10 | 1.5768 | 1.5533 | Both still falling together |
| 35 | 1.1436 | **1.3596** | Best validation loss |
| 50 | 1.0525 | 1.3811 | Validation has turned back up |
| 99 | **0.9577** | 1.4170 | Final |

The first epoch's mean starts above ln 68 because of weight tying: the output layer is the token
embedding table, initialized from N(0, 1), and the residual stream still carries each character's
own embedding, so the untrained model predicts every character to be followed by itself with near
certainty (a loss of 80–89 per batch over eight seeds, against 4.35 untied). The first steps of
training undo that.

Validation bottoms out at epoch 35 and drifts up by about 0.06 while the training loss keeps
falling: from there the model memorizes the book. Oz is about 205 k characters, small enough for a
4-block, 128-dimensional model to overfit quickly, so this run should stop around epoch 35; more
data or stronger regularization would help, more epochs would not.

The split is contiguous (the first 80 % of the text trains, the last 20 % is held out, and windows
are built inside each part), so no overlapping window leaks training text into validation.

### 3.6. Sample output ("Dorothy" prompt, temperature=0.8, top_k=40)

```text
Dorothy crows pushed the blew came to the two ark Oz, so that the ground was while he had secret to her a beautiful
breasts back not came burning to this, for she knew go, what hurt
every discovered with straw, and and then he panted him up in their brilliancy.

“I’m terribly to all be a very when it. What she never seemed to do me to come back to
Kansas—but if he is goes, and when nothing a little girl, who was
was so just the Wicked Witch of the West, and then they tamed as
Dorothy could not be carried him to do this.

“If you wear the fear!” she replied.

“Oh, yes,” said the Tin Woodman. “You must tremble
away him, so she will protect care not hurt,” said the Lion.

“When I shall be have the rest until they come dome from the Wicked
Witch and the Tin Woodman.

“Where is a great many?” asked Dorothy.

“And I shall get my brains in,” answered the Tin Woodman. “But we can grant the
dish, dressed in places and head to wait out the old woman. At first the
decided her paint he felt in a great re
```

Character names, quoted dialogue and sentence shape come out legibly; the grammar does not
hold together across a whole sentence. The loss is not comparable with the Shakespeare
notebooks — different text, different vocabulary size, different number of tokens.

---

## 4. Results across all stages

| Notebook | Model | Dataset | Initial loss | Final loss (train) | Final loss (val) | Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Bigram | Names | ~2.5 | ~2.5 | — | 20 |
| 02 | MLP | Names | 2.3573 | 2.2642 | — | 30 |
| 03 | MLP | Shakespeare | 2.5999 | 1.8584 | — | 10 |
| 04 | Positional LM | Shakespeare | ~3.06 | ~2.46 | — | 100 |
| 05 | Single-Head Attention | Shakespeare | 2.9403 | 2.2439 | — | 100 |
| 06 | TinyGPT | Shakespeare | 2.6609 | 1.3316 | — | 100 |
| **06_h23yonsei** | **TinyGPT (personal project)** | **Oz** | **6.4474** | **0.9577** | **1.4170** | **100** |

Notebook 04's loss stalls higher than the surrounding stages because it adds positional
embedding without any attention mechanism — every position is still processed
independently. Token interaction begins with self-attention in Notebook 05, and the loss
improves meaningfully from there.

Only the last row has a validation column, because only the final notebook holds data out.
Its validation loss sits above its training loss, which is what the held-out split is there
to show: see [section 3.5](#35-training-convergence). Losses are also only comparable down a
column within one dataset — Names, Shakespeare and Oz have different vocabularies and
different lengths.

---

## 5. Running and checking the notebooks

```bash
pip install -r requirements.txt
jupyter notebook            # or open any notebook in Colab
```

Every notebook downloads its own dataset on first run and picks the device itself, so each
one runs top to bottom with no edits, on CPU or GPU. Notebooks 01–05 take about ten minutes
together on a laptop CPU. Notebook 06 and the final notebook are the slow ones: about 15 and 20
minutes on a Colab T4, and about two and three hours on a 12-thread laptop CPU (an i7-1360P).

The losses quoted in [section 4](#4-results-across-all-stages) are checked against the
notebooks themselves:

```bash
python tools/check_notebooks.py              # compare the table with the stored outputs
python tools/check_notebooks.py --execute    # re-run every notebook first (hours on a CPU)
```

The script reads each notebook's `epoch N | train X | val Y` lines, takes the epoch count
and the first and last losses, and compares them with the table to the precision the table
quotes. With `--execute` the fresh outputs are compared within a 5 % tolerance: only the final
notebook seeds PyTorch, and even that seed repeats a run only on the same device and PyTorch
version (the stored outputs come from a T4, and a GPU draws its dropout masks from a different
random generator than a CPU).

The full re-execution last ran in September 2026 and all seven notebooks passed: 01–05 on a
laptop CPU, 06 and the final notebook on a Colab T4 with a newer PyTorch than the stored outputs.
There the final notebook reproduced the table's figures exactly, and every one of its 100 epochs
to within 0.0002.

---

## 6. Credits

| What | Source |
| --- | --- |
| Notebooks 01–06 | Course materials from ECO4126, Yonsei University |
| The curriculum they follow | Andrej Karpathy, [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [`makemore`](https://github.com/karpathy/makemore) (MIT) |
| `names.txt` (notebooks 01–02) | The `makemore` dataset |
| `shakespeare.txt` (notebooks 03–06) | Tiny Shakespeare, from [`char-rnn`](https://github.com/karpathy/char-rnn) |
| `input.txt` (final notebook) | *The Wonderful Wizard of Oz*, [Project Gutenberg #55](https://www.gutenberg.org/ebooks/55) — public domain in the US |

So the split of authorship is: notebooks 01–06 came with the course, and the writing in
this README about them is mine. `notebook_06_h23yonsei.ipynb` is mine end to end — a
different dataset, retuned hyperparameters, and the six modifications listed in
[section 3.4](#34-what-i-changed-from-the-notebook-06-baseline).

---

## License

The work that is mine — `notebook_06_h23yonsei.ipynb`, `tools/` and this README — is
released under the [MIT License](LICENSE). Notebooks 01–06 remain the course's material,
and the datasets belong to their sources above.
