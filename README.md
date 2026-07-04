# Sanskrit to English Neural Machine Translation

NLU Assignment 2 — Beladiya Nikunj Bhupatbhai (Roll No: G25AIT1203)

Translates Sanskrit sentences to English with a seq2seq Transformer. The pre-trained
IndicTrans2 model is evaluated zero-shot, then fine-tuned on the provided 10,000
Sanskrit–English training pairs, and the better of the two (selected on the dev set)
produces `submission.csv`.

## Files

| File | Purpose |
|---|---|
| `notebook_indictrans2.ipynb` | The single code file: training + evaluation + submission generation. This same notebook is used at evaluation time (see below). Includes the executed outputs of the full run. |
| `report.pdf` | Assignment report (architecture, training, results, examples, discussion). |
| `Data-set.zip` | The course-provided dataset: `train/dev/test` Sanskrit–English CSV pairs (10k/1k/1k), joined on `Source_id`. Only this dataset was used for training. Upload this zip to `/content` when running the notebook. |

## Pre-trained models used (disclosure)

- **`ai4bharat/indictrans2-indic-en-dist-200M`** (AI4Bharat, MIT license) — the
  translation model. A 211.78M-parameter Transformer encoder-decoder that natively
  supports Sanskrit (`san_Deva`) → English (`eng_Latn`). We fine-tune it for 3 epochs
  on the provided training pairs only.
- **RoBERTa-large** — downloaded internally by the `bert-score` library, used only to
  compute the BERTScore evaluation metric. It plays no part in translation.
- The `IndicTransToolkit` preprocessor and the model's own SentencePiece tokenizer are
  used unchanged.

All models run locally on the GPU after a one-time weight download from HuggingFace.
No external APIs are called anywhere in the code.

## How to run (Google Colab)

1. Open `notebook_indictrans2.ipynb` in Colab and set Runtime → Change runtime type → **T4 GPU**.
2. Add a HuggingFace token as a Colab secret named `HF_TOKEN` (key icon in the left
   sidebar, "Notebook access" ON). The IndicTrans2 repository is gated, so the account
   must have accepted access on the model page once.
3. Upload the dataset as `Data-set.zip` to `/content` via the Files panel
   (the zip contains the six CSVs).
4. Runtime → **Run all**. The notebook installs its own dependencies (first cell),
   trains, evaluates, and downloads `submission.csv` at the end.

Full run takes roughly 45 minutes on a free T4 (fine-tuning is the long step).

## Evaluation on the private test set

The notebook is the evaluation-time file. To score the private dataset, change only
the two test filenames in the `FILES` dict (Section 3 of the notebook):

```python
"test_sa": "<private_test_sa>.csv",
"test_en": "<private_test_en>.csv",
```

then Run all. Everything downstream (translation, BLEU, BERTScore, timing,
`submission.csv`) picks up the new files automatically.

## Results (public test set, single end-to-end run)

| Model | corpus BLEU | BERTScore F1 | inference time (1000 sents) | parameters |
|---|---|---|---|---|
| zero-shot | 0.1458 | 0.5113 | 62.7 s | 211,780,608 |
| **fine-tuned (submitted)** | **0.2114** | **0.5671** | 102.6 s | 211,780,608 |

Metrics per the assignment spec: default NLTK `corpus_bleu` (no custom weights) and
BERTScore F1 with `rescale_with_baseline=True`.
