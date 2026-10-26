# Receipt total extraction with LayoutLMv3

Detect the **total amount (tax included)** on receipt images by fine-tuning
[LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) as a token classification
model, then running the full **OCR → model → amount** pipeline on real-world receipt photos.

## Results

**CORD test set (100 receipts)**

| Metric | Value |
|---|---|
| Entity-level precision / recall / F1 (`TOTAL`) | 0.921 / 0.979 / **0.949** |
| Exact amount per receipt | **95 / 100** |

**End-to-end on 100 real phone photos (ExpressExpense, totals annotated by hand)**

| | Tesseract | docTR |
|---|---|---|
| True total read by the OCR | 38/100 | 96/100 |
| Correct amount (model only) | 30/100 | 69/100 |
| Correct amount (with post-processing) | 33/100 | **87/100** |

**Main finding:** the OCR is the bottleneck. Switching from Tesseract to docTR raises the
exact-amount rate from 33% to 87% without changing the model. Trained only on Indonesian
receipts (CORD), the model still finds totals on American receipts.

## Approach

1. **Data exploration** of CORD v2: how totals are annotated, multi-word amounts, receipts without a total.
2. **Label design:** BIO tagging (`B-TOTAL`, `I-TOTAL`, `O`) at word level.
3. **Inputs:** words, bounding boxes (normalised to 0–1000) and image, padded to 320 tokens.
4. **Fine-tuning** of LayoutLMv3-base (125M parameters) with AdamW and a linear schedule;
   best model selected on validation F1 (`seqeval`).
5. **Evaluation** on the test set, with error analysis receipt by receipt.
6. **End-to-end pipeline** on unseen photos, comparing two open-source OCR engines,
   with a light post-processing step.

## Run it

Open `receipt_total_layoutlmv3.ipynb` in Google Colab (a T4 GPU is enough).

- With `TRAIN = False` (default), the fine-tuned model is downloaded from the Hugging Face Hub:
  [`SedkiGharsallah/layoutlmv3-cord-total`](https://huggingface.co/SedkiGharsallah/layoutlmv3-cord-total).
- With `TRAIN = True`, the model is retrained (about 25 minutes on a T4).

All data is public: CORD comes from the Hugging Face Hub and the ExpressExpense images are
downloaded by the notebook.

## Stack

Python · PyTorch · Hugging Face Transformers & Datasets · seqeval · Tesseract · docTR · pandas · matplotlib

## Limitations and next steps

- Main model confusion: subtotal vs. total, and receipts without a total line (only 21 in training).
- Hyperparameters follow the official LayoutLMv3/CORD example and were not tuned.
- Next steps: train on similar amounts (subtotal, cash, item prices) with an explicit fallback rule,
  and fine-tune on receipts closer to the target domain.
