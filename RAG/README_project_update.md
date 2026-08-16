# WhatsApp Message Notification Router

An offline-first, personalized notification router for WhatsApp messages. For
each incoming text, image, or voice note, it returns one of three actions:

- `notify` — interrupt the user now
- `digest` — safe and useful, but can wait
- `mute` — low-value, repetitive, suspicious, or unsafe

The solution uses the supplied user, group, business, historical-message, and
interaction data. It does not use message IDs as labels or require an API key
to generate a submission.

## Design

The pipeline is deliberately layered so safety and personalization are both
auditable.

1. **Data joining** — loads all dataset CSVs and joins relevant user, group,
   business, and relationship context for each message.
2. **Multimodal extraction** — opens every referenced local image or audio
   asset. When installed, Pillow/Tesseract performs OCR and faster-whisper
   performs local transcription. Extracted media text is cached by content hash.
3. **Evidence retrieval** — finds comparable historical messages for the same
   user using sender, group, business, and content overlap, then uses past
   opens, replies, dismissals, mutes, and reports as evidence.
4. **Safety override** — scam-like OTP, KYC, credential, payment-pressure, and
   impersonation patterns route to `mute` before lower-priority reasoning.
5. **Personalized scoring** — direct mentions, trusted group admins, work or
   school operations, active business relationships, prior engagement, group
   mute state, opt-outs, repetition, and notification fatigue determine the
   final route.
6. **Optional LLM adjudication** — with `USE_LLM_REASONING=true` and an
   `OPENAI_API_KEY`, ambiguous safe messages may be assessed by the Responses
   API. Its strict JSON-schema output is validated before use; safety overrides
   cannot be changed by the LLM.

## Repository layout

```text
code/
  main.py                  # Submission entry point
  router.py                # Loading, media, retrieval, rules, routing
  llm_reasoner.py          # Optional schema-validated LLM adapter
  evaluate.py              # Evaluation against solved sample rows
  validate_submission.py   # Output contract checks
  requirements.txt
dataset/                   # Provided challenge data and media
output.csv                 # Generated predictions
log.txt                    # Chat transcript copy for submission
```

## Run

Requires Python 3.10+.

```powershell
# From the repository root
python code/main.py

# Check the submission contract
python code/validate_submission.py --dataset dataset --output output.csv

# Score against the 30 provided labeled examples
python code/evaluate.py --dataset dataset
```

`python code/main.py` writes `output.csv` at the repository root. It produces
one prediction for every row in `dataset/messages.csv` with the exact required
columns:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

Optional local media extraction packages are `pillow`, `pytesseract`, and
`faster-whisper`. The system remains runnable without them, using available
text and metadata while still validating referenced media assets.

## Evaluation and validation

`evaluate.py` runs the same router over `dataset/sample_messages.csv` in memory
and reports action and message-type accuracy. It never alters any dataset file.
`validate_submission.py` checks schema ordering, row count, unique message IDs,
allowed action labels, and confidence ranges before upload.

Hidden labels are not available locally, so sample metrics are diagnostic—not a
guarantee of final leaderboard performance.

## Submission

Upload these three artifacts to HackerRank:

1. `code.zip` — archive of the current `code/` files; excludes `.env`, virtual
   environments, caches, datasets, and build artifacts.
2. `output.csv` — final predictions for all messages.
3. `log.txt` — chat transcript.

Before uploading, rerun the validation command and rebuild `code.zip` after
any source change. Keep credentials only in environment variables; do not add
them to the archive or transcript.
