# WhatsApp message notification router

Runs an explainable, personalized router over messages plus the supplied user,
group, business, engagement, image and voice-note context. It emits exactly the
six required submission columns and never uses message IDs as labels.

## Run

Place the provided `dataset/` beside this `code/` directory, then run:

```powershell
python code/main.py
python code/router.py --dataset dataset --output output.csv
python code/evaluate.py --dataset dataset
python code/validate_submission.py --dataset dataset --output output.csv
```

The first command reads `messages.csv`, all contextual CSVs, and referenced
media paths. Images are OCR'd through optional `pytesseract`/Pillow; voice notes
are locally transcribed using optional `faster-whisper`. If those extras are not
installed, media is still opened/validated and the system uses all available
metadata rather than failing.

## Decision policy

Safety patterns and prior reports override other signals to `mute`. Otherwise
the router combines source/conversation context, direct mentions, muted-group
status, business verification and relationship, priority language, forwarded
content, and comparable historical interactions. The top relevant historical
messages are returned as evidence. `evaluate.py` runs the same system over the
labeled samples and reports action and category accuracy.

## Architecture

`router.py` owns data loading/joining, feature extraction, retrieval and the
deterministic rules-and-score fallback. It caches media extraction by content
hash under `dataset/.media_cache`, so repeat runs do not repeat OCR or ASR.
`llm_reasoner.py` contains a separate optional OpenAI Responses API adapter: it
supplies structured context and validates strict JSON-schema output before it
can be used. The default submission path stays deterministic and offline, so
missing credentials/network can never prevent producing `output.csv`.

The evaluation harness never alters dataset files; it routes the labeled sample
rows in memory and reports action/type accuracy. Before submission, verify the
output has 110 rows, the six prescribed columns, and only allowed labels.
