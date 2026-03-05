# Data Directory

This folder is intentionally kept for local data only.

## What Is Expected Here

```text
data/
  raw/CE-CSL/
    label/{train,dev,test}.csv
    video/{train,dev,test}/<Translator>/*.mp4
  meta/
    manifest.jsonl
    vocab.json
  processed/
    train/*.npz
    dev/*.npz
    test/*.npz
```

## Important

- Dataset files are not distributed with this repository.
- Respect CE-CSL licensing and terms before downloading or sharing any data.
- Generated files under `meta/` and `processed/` are ignored by default in git.
