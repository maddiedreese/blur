# Benchmark contract

Evaluation score files use JSON Lines with one object per base image:

```json
{"id":"stable-id","label":1,"score":0.87,"group":"held-out-generator"}
```

Run:

```sh
node scripts/evaluate.mjs scores.jsonl
```

Dataset preparation must keep an image and every recompressed, resized, or cropped derivative in the same split. Entire generator families and real-image sources should be held out. The test split must not be used to select models, fusion weights, or calibration parameters.

Required stress groups include camera photography, human digital art, CGI, screenshots, memes, recent held-out generators, metadata-stripped images, JPEG/WebP recompression, resizing, cropping, and screenshots. Report every group separately in addition to aggregate balanced accuracy.
