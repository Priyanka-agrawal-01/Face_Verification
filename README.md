# Face Verification Lab

A small, self-contained demo of how face verification works:
**detect → align → embed (128 numbers) → measure distance → apply a threshold**.

Everything runs in the browser with [`@vladmandic/face-api`](https://github.com/vladmandic/face-api)
(MIT). Photos are never uploaded anywhere. The server only serves the page,
the model weights and the dataset folder.

## Run it

Requires Node.js 18 or later.

```bash
npm install
npm start
```

Then open <http://127.0.0.1:5180>.

By default the server listens on `127.0.0.1` only. These environment variables change that:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Interface to listen on. Use `0.0.0.0` on a hosting provider. |
| `PORT` | `5180` | Port to listen on. Most hosts set this for you. |
| `DATA_DIR` | `./data` | Folder holding datasets, e.g. a mounted disk. |

The dataset folder is optional. Without it the Verify, Gallery and How it works tabs work as normal, the sample-pair buttons are hidden, and Evaluate can still score a folder picked from the visitor's own computer (processed in their browser).

## Deploy to Render

The repository includes a [Render Blueprint](https://render.com/docs/blueprint-spec) in `render.yaml`. It creates a free Node web service that runs `npm ci`, then `npm start` with `HOST=0.0.0.0`, and checks `/healthz`. Node 22 is selected by `.node-version`.

1. Push this project to a GitHub, GitLab or Bitbucket repository. `node_modules/` and `data/` are git-ignored, so the LFW dataset is not uploaded.
2. In the Render dashboard choose **New → Blueprint** and select the repository.
3. Confirm the `face-verification-lab` service. The first build takes a minute or two; every push to the default branch redeploys.

Things to know:

- **Free instances sleep** after about 15 minutes without traffic. The next visit wakes them, which takes up to a minute.
- **No dataset on Render.** `data/` is not in the repository, and LFW should not be redistributed with the app. To evaluate on the server, attach a [persistent disk](https://render.com/docs/disks) (paid plans), copy a dataset onto it, and set `DATA_DIR` to its mount path.
- **Production behaviour is built into `server.mjs`:** gzip for text assets (face-api drops from 1.3 MB to 0.3 MB), ETag revalidation, one-day caching for the library and model weights, a strict Content-Security-Policy and other security headers, HSTS behind HTTPS, a `/healthz` endpoint, and graceful shutdown on `SIGTERM`.
- Visitors download about 12 MB of model weights on their first visit. Browsers then cache them.

## What it does

| Tab | Question it answers |
| --- | --- |
| **Verify (1:1)** | Are these two photos the same person? Shows boxes, 68 landmarks, the embedding distance and a verdict at an adjustable threshold. Drop, paste or pick photos, or load a random same-person / different-people pair from a dataset that has `pairs.txt`. |
| **Gallery (1:N)** | Which enrolled person, if any, is this? Enrol a few consenting people, then rank a query photo against everyone. Embeddings live only in memory. |
| **Evaluate** | How accurate is it really? Scores labelled pairs, then reports accuracy, false accept and false reject rates, EER, ROC/AUC, cross-validated accuracy, the hardest mistakes, and a CSV export. |
| **How it works** | The pipeline explained step by step. |

## How it works

1. **Detect.** SSD MobileNet v1 finds every face and gives each a box and a confidence score. The largest face is treated as the subject. If nothing is found, it retries at a lower confidence.
2. **Landmarks and alignment.** A 68-point landmark network locates the eyes, nose, mouth and jaw. The face is cropped consistently from those points.
3. **Embed.** A ResNet-style recognition network maps the aligned face to a 128-dimensional vector. Training pulls photos of the same person together and pushes different people apart.
4. **Compare.** The Euclidean distance between two vectors measures dissimilarity.
5. **Decide.** Distance ≤ threshold means "same person". The threshold trades off:
   - **False accept rate (FAR):** different people wrongly accepted.
   - **False reject rate (FRR):** the same person wrongly rejected.

1:N identification is the same comparison repeated against every enrolled embedding, keeping the closest.
Large face-search systems do the same thing over millions of vectors with an approximate nearest-neighbour index.

## Evaluation method

- **Official LFW protocol.** `pairs.txt` has 10 folds of 300 same-person and 300 different-person pairs. With 2 or more folds selected, the app uses leave-one-fold-out cross-validation: it picks the best threshold on the other folds, measures accuracy on the held-out fold, then reports the mean ± standard deviation.
- **Generated pairs** (for your own folders). It uses every same-person pair up to a cap, plus an equal number of random different-person pairs. The random pairs use a fixed seed, so runs are reproducible.
- **Pairs where either photo has no detectable face** are skipped and counted, not silently dropped.
- **"Best threshold" accuracy** is measured on the same pairs used to choose it, so it is optimistic. Quote the cross-validated figure in a report.

## Data

### LFW (Labeled Faces in the Wild)

The research dataset lives in `data/lfw/<Person_Name>/<Person_Name>_0001.jpg`, with the official `pairs.txt` beside it.
It was downloaded from the figshare mirror used by scikit-learn:

```bash
curl -L -o data/download/lfw.tgz https://ndownloader.figshare.com/files/5976018
curl -L -o data/download/pairs.txt https://ndownloader.figshare.com/files/5976006
```

The archive's SHA-256 is `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`.
Extract it with:

```bash
tar -xzf data/download/lfw.tgz -C data
```

Then copy `pairs.txt` into `data/lfw/`. `data/` is git-ignored, so do not redistribute the dataset with your project.

### Your own photos

Make a folder with one sub-folder per person, either under `data/` (it then appears in the dataset list) or anywhere else (pick it with "Choose a folder…"):

```
my-photos/
  alice/  1.jpg 2.jpg 3.jpg
  bob/    1.jpg 2.jpg
```

Only include people who have agreed to it.

## Results on LFW

Official `pairs.txt`, folds 1–2: 1,200 pairs and 1,583 photos, run on this PC's WebGL in about 6 minutes.

| Metric | Value |
| --- | --- |
| Pairs scored | 1,198 (600 same, 598 different, 2 skipped because a photo had no detectable face) |
| **Cross-validated accuracy** (threshold chosen on the other fold) | **97.33% ± 0.33%** |
| Accuracy at the default threshold 0.60 | 97.58% (false accepts 0.50%, false rejects 4.33%) |
| Best threshold on these pairs | 0.62, giving 97.75% (optimistic) |
| Equal error rate | 2.42% at threshold 0.64 |
| ROC AUC | 0.9908 |

Engineering note: the first version skipped 68 of 1,200 pairs. The cause was faces cropped tight to the image edge. A final retry with a neutral border around the image cut that to 2 pairs.
Run all 10 folds for the full standard protocol. It takes about 30–40 minutes in the browser.

## Project layout

```
server.mjs        Static server (default 127.0.0.1:5180) plus /api/datasets, /api/dataset and /healthz
render.yaml       Render Blueprint for deployment
.node-version     Node major version used by Render
public/index.html UI markup: four tabs
public/theme.js   Applies the saved light/dark choice before first paint
public/core.js    Face pipeline and evaluation maths (pairs, thresholds, ROC, cross-validation)
public/app.js     UI wiring and SVG charts
public/styles.css Light and dark theme
data/             Datasets (git-ignored)
```

## Limitations and ethics

- **LFW is easy and dated.** The images are mostly frontal, well-lit news photos of public figures. Accuracy on phone selfies, CCTV or different ages will be lower.
- **Demographics.** LFW is skewed toward adult, male, lighter-skinned faces, and published studies show recognition error rates differ across groups. Report this alongside any accuracy figure.
- **Verification (1:1) is easier than open-set identification (1:N).** A threshold that looks fine for pairs can still let strangers through when compared against many people. In testing, an unenrolled LFW face matched an enrolled person at 0.592, under the default 0.60.
- **Consent.** The technique is simple. What turns it into a surveillance tool is a large index of faces collected without consent. Keep galleries small, consented and local.
