# Pipeline flow

Two views of the same system:

1. **One incoming photo** — the inference path from a raw sea photo to a waviness score.
2. **Full lifecycle** — capture, preprocessing, labeling, splitting, training, deployment, the public website, and the retraining loop.

The technical training stack is a separate document: [training-stack.md](training-stack.md).

## Context

- This is a localized project for **Samsun, Turkey**. Photos come from a fixed location and camera, with 3–4 recurring angles, usually 2–10 photos per day.
- The **lower part of the sea** matters most: it is the water closest to the coast and to the people swimming. The Step 2 crop therefore prefers bottom-anchored rectangles inside the lower sea band.
- The public GitHub repository holds **code only**. Sea photos, labels, and checkpoints are stored and shared through **Hugging Face**.
- Training data and inference use the same preprocessing profile (`fast-v1`): single-pass 448 px segmentation for Step 1, 384 px segmentation for Step 2, float16 on GPU, and a 224×224 model input.

## 1. One incoming photo — inference

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 35, "wrappingWidth": 440}}}%%
flowchart TD
    RAW["Incoming raw sea photo, iPhone HEIC or fixed camera capture"] --> DECODE["Decode image and apply EXIF orientation"]
    DECODE --> SEG1["Step 1: Sea segmentation with pretrained SegFormer water classes"]
    SEG1 --> CLEAN["Clean the water mask: erode edges and drop mixed pixels"]
    CLEAN --> STD["Keep only the sea: crop to the water area and fit on a black 224x224 square"]
    STD --> SEG2["Step 2: Segment the standardized image again"]
    SEG2 --> CROP["Lower-water rectangle search: bottom-anchored, within the lower sea band, the water closest to the coast"]
    CROP --> INPUT["Final 224x224 model input: the same representation used in training"]
    INPUT --> NORM["ImageNet normalization"]
    NORM --> MODEL["Frozen EfficientNet-B0 backbone and trained regression head with sigmoid output"]
    MODEL --> SCORE["Waviness score: 0.00 = calm, 1.00 = very wavy"]

    classDef capture fill:#eef6fb,stroke:#2b6f91,color:#12232b;
    classDef process fill:#eaf4ee,stroke:#2e7d4f,color:#12301f;
    classDef key fill:#fff5e6,stroke:#c47f17,color:#3a2a12;
    classDef ml fill:#f3eefb,stroke:#6b4fa1,color:#221631;
    classDef result fill:#fdecea,stroke:#b33a3a,color:#3b1515;

    class RAW,DECODE capture;
    class SEG1,CLEAN,STD,SEG2,NORM process;
    class CROP,INPUT key;
    class MODEL ml;
    class SCORE result;
```

### What each stage is doing

- **Decode**: opens the iPhone HEIC or camera image and normalizes its EXIF orientation, so the scene is upright before any processing.
- **Step 1 — sea segmentation**: the pretrained SegFormer identifies water classes (`water`, `sea`, `river`, `lake`). The mask is cleaned, the detected water is cropped, non-water pixels are blacked out, and the result is fitted onto a black 224×224 square. This removes sky, buildings, beach, and sand.
- **Step 2 — lower-water crop**: the standardized image is segmented again, and a bottom-anchored rectangle is selected from the lower sea band. That region is the water nearest the coast and nearest the swimmers.
- **Model input**: the selected crop is resized to 224×224 and normalized, which is exactly the representation the regression model was trained on.
- **Regression**: the frozen EfficientNet-B0 backbone extracts features, the small trained head produces one value, and the sigmoid keeps the score in `[0, 1]`.

The inference path recreates the training representation from a single raw photo. Step 2 works on the Step 1 standardized image, not on the original photo.

## 2. Full lifecycle — capture, training, inference, and retraining

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 30, "rankSpacing": 35, "wrappingWidth": 300}}}%%
flowchart TD
    subgraph CAPTURE["Daily capture - Samsun, Turkey"]
        PHOTOS["Sea photos<br/>fixed camera and 3-4 recurring angles<br/>2-10 per day"]
        RAW["Raw photos<br/>step-0-raw-data"]
        PHOTOS --> RAW
    end

    subgraph PREP["Shared preprocessing: fast-v1"]
        SEG1["Step 1: Sea segmentation<br/>standardize to 224x224"]
        SEG2["Step 2: Lower-water crop<br/>bottom of the sea, where people swim"]
        INPUT["224x224 model input"]
        RAW --> SEG1 --> SEG2 --> INPUT
    end

    subgraph DATA["Labels and dataset"]
        LABEL["Manual waviness label<br/>0.00-1.00 in steps of 0.05"]
        SPLIT["Dataset split<br/>train / validation / test<br/>fixed membership over time"]
        SEG2 --> LABEL --> SPLIT
    end

    subgraph TRAIN["Training and selection"]
        FIT["Train frozen EfficientNet-B0<br/>+ regression head<br/>early stopping on validation MAE"]
        CKPT["New checkpoint vN<br/>validation MAE recorded"]
        EVAL["Held-out test evaluation<br/>MAE and RMSE"]
        SPLIT -- train and validation --> FIT
        SPLIT -- test held out --> EVAL
        FIT --> CKPT
        CKPT --> EVAL
    end

    subgraph SERVE["Inference and public answer"]
        MODEL["Deployed waviness model"]
        SCORE["Waviness score 0-1"]
        SITE["swimmable.blaqc.space<br/>is it a good day to swim?"]
        INPUT --> MODEL
        CKPT -- deployed checkpoint --> MODEL
        MODEL --> SCORE --> SITE
    end

    HF[("Hugging Face<br/>sea photos / labels / checkpoints")]
    RAW -. images .-> HF
    LABEL -. labels .-> HF
    CKPT -. checkpoints .-> HF

    EVAL -. next training round .-> PHOTOS

    classDef capture fill:#eef6fb,stroke:#2b6f91,color:#12232b;
    classDef process fill:#eaf4ee,stroke:#2e7d4f,color:#12301f;
    classDef key fill:#fff5e6,stroke:#c47f17,color:#3a2a12;
    classDef ml fill:#f3eefb,stroke:#6b4fa1,color:#221631;
    classDef result fill:#fdecea,stroke:#b33a3a,color:#3b1515;
    classDef store fill:#f2f2f2,stroke:#666666,color:#222222;

    class PHOTOS,RAW capture;
    class SEG1,SEG2,LABEL,SPLIT,FIT,EVAL process;
    class INPUT,CKPT key;
    class MODEL,SCORE ml;
    class SITE result;
    class HF store;
```

### The two loops

Solid arrows show data and model flow; dotted arrows show storage and synchronization with Hugging Face.

- **Daily loop**: new photos arrive, pass through the shared preprocessing, are scored by the deployed model, and become the public swimmability answer on the website.
- **Retraining loop**: new photos are labeled, the fixed-membership split is updated with only the new samples, the model is retrained, a new versioned checkpoint is evaluated on the held-out test set, and that checkpoint becomes the deployed model. Existing images do not need to be reprocessed, and previous splits and checkpoints remain reproducible.

### Where things live

- **GitHub**: pipeline code, documentation, and experiment reports. No sea photos, masks, overlays, labels, or checkpoints.
- **Hugging Face**: the growing photo dataset, labels, and versioned checkpoints. The deployed inference pulls its checkpoint from there.
- **Website**: `https://swimmable.blaqc.space/` turns the current photo's waviness score into a practical answer for people deciding whether to swim.
