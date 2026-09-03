"""Run data sanity checks and create a filesystem-friendly augmentation gallery."""

from __future__ import annotations

import argparse
import html
import time
from pathlib import Path

import torch
from PIL import Image

from step_4_a_wave_dataset import WaveDataset, denormalize_image, build_training_transform
from step_4_b_wave_datamodule import WaveDataModule
from torchvision.transforms.functional import to_tensor


def _save_tensor(tensor: torch.Tensor, path: Path) -> None:
    image = (denormalize_image(tensor.detach().cpu()) * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(image, mode="RGB").save(path, quality=94)


def _write_gallery(output_dir: Path, dataset: WaveDataset, variants: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    training_transform = build_training_transform()
    cards: list[str] = []
    for index in range(len(dataset)):
        started = time.perf_counter()
        filename, label = dataset.samples[index]
        with Image.open(dataset.image_path(index)) as opened:
            original = opened.convert("RGB")
        original_tensor = to_tensor(original)
        safe_stem = f"{index:03d}_{Path(filename).stem}"
        original_name = f"{safe_stem}_processed.jpg"
        original.save(output_dir / original_name, quality=94)
        images = [("processed", original_name)]
        for variant in range(variants):
            # A gallery is easier to judge when a sampled augmentation is not
            # effectively an identity transform. This retry affects previews
            # only; training remains fully random and uses the same transform.
            tensor = training_transform(original)
            for _ in range(4):
                if (denormalize_image(tensor) - original_tensor).abs().mean().item() >= 0.01:
                    break
                tensor = training_transform(original)
            name = f"{safe_stem}_augmented_{variant + 1}.jpg"
            _save_tensor(tensor, output_dir / name)
            images.append((f"augmented #{variant + 1}", name))
        thumbs = "".join(
            f'<figure><a href="{html.escape(name)}" target="_blank"><img src="{html.escape(name)}" loading="lazy" alt="{html.escape(kind)}"></a><figcaption>{html.escape(kind)}</figcaption></figure>'
            for kind, name in images
        )
        cards.append(f'<article><h2>{html.escape(filename)}</h2><p>label: {label:.2f}</p><div class="images">{thumbs}</div></article>')
        print(f"preview {index + 1}/{len(dataset)}: {filename} ({time.perf_counter() - started:.3f}s)")
    document = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Training augmentation inspection</title>
<style>body{font:16px system-ui,sans-serif;margin:0;padding:1rem;background:#f3f4f6;color:#17202a}main{max-width:1400px;margin:auto}article{background:#fff;border-radius:10px;padding:1rem;margin:0 0 1rem;box-shadow:0 1px 4px #0002}h1,h2{margin:.2rem 0 .5rem}h2{font-size:1rem;overflow-wrap:anywhere}.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}figure{margin:0}img{display:block;width:100%;height:auto;border-radius:6px;cursor:zoom-in}figcaption{text-align:center;margin-top:.35rem;color:#53606b}</style><main><h1>Sea waviness training-data inspection</h1><p>Click any image to open it at full size. Augmentations are sampled independently.</p>""" + "".join(cards) + "</main>"
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("step-4-data-preview"))
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    started = time.perf_counter()
    module = WaveDataModule(args.data_dir, batch_size=args.batch_size)
    module.setup()
    print(f"train samples: {len(module.train_dataset)}")
    print(f"validation samples: {len(module.val_dataset)}")
    print(f"test samples: {len(module.test_dataset)}")
    print(f"batch size: {args.batch_size}")
    batch_images, batch_targets = next(iter(module.train_dataloader()))
    print(f"images: {list(batch_images.shape)}")
    print(f"targets: {list(batch_targets.shape)}")
    if batch_images.dtype != torch.float32 or batch_targets.dtype != torch.float32:
        raise AssertionError("Expected float32 image and target tensors")
    if batch_images.shape[1:] != (3, 224, 224) or batch_targets.ndim != 1:
        raise AssertionError("Unexpected batch shapes")
    # Deterministic transforms must produce identical tensors for the same image.
    val_a = module.val_dataset[0][0]
    val_b = module.val_dataset[0][0]
    test_a = module.test_dataset[0][0]
    test_b = module.test_dataset[0][0]
    if not torch.equal(val_a, val_b) or not torch.equal(test_a, test_b):
        raise AssertionError("Validation/test transforms are not deterministic")
    print("sanity checks: passed")
    _write_gallery(args.output_dir, module.train_dataset, args.variants)
    print(f"gallery: {(args.output_dir / 'index.html').resolve()}")
    print(f"completed in {time.perf_counter() - started:.3f}s")


if __name__ == "__main__":
    main()
