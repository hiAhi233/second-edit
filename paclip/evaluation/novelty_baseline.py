"""Evaluate the first-step model under EXACTLY the new dense-head protocol."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from paclip.data.novelty_dataset import NoveltyDataset
from paclip.data.supervision import pool_mask_targets
from paclip.data.fewshot import stable_digest
from paclip.models.dense_localizer import top_roi_score


@torch.no_grad()
def predict_baseline(model, rows, prompts, device, cache=None, output_dir=None):
    model.eval()
    anchors = model.encode_anchors(prompts)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    result = []
    dataset = NoveltyDataset(rows)
    for i in range(len(dataset)):
        s = dataset[i]
        out = model(s["image"][None].to(device), anchors)
        patch_logits = out["patch_logits"]
        logit = patch_logits[:, 1:2].float() - patch_logits[:, 0:1].float()
        if cache is not None:
            cache.put(s, out["f_patch_levels"][0], tuple(logit.shape[-2:]))
        logit = F.interpolate(logit, (224, 224), mode="bilinear", align_corners=False)
        image_score, _ = top_roi_score(logit, s["roi"][None].to(device), [s["roi_valid"]])
        _, gt = pool_mask_targets([s["mask_high"]], (224, 224))
        item = {**rows[i], "score": logit[0, 0].cpu().numpy(), "gt": gt[0].numpy().astype(bool),
                "roi": s["roi"].numpy().astype(bool), "roi_valid": s["roi_valid"], "image_score": float(image_score[0])}
        result.append(item)
        if output_dir:
            np.savez_compressed(Path(output_dir) / (stable_digest(s["sample_id"])[:20] + ".npz"),
                logit=item["score"], probability=logit.sigmoid()[0, 0].cpu().numpy(), gt=item["gt"], roi=item["roi"],
                sample_id=s["sample_id"], image_score=item["image_score"])
    return result
