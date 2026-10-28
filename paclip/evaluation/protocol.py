"""Historical checkpoint supervision defaults; no model or training imports."""
def postprocess_config_for_checkpoint(metadata=None):
    """旧胸腺瘤定位权重没有正常病例监督，不能用新建 Config 的默认 w_global 打开融合。"""
    from paclip.evaluation.postprocess import build_postprocess_config

    meta = dict(metadata or {})
    if meta.get("global_supervision_enabled") is None and meta.get("w_global") is None and not (
        isinstance(meta.get("loss"), dict) and meta["loss"].get("w_global") is not None
    ):
        meta["global_supervision_enabled"] = False
        meta["w_global"] = 0.0
    return build_postprocess_config(
        meta.get("score_mode"),
        meta,
        topk_ratio=float(meta.get("topk_ratio", 0.05)),
        mask_mode=meta.get("mask_mode") or "fixed",
        global_weight=meta.get("global_weight"),
    )
