import unittest
import torch
from paclip.models.dense_localizer import DenseLocalizer, top_roi_score
from paclip.models.novelty_model import NoveltyModel
from paclip.losses import local_alignment_loss, binary_local_loss, focal_bce, soft_dice_loss, boundary_loss
from paclip.data.supervision import pool_mask_targets
from paclip.diagnostics.novelty_fixtures import TinyEncoder


class DenseLocalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rectangular_grid_and_projection_gradients(self):
        model = DenseLocalizer()
        x = torch.randn(2, 3, 15, 512)
        output = model(x, (3, 5))
        self.assertEqual(output.shape, (2, 1, 12, 20))
        output.square().mean().backward()
        for projection in model.projections:
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0)

    def test_freeze_and_optimizer(self):
        model = NoveltyModel(TinyEncoder())
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        model.assert_optimizer(opt)
        model(torch.randn(1, 3, 224, 224), torch.ones(1, 224, 224))["dense_logits"].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.head.parameters()))

    def test_visual_ablation_only_adapter_unfrozen(self):
        model = NoveltyModel(TinyEncoder(), train_visual=True).train()
        model(torch.randn(1, 3, 224, 224), torch.ones(1, 224, 224))["dense_logits"].sum().backward()
        self.assertIsNone(model.encoder.backbone.weight.grad)
        self.assertIsNotNone(model.encoder.visual_inlayer_bank.adapters["brain"].weight.grad)

    def test_empty_roi_validity_not_fake_score(self):
        scores, valid = top_roi_score(torch.ones(2, 1, 3, 5), torch.zeros(2, 3, 5))
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(scores).all())

    def test_binary_loss_preserves_legacy_formula_and_gradient(self):
        x = torch.randn(2, 2, 3, 5, requires_grad=True)
        masks = [torch.zeros(29, 47), torch.zeros(30, 42)]
        masks[0][2, 2] = 1; masks[1][5:20, 4:30] = 1
        roi = torch.ones(2, 13, 17)
        logits = x[:, 1].float() - x[:, 0].float()
        target, hard = pool_mask_targets(masks, logits.shape[-2:])
        valid = torch.ones_like(logits, dtype=torch.bool)
        prob = logits.sigmoid()
        expected = focal_bce(logits, target, valid)
        expected = expected + soft_dice_loss(prob, target, valid) + .1 * boundary_loss(prob, target, valid)
        selected = hard * valid
        expected = expected + .1 * ((1 - prob) * selected).sum() / selected.sum().clamp_min(1)
        actual = local_alignment_loss(x, masks, roi)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(torch.autograd.grad(expected, x, retain_graph=True)[0], torch.autograd.grad(actual, x)[0]))

    def test_state_roundtrip(self):
        head = DenseLocalizer().eval()
        other = DenseLocalizer().eval()
        other.load_state_dict(head.state_dict(), strict=True)
        x = torch.randn(1, 3, 12, 512)
        self.assertTrue(torch.equal(head(x, (3, 4)), other(x, (3, 4))))
