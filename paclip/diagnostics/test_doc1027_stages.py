import tempfile
import unittest
from pathlib import Path
import torch
from paclip.models.dense_localizer import DenseLocalizer
from paclip.data.synthetic_feature import synthesize_feature
from paclip.data.synthetic_image import synthesize_image, ImageSyntheticConfig
from paclip.losses_novelty import lesion_rank_loss, grouped_novelty_loss
from paclip.engine.novelty_trainer import NoveltyTrainer
from paclip.data.patient_manifest import choose_support, resolve_support
from paclip.diagnostics.novelty_fixtures import fixture_rows, TinyEncoder


class Doc1027Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_document_decoder_sequence(self):
        head=DenseLocalizer(decoder_variant='doc1027')
        self.assertEqual([type(m).__name__ for m in head.decoder],['Conv2d','GroupNorm','GELU','Conv2d','Upsample','Conv2d'])
        self.assertEqual(head(torch.randn(2,3,15,512),(3,5)).shape,(2,1,12,20))

    def test_sigma_is_per_dimension_and_visual_gradient_is_retained(self):
        levels=torch.nn.functional.normalize(torch.randn(3,196,512),dim=-1).requires_grad_()
        out=synthesize_feature(levels,(14,14),torch.ones(224,224),3,rhos=[.1],noise_mode='per_dimension_gaussian',detach_reference=False)
        mask=out['mask_high'].flatten().bool();noise=(out['levels']-levels)[:,mask]
        self.assertAlmostEqual(float(noise.detach().std()),.1,delta=.01)
        self.assertGreater(float(noise.detach().norm(dim=-1).mean()),1.5)
        out['levels'].sum().backward()
        self.assertTrue(torch.equal(levels.grad,torch.ones_like(levels)))
        self.assertEqual(out['parameters']['sigma_per_dimension'],.1)
        self.assertFalse(out['parameters']['reference_detached'])

    def test_document_global_rank_formula(self):
        score=torch.tensor([[.1,.2],[.3,.4]],requires_grad=True)
        mask=torch.tensor([[1.,0.],[0.,0.]])
        loss,_=lesion_rank_loss(score,mask,torch.ones(2,2),margin=.5,mode='global_mean')
        torch.testing.assert_close(loss,torch.relu(.5-score[0,0]+score.flatten()[1:].mean()))
        combined=grouped_novelty_loss(torch.stack([score,score+1])[:,None],[mask,mask],torch.ones(2,2,2),
            ['real','synthetic'],['none','feature'],lambda_rank=.2,rank_mode='global_mean',rank_aggregation='positive_image_mean')
        torch.testing.assert_close(combined['loss_rank'],loss)

    def test_blur_and_local_noise_have_real_compact_masks(self):
        gray=torch.rand(224,224)*.6+.2
        for mode in ('gaussian_blur','local_noise'):
            out=synthesize_image(gray,torch.ones(224,224),4,config=ImageSyntheticConfig(modes=[mode]))
            self.assertTrue(torch.equal((out['gray']-gray).abs()>1e-6,out['mask_high'].bool()))

    def test_real_joint_step_with_document_gradient_and_loss_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=fixture_rows(tmp); support=choose_support(rows); selected=resolve_support(rows,support)
            cfg={'task':'thymoma','mode':'joint_synthetic','train_visual':True,'head_spec':{'decoder_variant':'doc1027'},
                 'feature_noise_mode':'per_dimension_gaussian','feature_detach_reference':False,'lambda_rank':.2,
                 'rank_mode':'global_mean','rank_aggregation':'positive_image_mean'}
            t=NoveltyTrainer(cfg,rows,selected,{},Path(tmp)/'run',TinyEncoder);t.output.mkdir()
            logged=t.train_step()
            self.assertEqual(logged['counts'],{'normal':4,'real':2,'syn_image':1,'syn_feature':1})
            self.assertTrue(logged['gradient_policy_verified'])
            self.assertTrue(torch.isfinite(t.model.encoder.visual_inlayer_bank.adapters['brain'].weight.grad).all())
