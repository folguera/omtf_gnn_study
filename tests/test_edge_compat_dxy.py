import unittest

import torch

from omtf_gmt.features_tps import (
    F_TPS_DEPTH_REGION,
    F_TPS_ETA1,
    F_TPS_PHI_REL,
    F_TPS_TF_LAYER,
)
from omtf_gmt.models.edge_compat_assign import assignment_supervision_loss
from omtf_gmt.models.edge_compat_dxy import build_edge_compat_dxy, _structured_edge_mask


class StructuredEdgeMaskTest(unittest.TestCase):
    def test_layer_station_and_geometry_constraints(self):
        stubs = torch.zeros(1, 7, 14)
        valid = torch.tensor([[True, True, True, True, True, True, False]])

        # Base node: layer 0, station 1, phi 0, eta 0.
        feature_indices = [F_TPS_PHI_REL, F_TPS_ETA1, F_TPS_TF_LAYER, F_TPS_DEPTH_REGION]
        stubs[0, 0, feature_indices] = torch.tensor([0.05, 0.1, 0.0, 0.0])
        stubs[0, 1, feature_indices] = torch.tensor([0.05, 0.1, 1.0 / 4.0, 1.0 / 3.0])
        stubs[0, 2, feature_indices] = torch.tensor([0.05, 0.1, 3.0 / 4.0, 2.0 / 3.0])
        stubs[0, 3, feature_indices] = torch.tensor([0.05, 0.1, 1.0, 1.0])
        stubs[0, 4, feature_indices] = torch.tensor([0.3, 0.0, 0.0, 0.0])
        stubs[0, 5, feature_indices] = torch.tensor([0.05, 0.3, 0.0, 0.0])

        edge_mask = _structured_edge_mask(stubs, valid)[0]

        self.assertTrue(edge_mask[0, 1])  # Adjacent layer and nearby in phi/eta.
        self.assertTrue(edge_mask[0, 2])  # Station gap of two is allowed.
        self.assertFalse(edge_mask[0, 3])  # Both topology gaps exceed one skipped layer.
        self.assertFalse(edge_mask[0, 4])  # Outside the phi window.
        self.assertFalse(edge_mask[0, 5])  # Outside the eta window.
        self.assertFalse(edge_mask[0, 6])  # Padding node.
        self.assertFalse(torch.diagonal(edge_mask).any())
        self.assertTrue(torch.equal(edge_mask, edge_mask.T))

    def test_assignment_loss_uses_packed_slot_track_id(self):
        assign_weights = torch.tensor([[[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]]])
        track_id = torch.tensor([[2, 0]])
        valid = torch.tensor([[True, True]])
        gen_pt = torch.tensor([[10.0, 0.0, 0.0]])
        target_track_id = torch.tensor([[2, 0, 0]])

        loss = assignment_supervision_loss(
            assign_weights, track_id, valid, gen_pt, target_track_id
        )

        self.assertAlmostEqual(float(loss), -torch.log(torch.tensor(0.9)).item(), places=6)

    def test_slotwise_dxy_attention_handles_empty_windows(self):
        model = build_edge_compat_dxy(hidden=8)
        stubs = torch.randn(2, 5, 14)
        valid = torch.tensor(
            [[True, True, True, False, False], [False, False, False, False, False]]
        )

        output = model(stubs, valid)

        self.assertEqual(output["dxy_pred"].shape, (2, 3))
        self.assertEqual(output["assign_weights"].shape, (2, 3, 5))
        self.assertTrue(torch.allclose(output["assign_weights"][0].sum(dim=-1), torch.ones(3)))
        self.assertTrue(torch.equal(output["assign_weights"][1], torch.zeros(3, 5)))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))


if __name__ == "__main__":
    unittest.main()