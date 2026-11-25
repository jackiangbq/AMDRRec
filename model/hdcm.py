"""Hybrid Dual-Contrastive Module (HDCM) loss composer."""

from torch import nn


class HybridDualContrastiveModule(nn.Module):


    def __init__(self, intent_intent_weight, fused_view_weight):
        super().__init__()
        self.intent_intent_weight = float(intent_intent_weight)
        self.fused_view_weight = float(fused_view_weight)

    def intent_intent_loss(self, user_loss, item_loss):
        return self.intent_intent_weight * (user_loss + item_loss)

    def fused_view_loss(self, user_loss, item_loss):
        return self.fused_view_weight * (user_loss + item_loss)


HDCM = HybridDualContrastiveModule
