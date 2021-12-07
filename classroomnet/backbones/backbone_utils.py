import warnings
from typing import Callable, Dict, Optional, List, Union

from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

from classroomnet.backbones.student import StudentModule

# from .. import mobilenet
# from .. import resnet
# from .._utils import IntermediateLayerGetter


class StudentBackbone(nn.Module):
    def __init__(
        self,
        num_teachers,
        feature_dim,
        statistics_list,
        distill_dim_list,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = StudentModule(num_teachers=num_teachers, feature_dim=feature_dim, statistics_list=statistics_list, distill_dim_list=distill_dim_list)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, z_to_train_list = self.body(x)
        x = self.fpn(x)
        return x, z_to_train_list
