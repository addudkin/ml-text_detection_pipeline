import torch
import torch.nn as nn
from typing import List, Union, Tuple


class RGBNormalizeLayer:
    """A layer for normalizing RGB images.

    Args:
        mean: A list of means for each channel in the image.
        std: A list of standard deviations for each channel in the image.

    Attributes:
        mean (torch.Tensor): A tensor holding the means for each channel in the image.
        std (torch.Tensor): A tensor holding the standard deviations for each channel in the image.

    Typical usage example:
        normalize = RGBNormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        input = torch.randn(1, 3, 32, 32)
        output = normalize.forward(input)
    """
    def __init__(
            self,
            mean: List[float],
            std: List[float]
    ):
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.std = torch.FloatTensor(std).view(-1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor and returns the output.

        Args:
            x: A 4D tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: A 4D tensor with the same shape as `x`.
        """
        x = torch.div(x, 255.)
        x = torch.sub(x, self.mean)
        x = torch.div(x, self.std)
        return x


class ModelWrapperLineSegmentation(nn.Module):
    """A wrapper class for PyTorch models.

    This class wraps a PyTorch model and normalizes the input tensors before forwarding them through the model.
    The normalization layer can be either an instance of `RGBNormalizeLayer` or `GrayScaleNormalizeLayer`,
    depending on the input shape of the tensors.

    Args:
        model: The model to wrap.
        mean: The mean for the normalization layer. If the input tensors have 3 channels,
            this should be a list of 3 floats. Otherwise, this should be a single float.
        std: The standard deviation for the normalization layer.
            If the input tensors have 3 channels, this should be a list of 3 floats. Otherwise, this should be a single float.

    Attributes:
        normalize_layer (Union[RGBNormalizeLayer, GrayScaleNormalizeLayer]): The normalization layer for the input tensors.
        model (nn.Module): The wrapped model.

    Typical usage example:
        in_channels = 1
        std = [1.0]
        mean = [0.0]

        model = ResNetChengCRNN(60, in_channels, 256, 2)
        x = torch.rand(size=[16, 3, 32, 256])

        wrapped_model = ModelWrapper(
            model=model,
            mean=mean,
            std=std
        )
        wrapped_out = wrapped_model(x)
    """
    def __init__(
            self,
            model: nn.Module,
            mean: Union[List[float], float],
            std: Union[List[float], float]
    ):
        super().__init__()
        self.normalize_layer = RGBNormalizeLayer(mean, std)
        self.model = model
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor and forwards it through the wrapped model.

        This function normalizes the input tensor using the normalization layer, then forwards it through the wrapped model.
        The output of the model is then passed through a softmax function and rearranged to have the batch size
        as the second dimension.

        Args:
            x: A 4D tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: A 3D tensor with shape (batch_size, sequence_length, num_classes).
        """
        out = self.normalize_layer.forward(x)
        out = self.model(out)

        #y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        out = self.sigmoid(out[:, 0, :, :]) # > 0.5).to(torch.float).unsqueeze(0)

        return out.unsqueeze(0)


class ModelWrapperTextDetection(nn.Module):
    """A wrapper class for PyTorch models.

    This class wraps a PyTorch model and normalizes the input tensors before forwarding them through the model.
    The normalization layer can be either an instance of `RGBNormalizeLayer` or `GrayScaleNormalizeLayer`,
    depending on the input shape of the tensors.

    Args:
        model: The model to wrap.
        mean: The mean for the normalization layer. If the input tensors have 3 channels,
            this should be a list of 3 floats. Otherwise, this should be a single float.
        std: The standard deviation for the normalization layer.
            If the input tensors have 3 channels, this should be a list of 3 floats. Otherwise, this should be a single float.

    Attributes:
        normalize_layer (Union[RGBNormalizeLayer, GrayScaleNormalizeLayer]): The normalization layer for the input tensors.
        model (nn.Module): The wrapped model.

    Typical usage example:
        in_channels = 1
        std = [1.0]
        mean = [0.0]

        model = ResNetChengCRNN(60, in_channels, 256, 2)
        x = torch.rand(size=[16, 3, 32, 256])

        wrapped_model = ModelWrapper(
            model=model,
            mean=mean,
            std=std
        )
        wrapped_out = wrapped_model(x)
    """
    def __init__(
            self,
            model: nn.Module,
            mean: Union[List[float], float],
            std: Union[List[float], float]
    ):
        super().__init__()
        self.normalize_layer = RGBNormalizeLayer(mean, std)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor and forwards it through the wrapped model.

        This function normalizes the input tensor using the normalization layer, then forwards it through the wrapped model.
        The output of the model is then passed through a softmax function and rearranged to have the batch size
        as the second dimension.

        Args:
            x: A 4D tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: A 3D tensor with shape (batch_size, sequence_length, num_classes).
        """
        out = self.normalize_layer.forward(x)
        out = self.model(out)[:, 0, :, :]
        return out


class ModelWrapperSingleTextDetection(nn.Module):
    """A wrapper class for PyTorch models.

    This class wraps a PyTorch model and normalizes the input tensors before forwarding them through the model.
    The normalization layer can be either an instance of `RGBNormalizeLayer` or `GrayScaleNormalizeLayer`,
    depending on the input shape of the tensors.

    Args:
        model: The model to wrap.
        mean: The mean for the normalization layer. If the input tensors have 3 channels,
            this should be a list of 3 floats. Otherwise, this should be a single float.
        std: The standard deviation for the normalization layer.
            If the input tensors have 3 channels, this should be a list of 3 floats. Otherwise, this should be a single float.

    Attributes:
        normalize_layer (Union[RGBNormalizeLayer, GrayScaleNormalizeLayer]): The normalization layer for the input tensors.
        model (nn.Module): The wrapped model.

    Typical usage example:
        in_channels = 1
        std = [1.0]
        mean = [0.0]

        model = ResNetChengCRNN(60, in_channels, 256, 2)
        x = torch.rand(size=[16, 3, 32, 256])

        wrapped_model = ModelWrapper(
            model=model,
            mean=mean,
            std=std
        )
        wrapped_out = wrapped_model(x)
    """
    def __init__(
            self,
            model: nn.Module,
            mean: Union[List[float], float],
            std: Union[List[float], float]
    ):
        super().__init__()
        self.normalize_layer = RGBNormalizeLayer(mean, std)
        self.model = model
        self.sigm = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalizes the input tensor and forwards it through the wrapped model.

        This function normalizes the input tensor using the normalization layer, then forwards it through the wrapped model.
        The output of the model is then passed through a softmax function and rearranged to have the batch size
        as the second dimension.

        Args:
            x: A 4D tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: A 3D tensor with shape (batch_size, sequence_length, num_classes).
        """
        out = self.normalize_layer.forward(x)
        out = self.model(out)
        bin_out = out[:, 0, :, :]
        line_out = self.sigm(out[:, 3, :, :])

        return torch.stack([bin_out, line_out], dim=1)

