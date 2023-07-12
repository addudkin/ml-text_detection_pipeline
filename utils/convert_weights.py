import os
import sys
import torch
import torch.nn as nn

import numpy as np

from time import time
from omegaconf import DictConfig
from typing import List

sys.path.append(os.getcwd())
from utils.tools import get_config, load_json
from models import get_model
from utils.normalizer import ModelWrapperTextDetection


def tracing2jit(
        model: torch.nn.Module,
        x: torch.Tensor,
        path2model: str
) -> None:
    """Traces a PyTorch model and saves the result to disk.

    This function traces the provided PyTorch model using the provided input tensor and saves the traced model to the specified
    file path. Tracing a model allows it to be optimized for execution with the PyTorch JIT compiler, which can improve
    performance.

    Args:
        model: The PyTorch model to trace.
        x: The input tensor to use for tracing.
        path2model: The file path to save the traced model to.
    """
    traced_model = torch.jit.trace(model, x)
    print(f'Model will be saved to {path2model}')
    torch.jit.save(traced_model, path2model)


def run_jit(
        x: torch.Tensor,
        path2model: str
) -> torch.Tensor:
    """Runs a traced PyTorch model and returns the output.

    This function loads the traced PyTorch model stored at the specified file path, sets the model to evaluation mode, and
    runs the model on the provided input tensor. The function then returns the output of the model and print the spead of
    forward pass.

    Args:
        x: The input tensor to pass to the model.
        path2model: The file path to the traced model.

    Returns:
        torch.Tensor: The output of the model.
    """
    jit_model = torch.jit.load(path2model, map_location="cpu")
    jit_model.eval()

    with torch.no_grad():
        start = time()
        jit_out = jit_model(x)
        print('jit_out', jit_out.shape)
        print(f"JIT cpu FPS {1 / (time() - start)}")

    return jit_out


def average_weights(
        config: DictConfig,
        top_checkpoints: List
) -> nn.Module:
    """Averages the weights of the top models specified in the list of checkpoints.

    This function takes a list of file paths to top models and averages their weights to create a new model. The
    configuration and model type are specified in the provided `config` dictionary. The function returns the resulting
    model.

    Args:
        config: A dictionary containing the configuration for the model. This should include the model type
            and any other necessary parameters.
        top_checkpoints: A list of file paths to the top models whose weights will be averaged.

    Returns:
        nn.Module: A PyTorch model with the averaged weights.
    """
    assert len(top_checkpoints) > 0, "There aren't top checkpoints to average"

    model_dict = None
    model = None

    for checkpoint in top_checkpoints:
        config["checkpoint"]["path2best"] = checkpoint
        model = get_model(config)
        model.eval()
        if model_dict is None:
            model_dict = model.state_dict()
        else:
            for k, v in model.state_dict().items():
                model_dict[k] += v

    for k, v in model_dict.items():
        model_dict[k] = v / len(top_checkpoints)

    model.load_state_dict(model_dict)
    return model


def tracing_weights(
        model: nn.Module,
        config: DictConfig,
) -> None:
    """Trace the weights of the given PyTorch model and test it on the jit backend.

    Args:
        model: A PyTorch model.
        config: A dictionary containing configuration parameters.
    """
    # Create input_example
    input_example = torch.randint(
        0, 255, [
            1,
            3, # TODO: Потом придется параметризовать при grayscale режиме
            config["train"]["image_size"][0],
            config["train"]["image_size"][1]
        ]
    )

    model = ModelWrapperTextDetection(
        model,
        std=config["data"]["std"],
        mean=config["data"]["mean"]
    )

    path2model = os.path.join(
        os.getcwd(),
        "weights",
        "best_model_jit.pth"
    )

    # Trace model to jit
    tracing2jit(model, input_example, path2model)

    # Spead test
    with torch.no_grad():
        start = time()
        torch_class = model(input_example)
        print(f"Torch cpu FPS {1 / (time() - start)}")

    # Check outputs
    jit_class = run_jit(input_example, path2model)
    np.testing.assert_almost_equal(jit_class.data.cpu().numpy(), torch_class.data.cpu().numpy(), decimal=3)
    print("Exported model has been executed on  jit backend, and the result looks good!")


if __name__ == '__main__':
    config = get_config()

    path2weights = os.path.join(
        os.getcwd(),
        "weights",
        config["description"]["project_name"],
        config["description"]["experiment_name"]
    )

    # Tracing weights if you need using path to weights
    if config["checkpoint"]["path2best"]:
        model = get_model(config)
        tracing_weights(model, config)

    # Average weights if you need using flag -a True
    if config["checkpoint"]["do_average"]:
        path2checkpoints = os.path.join(path2weights, "top_checkpoints.json")
        print(path2checkpoints)
        assert os.path.exists(path2checkpoints), "top_checkpoints.json wasn't found"
        checkpoints = load_json(path2checkpoints)
        model = average_weights(config, checkpoints)
        path2averaged = os.path.join(path2weights, 'averaged.pth')
        print(f'model will be save into {path2averaged}')
        torch.save(model.state_dict(), path2averaged)


