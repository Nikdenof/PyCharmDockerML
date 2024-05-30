import pytest
import torch
from src.model import SimpleMLP


@pytest.fixture
def model():
    return SimpleMLP(input_size=10, hidden_size=20, output_size=1)


def test_model_creation(model):
    assert isinstance(model, SimpleMLP), "Model should be an instance of SimpleMLP"


def test_forward_pass(model):
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    assert output.shape == (1, 1), "Output shape should be (1, 1)"
    assert not torch.isnan(output).any(), "Output should not contain NaN values"


def test_training_step(model):
    input_tensor = torch.randn(1, 10)
    target = torch.randn(1, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "Loss should be a positive value"
