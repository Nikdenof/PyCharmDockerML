import torch
from src.model import SimpleMLP


def main():
    model = SimpleMLP(input_size=10, hidden_size=20, output_size=1)
    output = model(torch.randn(1, 10))
    print(output.item())


if __name__ == "__main__":
    main()
