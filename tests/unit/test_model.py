# tests/unit/test_model.py
import torch
from src.rnn_pipeline.models.rnn import TextClassifier
def test_output_shape(model, batch):
    output = model(batch)
    assert output.shape == (16, 3)
 
def test_no_nan_in_output(model, batch):
    output = model(batch)
    assert not torch.isnan(output).any()
 
def test_gradients_flow(model, batch):
    model(batch).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad: {name}"
 
def test_overfit_single_batch(model):
    x, y = torch.randint(0,200,(8,10)), torch.tensor([0,1,2,0,1,2,0,1])
    init_loss = None
    for _ in range(50):
        loss = criterion(model(x), y) # type: ignore
        loss.backward()
        optimizer.step() # type: ignore
        init_loss = init_loss or loss.item()
    assert loss.item() < init_loss * 0.5 # type: ignore
 
def test_eval_mode_is_deterministic(model, batch):
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), model(batch))
