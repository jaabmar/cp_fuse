import pytest
import torch
from unittest.mock import MagicMock
from transformers import GenerationConfig
from cp_fuse.cp_model import CPModel


@pytest.fixture
def mock_models():
    """Fixture to create mock models with parameters on a specific device."""
    model1 = MagicMock()
    model2 = MagicMock()

    # Mock configs
    model1.config = MagicMock(
        vocab_size=30522,
        pad_token_id=0,
        eos_token_id=2,
    )
    model2.config = MagicMock(
        vocab_size=30522,
        pad_token_id=0,
        eos_token_id=2,
    )

    # Mock parameters
    param1 = torch.nn.Parameter(torch.randn(1, 1, device="cpu"))
    param2 = torch.nn.Parameter(torch.randn(1, 1, device="cpu"))
    model1.parameters.return_value = iter([param1])
    model2.parameters.return_value = iter([param2])

    def to_device_model1(device):
        for param in model1.parameters():
            param.data = param.data.to(device)

    def to_device_model2(device):
        for param in model2.parameters():
            param.data = param.data.to(device)

    model1.to.side_effect = to_device_model1
    model2.to.side_effect = to_device_model2

    return model1, model2


@pytest.fixture
def cp_model(mock_models):
    """Fixture to create a CPModel instance."""
    model1, model2 = mock_models
    return CPModel(model1, model2, grid_size=5, verbose=True, device="cuda")

def test_generate_input_validation(cp_model):
    """Test validation of inputs during generation."""
    input_ids = torch.tensor([[1, 2, 3]])
    generation_config = GenerationConfig(
        max_new_tokens=256,
        num_return_sequences=1,
        eos_token_id=2,
        pad_token_id=0,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
    )

    # Test invalid max_length
    with pytest.raises(ValueError, match="`max_length` should be a strictly positive integer."):
        generation_config.max_length = -1
        cp_model.generate(input_ids, generation_config)

    # Test max_new_tokens overriding max_length
    generation_config.max_length = 10
    generation_config.max_new_tokens = 5
    cp_model._validate_generate_inputs(input_ids, generation_config)
    assert generation_config.max_length == 8  # 3 (input length) + 5 (max_new_tokens)

    # Test invalid temperature for sampling
    generation_config.do_sample = True
    with pytest.raises(ValueError, match="`temperature` should be positive for sampling decoding."):
        generation_config.temperature = 0
        cp_model.generate(input_ids, generation_config)

    
def test_generate_basic(cp_model):
    """Test basic generation flow with optimized settings."""
    input_ids = torch.tensor([[1, 2, 3]])
    generation_config = GenerationConfig(
        max_new_tokens=2, 
        num_return_sequences=1,
        eos_token_id=2,
        pad_token_id=0,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
    )

    # Mock forward passes with a smaller vocabulary size for faster computation
    cp_model.model_forward = MagicMock(
        side_effect=[
            (torch.log_softmax(torch.randn(1, 100), dim=-1), None), 
            (torch.log_softmax(torch.randn(1, 100), dim=-1), None),
        ]
    )

    # Mock _decode to avoid running the actual generation loop
    cp_model._decode = MagicMock(
        return_value=MagicMock(
            sequences=torch.tensor([[1, 2, 3, 4, 5]]),  # Mocked output sequences
            logits=None,
        )
    )

    output = cp_model.generate(input_ids, generation_config)

    # Check that the mocked _decode was called
    cp_model._decode.assert_called_once()

    # Validate output sequence shape
    assert output.sequences.shape[0] == input_ids.shape[0]
    assert output.sequences.shape[1] == 5  # Length of the mocked sequence

    
def test_solve_optimization(cp_model):
    """Test optimization of weights."""
    logits1 = torch.randn(2, 30522)
    logits2 = torch.randn(2, 30522)
    path_logprob1 = torch.zeros((2, 1))
    path_logprob2 = torch.zeros((2, 1))

    b0, b1, b2 = cp_model.solve_optimization(logits1, logits2, path_logprob1, path_logprob2, grid_size=3)
    assert b0.shape == path_logprob1.shape
    assert b1.shape == path_logprob1.shape
    assert b2.shape == path_logprob1.shape


def test_prepare_attention_mask(cp_model):
    """Test attention mask preparation."""
    input_ids = torch.tensor([[1, 2, 3, 0, 0]])
    generation_config = GenerationConfig(pad_token_id=0)

    attention_mask = cp_model._prepare_attention_mask(input_ids, None, generation_config)
    assert (attention_mask == torch.tensor([[1, 1, 1, 0, 0]])).all()
    
def test_objective(cp_model):
    """Test objective function computation."""
    logits1 = torch.log_softmax(torch.randn(2, 30522), dim=-1) # [batch_size, vocab_size]
    logits2 = torch.log_softmax(torch.randn(2, 30522), dim=-1) # [batch_size, vo cab_size]
    path_logprob1 = torch.zeros((2, 1))
    path_logprob2 = torch.zeros((2, 1))
    b1_expanded = torch.zeros((2, 4, 30522)) # (batch_size, grid_size^2, vocab_size)
    b2_expanded = torch.ones((2, 4, 30522)) # (batch_size, grid_size^2, vocab_size)
    b0 = torch.ones((2, 1))

    loss = cp_model.objective(b0.unsqueeze(1), b1_expanded, b2_expanded, logits1.unsqueeze(1), 
                              logits2.unsqueeze(1), path_logprob1, path_logprob2)
    assert loss.shape == torch.Size([2, 4])  # [batch_size, grid_size^2]
    
def test_get_logits(cp_model):
    """Test combined logits computation."""
    logits1 = torch.log_softmax(torch.randn(2, 30522), dim=-1)
    logits2 = torch.log_softmax(torch.randn(2, 30522), dim=-1)
    b0 = torch.zeros((2, 1))
    b1 = torch.ones((2, 1))
    b2 = torch.ones((2, 1))

    combined_logits = cp_model._get_logits(b0, b1, b2, logits1, logits2)
    assert combined_logits.shape == logits1.shape


def test_optimize_grid(cp_model):
    """Test grid optimization for weights."""
    logits1 = torch.log_softmax(torch.randn(2, 30522), dim=-1)
    logits2 = torch.log_softmax(torch.randn(2, 30522), dim=-1)
    path_logprob1 = torch.zeros((2, 1))
    path_logprob2 = torch.zeros((2, 1))

    b0, b1, b2 = cp_model._optimize_grid(logits1, logits2, path_logprob1, path_logprob2, grid_size=3)
    assert b0.shape == torch.Size([2, 1])
    assert b1.shape == torch.Size([2, 1])
    assert b2.shape == torch.Size([2, 1])
