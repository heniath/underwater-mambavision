import torch

from mambavision.mambavision import (
    Attention,
    DropPath,
    MambaVision,
    MambaVisionConfig,
    MambaVisionMixer,
    mambavision_t,
    window_partition,
    window_reverse,
)


def tiny_config(**overrides):
    values = dict(
        num_classes=7,
        stem_dim=8,
        dims=(8, 16, 24, 32),
        depths=(1, 1, 2, 2),
        num_heads=(1, 2, 3, 4),
        window_sizes=(4, 4, 4, 2),
        state_size=4,
        drop_path_rate=0.2,
    )
    values.update(overrides)
    return MambaVisionConfig(**values)


def test_window_partition_round_trip_with_padding():
    x = torch.randn(2, 5, 11, 7)
    windows, original_hw, padded_hw = window_partition(x, 4)
    restored = window_reverse(windows, 4, original_hw, padded_hw)
    assert windows.shape == (12, 16, 5)
    torch.testing.assert_close(restored, x)


def test_mixer_dimensions_and_finite_gradients():
    mixer = MambaVisionMixer(dim=12, state_size=3)
    x = torch.randn(2, 9, 12, requires_grad=True)
    output = mixer(x)
    assert mixer.branch_dim == 6
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in mixer.parameters())


def test_hybrid_placement_and_drop_path_eval_behavior():
    model = MambaVision(tiny_config())
    for stage in model.stages[2:]:
        assert isinstance(stage.blocks[0].mixer, MambaVisionMixer)
        assert isinstance(stage.blocks[-1].mixer, Attention)
    layer = DropPath(0.75).eval()
    x = torch.randn(3, 4, 5)
    torch.testing.assert_close(layer(x), x)


def test_small_model_arbitrary_resolution_forward_backward_and_selection():
    torch.manual_seed(1)
    model = MambaVision(tiny_config())
    x = torch.randn(2, 3, 65, 57, requires_grad=True)
    logits = model(x)
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    model.eval()
    with torch.no_grad():
        selected = model.forward_features(x.detach(), out_indices=(3, 1))
        embedding = model.forward_embedding(x.detach())
        first = model(x.detach())
        second = model(x.detach())
    assert [feature.shape for feature in selected] == [(2, 32, 3, 2), (2, 16, 9, 8)]
    assert embedding.shape == (2, 32)
    torch.testing.assert_close(first, second)


def test_mambavision_t_defaults_and_224_stage_progression():
    model = mambavision_t().eval()
    assert model.config.dims == (80, 160, 320, 640)
    assert model.config.depths == (1, 3, 8, 4)
    assert model.config.num_heads == (2, 4, 8, 16)
    assert model.config.window_sizes == (8, 8, 14, 7)

    # Stage shape checks use hooks so this single pass also validates logits and
    # embedding without running the relatively slow native scan three times.
    shapes = []
    handles = [stage.register_forward_hook(lambda _, __, out: shapes.append(out.shape)) for stage in model.stages]
    with torch.no_grad():
        embedding = model.forward_embedding(torch.randn(1, 3, 224, 224))
        logits = model.head(embedding)
    for handle in handles:
        handle.remove()
    assert shapes == [
        torch.Size((1, 80, 56, 56)),
        torch.Size((1, 160, 28, 28)),
        torch.Size((1, 320, 14, 14)),
        torch.Size((1, 640, 7, 7)),
    ]
    assert embedding.shape == (1, 640)
    assert logits.shape == (1, 1000)
