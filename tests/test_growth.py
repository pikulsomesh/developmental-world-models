import torch

from dwm.models import SenseWorldModel


def _check(sense, obs_shape):
    torch.manual_seed(0)
    m = SenseWorldModel(sense, width=1, window=1)
    o = torch.rand(2, 8, *obs_shape)
    a = torch.rand(2, 8, 2)
    z0, zh0 = m(o, a)
    r0 = m.decode_seq(zh0)
    for w, k in [(2, 2), (3, 4), (4, 6)]:
        m.grow(w, k)
        z, zh = m(o, a)
        assert torch.allclose(z, z0, atol=1e-5), sense
        assert torch.allclose(m.decode_seq(zh), r0, atol=1e-5), sense


def test_growth_preserves_function():
    _check("vision", (3, 32, 32))
    _check("audio", (64,))
    _check("touch", (64,))
    _check("proprio", (4,))
