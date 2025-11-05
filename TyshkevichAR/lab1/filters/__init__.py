from .resolution import change_resolution
from .sepia import apply_sepia
from .vignette import apply_vignette
from .pixelation import apply_pixelation
from .simple_border import apply_simple_border
from .shaped_border import apply_shaped_border
from .lens_flare import apply_lens_flare
from .watercolor_paper import apply_watercolor_paper

__all__ = [
    'change_resolution',
    'apply_sepia',
    'apply_vignette',
    'apply_pixelation',
    'apply_simple_border',
    'apply_shaped_border',
    'apply_lens_flare',
    'apply_watercolor_paper'
]