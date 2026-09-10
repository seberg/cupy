from __future__ import annotations

from cupy._core._kernel import create_ufunc
from cupy._util import bc32_loop, bf16_loop, c32_loop


elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', *bf16_loop(), 'f->f', 'd->d',
     *c32_loop(), *bc32_loop(), 'F->F', 'D->D',
     ('V->V', None, lambda op, x, y: (y, y))),  # copy casts input to output.
    'out0 = in0',
    default_casting='unsafe')
