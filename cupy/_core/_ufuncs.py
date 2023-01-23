from cupy._core._kernel import create_ufunc, _Op


def _fix_to_sctype(dtype, sctype):
    if dtype.type == sctype:
        return dtype
    elif dtype.kind == "S":
        length = dtype.itemsize
    elif dtype.kind == "U":
        length = dtype.itemsize // 4
    else:
        raise ValueError("CuPy currently only supports string conversions.")

    return np.dtype((sctype, length))

def _s_copy_resolver(op, arginfo):
    # Support only U->S and S->U casts right now
    sctype = op.in_type[0]

    in_dtype = _fix_to_sctype(arginfo[0].dtype, sctype)
    out_dtype = in_dtype  # could call _fix_to_sctype just to sanity check

    return in_dtype, out_dtype


elementwise_copy = create_ufunc(
    'cupy_copy',
    ('?->?', 'b->b', 'B->B', 'h->h', 'H->H', 'i->i', 'I->I', 'l->l', 'L->L',
     'q->q', 'Q->Q', 'e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    'out0 = in0',
    default_casting='unsafe', custom_ops=[
        _Op((np.bytes_,), (np.bytes_,), "out0 = in0", None, _s_copy_resolver),
        _Op((np.str_,), (np.str_,), "out0 = in0", None, _s_copy_resolver),
    ])
