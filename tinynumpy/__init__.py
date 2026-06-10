import builtins as _builtins
import math as _math

nan = _builtins.float("nan")
inf = _builtins.float("inf")

float64 = _builtins.float
float32 = _builtins.float
int64 = _builtins.int
int32 = _builtins.int
bool_ = _builtins.bool
object_ = _builtins.object


class ndarray(list):
    def __init__(self, data=()):
        if isinstance(data, ndarray):
            data = data.tolist()

        if isinstance(data, _builtins.list):
            super().__init__(data)
        elif isinstance(data, _builtins.tuple):
            super().__init__(list(data))
        else:
            super().__init__([[data]])

    def view(self, cls=None):
        if cls is None or cls is ndarray:
            return self
        try:
            self.__class__ = cls
            return self
        except Exception:
            return self

    @property
    def shape(self):
        if len(self) == 0:
            return (0,)
        first = self[0]
        if isinstance(first, (_builtins.list, _builtins.tuple, ndarray)):
            return (len(self), len(first))
        return (len(self),)

    @property
    def size(self):
        s = self.shape
        if len(s) == 2:
            return s[0] * s[1]
        return s[0]

    def tolist(self):
        return [x.tolist() if isinstance(x, ndarray) else x for x in self]

    def reshape(self, shape, *shapes, order='C'):
        return self

    def collapse(self, shape):
        return self

    def __ne__(self, other):
        return ndarray([[x != other for x in row] if isinstance(row, list) else row != other for row in self])

    def __eq__(self, other):
        return ndarray([[x == other for x in row] if isinstance(row, list) else row == other for row in self])


class errstate:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _is_seq(x):
    return isinstance(x, (_builtins.list, _builtins.tuple, ndarray))


def array(a=(), dtype=None, copy=True, order=None, subok=False, ndmin=0):
    if isinstance(a, ndarray):
        return a
    return ndarray(a)


def asarray(a, dtype=None, order=None):
    return array(a, dtype=dtype)


def asanyarray(a, dtype=None, order=None):
    return asarray(a, dtype=dtype)


def atleast_1d(a):
    if _is_seq(a):
        return a
    return ndarray([a])


def shape(a):
    if hasattr(a, "shape"):
        return a.shape
    if _is_seq(a):
        try:
            if _is_seq(a[0]):
                return (len(a), len(a[0]))
        except Exception:
            pass
        return (len(a),)
    return ()


def resize(a, new_shape):
    return asarray(a)


def where(cond, x, y):
    if isinstance(cond, ndarray):
        out = []
        for row in cond:
            if isinstance(row, list):
                out.append([x if c else y for c in row])
            else:
                out.append(x if row else y)
        return ndarray(out)
    return x if cond else y


def broadcast(*args):
    return args


def frompyfunc(func, nin, nout):
    def wrapped(*args):
        return vectorize(func)(*args)
    return wrapped


def vectorize(pyfunc, otypes=None, excluded=None, signature=None):
    excluded = set(excluded or [])

    def wrapped(*args, **kwargs):
        # Scalar mode.
        if not any(_is_seq(a) for a in args):
            return ndarray([[pyfunc(*args, **kwargs)]])

        # Find first sequence-like argument.
        seq = None
        for i, a in enumerate(args):
            if i not in excluded and _is_seq(a):
                seq = a
                break

        if seq is None:
            return ndarray([[pyfunc(*args, **kwargs)]])

        # Flatten-ish 1x1 / list behavior.
        try:
            rows = seq
            out = []
            for r_index, row in enumerate(rows):
                if _is_seq(row):
                    out_row = []
                    for c_index, _ in enumerate(row):
                        aa = []
                        for i, a in enumerate(args):
                            if i in excluded or not _is_seq(a):
                                aa.append(a)
                            else:
                                try:
                                    aa.append(a[r_index][c_index])
                                except Exception:
                                    aa.append(a)
                        out_row.append(pyfunc(*aa, **kwargs))
                    out.append(out_row)
                else:
                    aa = []
                    for i, a in enumerate(args):
                        if i in excluded or not _is_seq(a):
                            aa.append(a)
                        else:
                            try:
                                aa.append(a[r_index])
                            except Exception:
                                aa.append(a)
                    out.append(pyfunc(*aa, **kwargs))
            return ndarray(out)
        except Exception:
            return ndarray([[pyfunc(*args, **kwargs)]])

    return wrapped


def choose(a, choices, out=None, mode='raise'):
    def one(i):
        i = _builtins.int(i)
        if mode == 'clip':
            i = max(0, min(i, len(choices) - 1))
        elif mode == 'wrap':
            i = i % len(choices)
        return choices[i]

    if _is_seq(a):
        return ndarray([one(i) for i in a])
    return one(a)


def isnan(x):
    try:
        return x != x
    except Exception:
        return False


def isfinite(x):
    try:
        return not isnan(x) and x != inf and x != -inf
    except Exception:
        return False


def isinf(x):
    try:
        return x == inf or x == -inf
    except Exception:
        return False


def sum(a, axis=None, dtype=None, out=None):
    return _builtins.sum(a)


def prod(a, axis=None, dtype=None, out=None):
    r = 1
    for x in a:
        r *= x
    return r


def maximum(a, b):
    return a if a >= b else b


def minimum(a, b):
    return a if a <= b else b


def abs(x):
    return _builtins.abs(x)


def floor(x):
    return _math.floor(x)


def ceil(x):
    return _math.ceil(x)


def round(x, decimals=0):
    return _builtins.round(x, decimals)


try:
    import tinynumpy.tinylinalg as linalg
except Exception:
    linalg = None

def _scalar(x):
    try:
        if hasattr(x, "shape") and x.shape == (1, 1):
            return x[0][0]
    except Exception:
        pass

    try:
        if isinstance(x, (list, tuple)) and len(x) == 1:
            y = x[0]
            if isinstance(y, (list, tuple)) and len(y) == 1:
                return y[0]
    except Exception:
        pass

    return x


def vectorize(pyfunc, otypes=None, excluded=None, signature=None):
    excluded = set(excluded or [])

    def wrapped(*args, **kwargs):
        # Important formulas hack:
        # Treat 1x1 arrays as scalars.
        sargs = [_scalar(a) for a in args]

        # If everything became scalar, just call once.
        if not any(_is_seq(a) for a in sargs):
            return ndarray([[pyfunc(*sargs, **kwargs)]])

        seq = None
        for i, a in enumerate(sargs):
            if i not in excluded and _is_seq(a):
                seq = a
                break

        if seq is None:
            return ndarray([[pyfunc(*sargs, **kwargs)]])

        out = []

        for r_index, row in enumerate(seq):
            if _is_seq(row):
                out_row = []

                for c_index, _ in enumerate(row):
                    aa = []
                    for i, a in enumerate(sargs):
                        if i in excluded:
                            aa.append(a)
                        else:
                            aa.append(_scalar(a))
                    out_row.append(pyfunc(*aa, **kwargs))

                out.append(out_row)
            else:
                aa = []
                for i, a in enumerate(sargs):
                    if i in excluded:
                        aa.append(a)
                    else:
                        aa.append(_scalar(a))
                out.append(pyfunc(*aa, **kwargs))

        return ndarray(out)

    return wrapped

def _array_scalar_value(x):
    # unwrap ndarray/list shaped like [[value]]
    try:
        if hasattr(x, "shape") and x.shape == (1, 1):
            return x[0][0]
    except Exception:
        pass

    try:
        if len(x) == 1:
            y = x[0]
            if isinstance(y, (list, tuple, ndarray)) and len(y) == 1:
                return y[0]
            return y
    except Exception:
        pass

    return x


def _ndarray_float(self):
    return float(_array_scalar_value(self))


def _ndarray_int(self):
    return int(_array_scalar_value(self))


def _ndarray_bool(self):
    return bool(_array_scalar_value(self))


ndarray.__float__ = _ndarray_float
ndarray.__int__ = _ndarray_int
ndarray.__bool__ = _ndarray_bool

def _unwrap(x):
    try:
        if hasattr(x, "shape") and x.shape == (1, 1):
            return x[0][0]
    except Exception:
        pass

    try:
        if isinstance(x, (list, tuple, ndarray)) and len(x) == 1:
            y = x[0]
            if isinstance(y, (list, tuple, ndarray)) and len(y) == 1:
                return y[0]
            return y
    except Exception:
        pass

    return x


def _binary_elemwise(a, b, op):
    a = _unwrap(a)
    b = _unwrap(b)

    if isinstance(a, (list, tuple, ndarray)) and isinstance(b, (list, tuple, ndarray)):
        return ndarray([_binary_elemwise(x, y, op) for x, y in zip(a, b)])

    if isinstance(a, (list, tuple, ndarray)):
        return ndarray([_binary_elemwise(x, b, op) for x in a])

    if isinstance(b, (list, tuple, ndarray)):
        return ndarray([_binary_elemwise(a, y, op) for y in b])

    return op(a, b)


def _unary_elemwise(a, op):
    a = _unwrap(a)

    if isinstance(a, (list, tuple, ndarray)):
        return ndarray([_unary_elemwise(x, op) for x in a])

    return op(a)


def logical_and(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: bool(x) and bool(y))


def logical_or(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: bool(x) or bool(y))


def logical_not(a, *args, **kwargs):
    return _unary_elemwise(a, lambda x: not bool(x))


def logical_xor(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: bool(x) ^ bool(y))


def equal(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x == y)


def not_equal(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x != y)


def greater(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x > y)


def greater_equal(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x >= y)


def less(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x < y)


def less_equal(a, b, *args, **kwargs):
    return _binary_elemwise(a, b, lambda x, y: x <= y)

class _FakeUFunc:
    def __init__(self, func, identity=None):
        self.func = func
        self.identity = identity

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def reduce(self, a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        vals = a

        # unwrap 1x1 fake arrays
        vals = _unwrap(vals)

        # flatten nested lists/arrays
        flat = []

        def walk(x):
            x = _unwrap(x)
            if isinstance(x, (list, tuple, ndarray)):
                for y in x:
                    walk(y)
            else:
                flat.append(x)

        walk(vals)

        if not flat:
            return self.identity if initial is None else initial

        if initial is not None:
            res = initial
        else:
            res = flat[0]
            flat = flat[1:]

        for x in flat:
            res = self.func(res, x)

        return res


logical_and = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: bool(x) and bool(y)), True)
logical_or  = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: bool(x) or bool(y)), False)
logical_xor = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: bool(x) ^ bool(y)), False)

equal         = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x == y))
not_equal     = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x != y))
greater       = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x > y))
greater_equal = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x >= y))
less          = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x < y))
less_equal    = _FakeUFunc(lambda a, b, *args, **kwargs: _binary_elemwise(a, b, lambda x, y: x <= y))

# --- more fake numpy ufunc/math support for formulas ---

def _elemwise1(a, op):
    a = _unwrap(a)
    if isinstance(a, (list, tuple, ndarray)):
        return ndarray([_elemwise1(x, op) for x in a])
    return op(a)


def _elemwise2(a, b, op):
    return _binary_elemwise(a, b, op)


class _FakeUFunc:
    def __init__(self, func, identity=None):
        self.func = func
        self.identity = identity

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def reduce(self, a, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
        flat = []

        def walk(x):
            x = _unwrap(x)
            if isinstance(x, (list, tuple, ndarray)):
                for y in x:
                    walk(y)
            else:
                flat.append(x)

        walk(a)

        if not flat:
            return self.identity if initial is None else initial

        if initial is not None:
            res = initial
        else:
            res = flat[0]
            flat = flat[1:]

        for x in flat:
            res = self.func(res, x)

        return res


# logical/comparison ufunc-ish objects
logical_and = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: bool(x) and bool(y)), True)
logical_or  = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: bool(x) or bool(y)), False)
logical_xor = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: bool(x) ^ bool(y)), False)

equal         = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x == y))
not_equal     = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x != y))
greater       = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x > y))
greater_equal = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x >= y))
less          = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x < y))
less_equal    = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x <= y))


# numeric ufunc-ish objects
add      = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x + y), 0)
subtract = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x - y))
multiply = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x * y), 1)
divide   = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x / y))
true_divide = divide
power    = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, lambda x, y: x ** y))


# math functions commonly imported by formulas
sin = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.sin))
cos = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.cos))
tan = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.tan))

arcsin = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.asin))
arccos = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.acos))
arctan = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.atan))
arctan2 = _FakeUFunc(lambda a, b, *args, **kwargs: _elemwise2(a, b, _math.atan2))

asin = arcsin
acos = arccos
atan = arctan
atan2 = arctan2

sinh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.sinh))
cosh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.cosh))
tanh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.tanh))

exp = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.exp))
log = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.log))
log10 = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.log10))
sqrt = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.sqrt))

negative = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: -x))
positive = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: +x))
absolute = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _builtins.abs))
fabs = absolute

pi = _math.pi
e = _math.e

# --- extra math aliases so formulas.functions.math imports ---

arcsinh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.asinh))
arccosh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.acosh))
arctanh = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.atanh))

asinh = arcsinh
acosh = arccosh
atanh = arctanh

def radians(x):
    return _elemwise1(x, _math.radians)

def degrees(x):
    return _elemwise1(x, _math.degrees)

def sign(x):
    return _elemwise1(x, lambda v: 1 if v > 0 else (-1 if v < 0 else 0))

def trunc(x):
    return _elemwise1(x, _math.trunc)

def fix(x):
    return trunc(x)

def floor(x):
    return _elemwise1(x, _math.floor)

def ceil(x):
    return _elemwise1(x, _math.ceil)

def mod(a, b):
    return _elemwise2(a, b, lambda x, y: x % y)

remainder = mod

str_ = str
unicode_ = str
bytes_ = bytes

log2 = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, _math.log2))

isreal = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: not isinstance(x, complex) or x.imag == 0))
real = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: x.real if hasattr(x, "real") else x))
imag = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: x.imag if hasattr(x, "imag") else 0))

conj = _FakeUFunc(lambda a, *args, **kwargs: _elemwise1(a, lambda x: x.conjugate() if hasattr(x, "conjugate") else x))
conjugate = conj

def angle(z, deg=False):
    z = _unwrap(z)
    try:
        v = _math.atan2(z.imag, z.real)
        return _math.degrees(v) if deg else v
    except Exception:
        return 0

def mean(a, axis=None):
    vals = []

    def walk(x):
        x = _unwrap(x)
        if isinstance(x, (list, tuple, ndarray)):
            for y in x:
                walk(y)
        else:
            vals.append(float(x))

    walk(a)
    return sum(vals) / len(vals) if vals else 0.0

def std(a, axis=None, ddof=0):
    vals = []

    def walk(x):
        x = _unwrap(x)
        if isinstance(x, (list, tuple, ndarray)):
            for y in x:
                walk(y)
        else:
            vals.append(float(x))

    walk(a)

    if not vals:
        return 0.0

    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - ddof)
    return v ** 0.5

def var(a, axis=None, ddof=0):
    s = std(a, axis=axis, ddof=ddof)
    return s * s
