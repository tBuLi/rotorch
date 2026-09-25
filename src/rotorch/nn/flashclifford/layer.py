import math

import torch
from kingdon import MultiVector
from sympy import Symbol, erf
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter

from ..cgenn import MVLinear
from ..cgenn.gp import number_of_weights_wgp, wgp
from ..utils import EPS, materialize_constants, register


def gelu(x):
    """The GELU gate, :math:`\\Phi(x) = (1 + \\mathrm{erf}(x / \\sqrt{2})) / 2`."""
    return 0.5 * (1 + erf(x / math.sqrt(2)))


def gelu_wgp(X: MultiVector, Y: MultiVector, weights: MultiVector[None]) -> MultiVector:
    """
    Both operands gated by the GELU of their scalar part, then their weighted geometric product: what flash-clifford fuses into one kernel, short of its normalization.
    Compiled over sympy symbols, so that the gate is differentiated along with the product.

    The gated operands enter the product as symbols of their own, and the gated coefficients take their place afterwards.
    Kingdon expands a product of symbolic coefficients, which would multiply every term of the product out against both gates: in Cl(3) that is 2.5 times the multiplications forward and twice backward.
    """
    gated = {Symbol(f'{v}_gelu'): v * gelu(M.e) for M in (X, Y) for v in M.values()}
    stand_in = lambda M: M.map(lambda v: Symbol(f'{v}_gelu'))
    return wgp(stand_in(X), stand_in(Y), weights).map(lambda v: v.subs(gated))


def rms_norm(X: MultiVector) -> MultiVector:
    """Every grade divided by the root of its squared coefficients, summed over its blades and averaged over the features: flash-clifford's grade-wise RMSNorm."""
    grades = [X.grade(g).values() for g in X.grades]
    return X.fromkeysvalues(X.algebra, X.keys(), torch.cat([v * torch.rsqrt((v * v).sum(0).mean(-1, keepdim=True) + EPS) for v in grades]))


class Layer(LazyModuleMixin, nn.Module):
    """
    Known as Layer in flash-clifford: a grade-wise linear map of the input, input and map both gated by the GELU of their scalar part, their weighted geometric product,
    and grade-wise RMS normalization. flash-clifford offers it in place of the nonlinearity, product and normalization of a layer of cgenn's CEMLP.

    The gating and the product are one kingdon operator, :func:`gelu_wgp`, which the triton backend turns into one kernel forward and one backward, as flash-clifford writes
    them by hand, in any dimension rather than in two and three only. Like flash-clifford's, the normalization is equivariant for a Euclidean algebra.
    Unlike flash-clifford, the bias of the linear map goes on the scalar part only: added to every blade, as flash-clifford adds it, it would break equivariance.
    """

    weight: UninitializedParameter

    def __init__(self, features, normalize=True):
        super().__init__()
        self.features = features
        self.normalize = normalize
        self.weight = UninitializedParameter()
        self.linear = MVLinear(features, features)

    def initialize_parameters(self, input: MultiVector):
        if not self.has_uninitialized_params():
            return

        self.algebra = input.algebra
        self.gelu_wgp = register(self.algebra, gelu_wgp, codegen_symbolcls=Symbol)

        with torch.no_grad():
            self.weight.materialize((number_of_weights_wgp(input, self.linear(input)), self.features))
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=1 / math.sqrt(self.algebra.d + 1))
        # flash-clifford's, which is smaller than MVLinear's own.
        nn.init.normal_(self.linear.weight, std=1 / math.sqrt(self.features * (self.algebra.d + 1)))

    def forward(self, input: MultiVector) -> MultiVector:
        input = materialize_constants(input)
        product = self.gelu_wgp(input, self.linear(input), input.algebra.scalar(e=self.weight))
        return rms_norm(product) if self.normalize else product
