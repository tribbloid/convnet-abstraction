# %%

# to understand acceleration of rotation equivariant conv first let's study DFT, the simplest harmonic series:
# assuming after DFT 2 signals f_a, f_b become phi_a = [a1, a2, a3] and phi_b = [b1, b2, b3] respectively
# we claim that DFT(f_a >*< f_b) = DFT(conv(f_a, v_b)) = phi_a .* phi_b = [a1 b1, a2 b2, a3 b3]
# it goes like this:
# assuming w1, w2, w3 are Fourier bases that are orthogonal to each other
# f_a = a1 w1 + a2 w2 + a3 w3
# f_b = b1 w1 + b2 w2 + b3 w3
# DFT^-1(phi_a .* phi_b) = a1 b1 w1 + a2 b2 w2 + a3 b3 w3
# f_a >*< f_b = a1 b1 conv(w1, w1) + a2 b2 conv(w2, w2) + a3 b3 conv(w3 w3)
# remember conv(x, y) = <x, D(y, .)>, if w1, w2, w3 are orthogonal with each other their results are 0
#
# the original image (greyscale) is defined as f_a: C => R
# converting to frequency domain: phi_a: C => C
# in the mean time the filter in frequency domain phi_b: C => C
# conv becomes pointwise product: phi_a .* phi_b: C => C
# now if the image is rotated: D f_a = f_a (. e^{p i})
# DFT(D f_a) = e^{p i} phi_a: C => C
# replacing into pointwise product: e^{p i} phi_a .* phi_b
#
# eventually, it turns out that 2 properties are equally important:
# - harmonic/orthogonal basis -> conv converted to pointwise multiplication, factorisation become < , >
# - rotation equivariance -> no need to have an extra dimension for output of the layer (only 1 extra multiplication to get response after rotation)
# fourier transform is one of them that has 2 properties, but not the only one.
# what else?
# we want to construct a series of harmonic bases, assuming D: R^2 => R^2 = [D1, D2] is one of them
# in frequency domain: phi_a_d = <D, f_a> = [<D1, f_a>, <D2, f_a>]
# rotation equivariance for all D means rotation equivariance for the harmonic transformation
# to prove that D is rotation equivariant, assuming that R is the rotation operator: R f_a = f_a(R.[x, y])
# we should have:
#
# <D, R f_a> = R . <D, f_a>   \forall f_a
#
# <R^-1 D, f_a> = R . <D, f_a>
#
# as a consequence D has to be equivariant to R
# condition of equivariance that yield generalized convolution
# R is a group transformation (e.g. translation, rotation).
# for fully connected layer:
# Q <w, f> = <w, D f> = <D^-1 w, f>    \forall f
#
# the solution can only be that <w, D f> becomes a group in which


# <R^-1 [D1, D2]^T, f_a> = R . [<D1, f_a>, <D2, f_a>]^T   \forall f_a

a = 1+ 2
b = a + 3