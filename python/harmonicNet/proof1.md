
input f(x): (F = R^n => S) (can be a graph, use R^n for generalisability)
fully connected layer w(x, x+): R^n x R^n => S
morphing operator: A {f(x)} = AL(f(AR(x))): F => F

output g(x+): R^n => S = <f(x), w(x, x+)> dx

equivariant condition: <A {f(x)}, w(x, x+)> dx = Q {<f(x), w(x, x+)> dx}    \forall f(x)

assuming only AR is in consideration (e.g. a translation):

<f(AR x), w(x, x+)> dx = <f(x), w(x, QR x+))> dx

<f(x), w(AR^-1 x, x+)> dx = <f(x), w(x, QR x+))> dx     \forall f(x)

<f(x), w(AR^-1 x, QR^-1 y+)> dx = <f(x), w(x, y+))> dx     \forall f(x)

w(AR^-1 x, QR^-1 y+) = w(x, y+)

w(AR x, QR y+) = w(x, y+)

as a result:

assuming that x+ = H x, v(x) = w(H x, QH H x) = w(x, H x), then:

g(x+) = g(H x) = <f(x), w(x, H x)> dH =

assuming that x+ = Q x0, then

$$
g(x+) = g(Q x0) = <f(y), w(y, Q x0)> dy
    = <f(y), w(P^-1 y, x0)> dy
    = conv(f(y), w(y, x0))(P^-1)
$$

QED
