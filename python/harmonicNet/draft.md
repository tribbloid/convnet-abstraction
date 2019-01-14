# Glossary

Monoid is something you can run MapReduce on:

```
apple.merge(pen).merge(pineapple).merge(pen).merge(face).merge(palm) =
apple.merge(pen)                    # core 1
  .merge(pineapple.merge(pen))      # core 2
  .merge(face.merge(palm))          # core 3
```

Group is an invertible monoid (so you can undo once)

Commutative (Abelian) monoid/group is an monoid where order of reduce doesn't matter:

```

```

Higher-order function/operator is something like reduce itself

```
insert python reduce example
```

# What is conv?
## (actually cross-correlation, but who cares)

assuming f_1, f_2: (F = R^n => S)

defined for an operator A: F => F

conv(f_1, f_2)(A) = <f_1(x), A {f_2(x)}> dx: (F => F) => S

The common assumption is that operator A can be broken into left AL and right AR:

A {f(x)} = AL(f(AR(x)))

# How does it help? - Equivariance

... define equivariance

# How does it help? - Equivariance in linear layer weights

input f(x): (F = R^n => S) (can be a graph, use R^n for generalisability)
fully connected layer w(x, x+): R^n x R^n => S
output g(x+): R^n => S = <f(x), w(x, x+)> d x
morphing operator: A {f(x)} = AL(f(AR(x))): F => F

g(x+) = <f(x), w(x, x+)> dx

condition: <A {f(x)}, w(x, x+)> dx = Q_A {<f(x), w(x, x+)> dx}    \forall f(x)

if x+ is fixed, and w(x+)(x) = w(x, x+)     # function currying, it becomes:

conv(w(x+), f)(A) = Q_A {<f(x), w(x+)(x)> dx} = Q_A {g(x+)} : objective is to get rid of x+:

assuming that only RA remains:

conv(w(x+), f)(AR) = conv(f, w(x+))(AR^-1) =
 <f(x), w(Q_AR x+)(x)> dx = conv(f, w(x+))(Q_AR)

so either:

1. for every w(x+), AR^-1 = Q_AR

2. w(AR^-1 x+) = w(AR x+)



(operator A {.} is a group action, so Q {A {.}} = Q_A {.}, where Q_A {.} is the group action of the same group)

we'll introduce some assumptions here:

if both A and Q are RIGHT LINEAR operators, then:

conv(w(x+), f)(A)) = <f(x), Q_A {w(x+)(x)}> dx = conv(f, w(x+))(Q A)

in this case the set of f(x) and g(x+) becomes a G-set

conv is now formally defined as G-conv

operator conv:

C_A {f} = conv(w(x+), f)(A)


for any 2 points x1 and x2:

conv(w(x1), f)(A) = q(x1)
conv(w(x2), f)(A) = q(x2)




write everything (function, operator as matrices!) that's the only way to use existing tools!


Q^-1 {conv(w(x+), f)(A)} = <f, w(x+)> = conv(w(x+, f)(I)

# How does it help? - if A and Q are only translation / panning

now let's confine to the case where A and Q are linear

... became conventional concept of plannar/volumetric conv

# How does it help? - if A is both translation & rotation

