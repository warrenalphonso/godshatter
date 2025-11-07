from collections import defaultdict


class Value:
    def __init__(self, value, grad=lambda x, _: x, deps=None):
        self.data = value
        self.grad_fn = grad
        self.deps = deps or []
        self.computed_grad = 0

    @property
    def grad(self):
        return self.computed_grad

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.computed_grad})"

    def __neg__(self):
        "gradient of next node wrt to us is -1"
        return Value(-self.data, grad=lambda x, _: -x, deps=[self])

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.data + other, grad=lambda x, _: x, deps=[self])

        return Value(self.data + other.data, grad=lambda x, _: x, deps=[self, other])

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return -other + self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.data * other, grad=lambda x, _: other * x, deps=[self])
        return Value(self.data * other.data, grad=lambda x, parent: other.data * x if parent is self else self.data * x, deps=[self, other])

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, denominator):
        return 1/denominator * self

    def __rtruediv__(self, numerator):
        "d/dx (c/x) = -c/x^2"
        return Value(numerator / self.data, grad=lambda x, _: -numerator / (self.data ** 2) * x, deps=[self])

    def __pow__(self, other):
        return Value(self.data ** other, grad=lambda x, _: other * (self.data ** (other - 1)) * x, deps=[self])

    def relu(self):
        return Value(self.data if self.data > 0 else 0.0, grad=lambda x, _: x if self.data > 0 else 0.0, deps=[self])

    def backward(self, parent_grad=None):
        "Compute gradients back through the DAG"
        dag = defaultdict(set)
        children = defaultdict(list)

        stack = [self]
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for parent in node.deps:
                dag[parent].add(node)
                children[parent].append(node)
                stack.append(parent)

        self.computed_grad = 1
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            if not dag[v]:
                # Pass gradient from each child
                for child in children[v]:
                    v.computed_grad += child.grad_fn(child.computed_grad, child.deps[0] if len(child.deps) == 1 or child.deps[1] is not v else child.deps[1])
                visited.add(v)
                for parent in v.deps:
                    dag[parent].discard(v)
                    if not dag[parent]:
                        build_topo(parent)

        build_topo(self)

if __name__ == "__main__":
    # Simple test
    x = Value(2.0)
    w = x * x
    z = w + w
    # y = x + x + 2 + x + 1
    # z = 2 * y * x + x
    z.backward()
    print(f"{x=}, {w=}, {z=}")
