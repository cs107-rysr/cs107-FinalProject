import spladtool_forward as st
import spladtool_forward.functional as F

f = lambda x: x - F.exp(-2.0 * F.sin(4.0 * x) * F.sin(4.0 * x))

def newton(f, x_k, tol=1.0e-8, max_it=100):
    """Newton Raphson method using spladtool_forward Autodifferentiation package"""
    x_k = st.Tensor(x_k)
    root = None
    for k in range(max_it):
        y = f(x_k)
        dx_k = - y.data / y.grad
        if (abs(dx_k) < tol):
            root = x_k + dx_k
            print(f"Found root {root.data} at iter {k+1}")
            break
        print(f"Iter {k+1}: Dx_k = {dx_k}")
        x_k += dx_k
    return root.data
