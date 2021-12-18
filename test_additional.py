import spladtool.spladtool_forward as sf

x = sf.tensor([1, 2])
f = sf.exp(-(x[0] - 2) ** 2 - (x[1] - 1) ** 2)
print(f.grad)