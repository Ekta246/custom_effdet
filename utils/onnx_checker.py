import onnx
onnx_model='./effdet0_baseline.onnx'
x=onnx.checker.check_model(onnx_model)
print(x)

