"""Export the trained cat/dog model to ONNX and verify it matches PyTorch."""
import sys
import numpy as np
import torch
from torchvision import transforms as T
from PIL import Image

# The ONNX exporter prints unicode (e.g. checkmarks); force UTF-8 so it
# doesn't crash on consoles with a non-UTF-8 codepage (e.g. Windows cp1251).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
import timm
import onnx
import onnxruntime as ort
from pathlib import Path


# ----------------------- CONFIG -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_NAME   = "rexnet_150"
SAVE_PREFIX  = "cat_dog"
WEIGHTS      = PROJECT_ROOT / "saved_models" / f"{SAVE_PREFIX}_best_model.pth"
ONNX_PATH    = PROJECT_ROOT / "saved_models" / f"{SAVE_PREFIX}_model.onnx"
CLASSES      = ["cat", "dog"]
IM_SIZE      = 224
OPSET        = 13
# ------------------------------------------------------


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Trained weights not found at {WEIGHTS}. Run `python main.py` first.")

    # 1) Load the trained PyTorch model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()

    # 2) Export to ONNX (dynamic batch axis -> any batch size works at inference)
    dummy_input = torch.randn(1, 3, IM_SIZE, IM_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        export_params=True,
        opset_version=OPSET,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,   # use the classic exporter (respects opset & dynamic_axes directly)
    )
    print(f"Exported ONNX -> {ONNX_PATH}")

    # 3) Validate the ONNX graph structure
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print("ONNX model structure is valid.")

    # 4) Verify outputs match between PyTorch and ONNX Runtime
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(str(ONNX_PATH))
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]

    preprocess = T.Compose([T.Resize((224,224)),                        
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

    test_cat_dir = PROJECT_ROOT / "datasets" / "cat_dog" / "dataset" / "test" / "cat"
    image_path = sorted(test_cat_dir.glob("*.jpg"))[0]  
    image = Image.open(image_path).convert('RGB')

    input_tensor = preprocess(image).unsqueeze(0)
    input_numpy = input_tensor.numpy()

    input_name = session.get_inputs()[0].name
    print(input_name)

    output = session.run(None, {input_name: input_numpy})

    output_array = output[0]
    predicted_class = np.argmax(output_array, axis=1)
    print(f"Predicted class index: {predicted_class[0]}")
    predicted_class_name = CLASSES[predicted_class[0]]
    
    print(f"Predicted class name: {predicted_class_name}")
    

    max_diff = np.abs(torch_out - onnx_out).max()
    print(f"Max difference PyTorch vs ONNX: {max_diff:.2e}")
    print("PASSED: outputs match." if max_diff < 1e-3 else "WARNING: outputs differ more than expected.")


if __name__ == "__main__":
    main()
