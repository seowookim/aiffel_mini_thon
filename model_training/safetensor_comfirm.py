from safetensors import safe_open
import torch

# safetensors 파일 경로
safetensors_file_path = "./results_kogpt2_gen/checkpoint-227260/adapter_model.safetensors"

# safetensors 파일 열기
with safe_open(safetensors_file_path, framework="pt") as f:
    # 키 목록 출력
    keys = f.keys()
    print(keys)

    # 각 키에 대한 텐서 가져오기 및 출력 (처음 몇 개 값만 출력)
    for key in keys:
        weights = f.get_tensor(key)
        print(f"Key: {key}, Weights: {weights[:5]}")  # 처음 5개의 값만 출력