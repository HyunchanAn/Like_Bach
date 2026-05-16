import time
import torch
import numpy as np
import os
import sys

# 프로젝트 루트를 경로에 추가하여 src 모듈을 불러올 수 있게 함
sys.path.append(os.getcwd())

from src.v4.models import BachTransformer, BachTokenizer
from src.v4.neural_engine import NeuralBachEngine

def benchmark_inference_latency(engine, iterations=3):
    print("\n--- 1. Inference Latency Benchmark ---")
    latencies = []
    
    # Mock input: 4 measures of soprano melody (C major scale)
    mock_subject = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0},
        {"pitch": 62, "duration": 1.0, "offset": 1.0},
        {"pitch": 64, "duration": 1.0, "offset": 2.0},
        {"pitch": 65, "duration": 1.0, "offset": 3.0},
    ]
    
    print(f"Testing with {iterations} iterations (4 measures SATB generation)...")
    
    for i in range(iterations):
        start_time = time.time()
        # 실제 엔진의 생성 함수 호출
        _ = engine.generate_response(mock_subject, target_measures=4)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        print(f"Iteration {i+1}: {latency*1000:.2f}ms")
    
    avg_latency = np.mean(latencies)
    print(f"Average Latency: {avg_latency*1000:.2f}ms")
    return avg_latency

def benchmark_resource_utilization():
    print("\n--- 2. Resource Utilization Benchmark ---")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Device: {device_name} ({device})")
        print(f"VRAM Total: {vram_total:.2f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Device: Apple Silicon (MPS)")
    else:
        print("Device: CPU")
    
    return device

if __name__ == "__main__":
    print("Like Bach v4.5 Performance Evaluation Suite")
    print("="*50)
    
    # 1. 환경 확인
    device = benchmark_resource_utilization()
    
    # 2. 엔진 초기화 (모델이 없으면 랜덤 가중치로 생성됨)
    print("\nInitializing Neural Bach Engine...")
    engine = NeuralBachEngine()
    
    # 3. 성능 측정
    try:
        avg_lat = benchmark_inference_latency(engine)
        
        print("\n" + "="*50)
        print("Final Performance Report (Current Environment)")
        print("-" * 50)
        print(f"{'Metric':<25} | {'Result'}")
        print("-" * 50)
        print(f"{'Device':<25} | {device.upper()}")
        print(f"{'Avg Inference (4-bar)':<25} | {avg_lat*1000:.2f}ms")
        print(f"{'Model Parameters':<25} | 25M (approx)")
        print(f"{'Status':<25} | Operational")
        print("="*50)
        
    except Exception as e:
        print(f"\n[Error during benchmark] {str(e)}")
        import traceback
        traceback.print_exc()
