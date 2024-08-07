import torch
import time

TOPS = 1e12
# 确保在GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def torch_gemm_benchmark(M, K, N, use_tensor_core=True):
    
    if use_tensor_core:
        print('benchmark torch_gemm with tensor core ====================>>')
    else:
        print('benchmark torch_gemm with cuda core ======================>>')

    # 设置矩阵大小
    M, K, N  = 4096, 4096, 4096
    warmup_iter = 10
    repeat_iter = 100


    # 生成随机矩阵
    if use_tensor_core:
        A = torch.randn(M, K, device=device, dtype=torch.float16)
        B = torch.randn(K, N, device=device, dtype=torch.float16)
    else:
        A = torch.randn(M, K, device=device)
        B = torch.randn(K, N, device=device)


    # 预热GPU，确保初次调用不会影响测试结果
    for i in range(warmup_iter):
        C = torch.matmul(A, B)

    torch.cuda.synchronize()  # 同步GPU操作
    # 测试开始时间
    start_time = time.time()

    # 执行矩阵乘法
    for i in range(repeat_iter):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()  # 同步GPU操作
    # 测试结束时间
    end_time = time.time()

    # 计算并打印执行时间
    execution_time = (end_time - start_time)/repeat_iter
    compute_flops = M * N * K * 2
    print(f'GEMM  execution time: {execution_time*1000} ms')
    print(f'real compute TOPS: {compute_flops/execution_time/TOPS} TOPS')

if __name__ == '__main__':
    torch_gemm_benchmark(4096, 4096, 4096, use_tensor_core=False)
    torch_gemm_benchmark(4096, 4096, 4096, use_tensor_core=True)

"""
>>> import torch
>>> torch.__version__
'2.1.0+cu121'

不用tensor core
GEMM  execution time: 0.007350144386291504 seconds
real compute TOPS: 18.69881001634914 TOPS

使用tensor core
GEMM  execution time: 0.0005797941684722901 seconds
real compute TOPS: 237.0478368800092 TOPS

CUDA_VISIBLE_DEVICES=2 python benchmark_cublas_use_torch.py
real compute TOPS: 266.0478368800092 TOPS
"""