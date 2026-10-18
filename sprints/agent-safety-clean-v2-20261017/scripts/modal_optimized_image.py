"""Build and import-check optimized kernels without allocating a GPU."""
import modal

app=modal.App('agent-safety-clean-v2-20261017-kernel-build')
image=(modal.Image.from_registry('vllm/vllm-openai:v0.31.0',add_python='3.12')
       .entrypoint([])
       .pip_install('transformers==5.20.1','accelerate==1.15.0','scipy==1.18.1')
       .pip_install('flash-linear-attention==0.5.2','ninja')
       .apt_install('g++')
       .env({'TORCH_CUDA_ARCH_LIST':'9.0','MAX_JOBS':'4','CC':'gcc','CXX':'g++','CUDAHOSTCXX':'g++'})
       .run_commands('pip install causal-conv1d==1.7.0 --no-build-isolation'))


@app.function(image=image,cpu=(2,2),memory=(4096,4096),timeout=120,
              retries=0,max_containers=1,scaledown_window=2)
def probe():
    import importlib.metadata
    import fla
    import causal_conv1d
    return {name:importlib.metadata.version(name) for name in ['torch','transformers','flash-linear-attention','causal-conv1d']}


@app.local_entrypoint()
def main():
    print(probe.remote())
