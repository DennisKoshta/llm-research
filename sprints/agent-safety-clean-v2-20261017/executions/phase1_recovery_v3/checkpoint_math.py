"""Static projection and deterministic E4M3 block quantization; no runtime hooks."""
def project_weight(weight, direction, alpha=1.0):
    import torch
    if weight.dtype!=torch.bfloat16:
        raise ValueError('Projection input must be BF16')
    if weight.ndim!=2 or direction.numel()!=weight.shape[0]:
        raise ValueError('Residual-writing dimension mismatch')
    if alpha==0 or torch.count_nonzero(direction).item()==0:
        output=weight.clone()
        assert torch.equal(output.view(torch.uint8),weight.view(torch.uint8))
        return output
    r=direction.to(device=weight.device,dtype=torch.float32)
    if not torch.isclose(torch.linalg.vector_norm(r),torch.tensor(1.,device=r.device),atol=1e-5):
        raise ValueError('Projection direction must be unit length')
    # Accumulate the projection in FP32, then round the modified weights to BF16.
    w=weight.float()
    return (w-alpha*r[:,None]*(r@w)[None,:]).to(torch.bfloat16)


def quantize_e4m3(weight, block=128):
    import torch
    if weight.ndim!=2 or any(d%block for d in weight.shape):
        raise ValueError('Only full 128-by-128 blocks are supported')
    rows,cols=weight.shape
    tiles=weight.float().reshape(rows//block,block,cols//block,block).permute(0,2,1,3)
    scale=tiles.abs().amax(dim=(-2,-1))/448.0
    scale=scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized=(tiles/scale[:,:,None,None]).clamp(-448,448).to(torch.float8_e4m3fn)
    return quantized.permute(0,2,1,3).reshape(rows,cols).contiguous(),scale.contiguous()
