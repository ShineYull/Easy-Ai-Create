# Easy-Ai-Create
易用型的ai创作项目，根据ai专栏衍生的项目。

# python环境
使用的是python3.10。

# 常见问题
1、在mac上启动项目时遇到stablediffusion不可用，或者是报错的问题：  
答：需要使用pip将xformers卸载，并将audiocraft中使用到xformers模块的地方注视掉，或者直接将audiocraft给注视掉就行了。  

分析：这是因为mac上用不了cuda，所以需要使用cpu去跑stablediffusion，但是audiocraft又需要xformers模块，所以就报错了。  

报错的话，应该会像如下这样：  
```shell
NotImplementedError: No operator found for `memory_efficient_attention_forward` with inputs:
     query       : shape=(10, 4096, 1, 64) (torch.float32)
     key         : shape=(10, 4096, 1, 64) (torch.float32)
     value       : shape=(10, 4096, 1, 64) (torch.float32)
     attn_bias   : <class 'NoneType'>
     p           : 0.0
`flshattF` is not supported because:
    device=cpu (supported: {'cuda'})
    dtype=torch.float32 (supported: {torch.bfloat16, torch.float16})
    Operator wasn't built - see `python -m xformers.info` for more info
`tritonflashattF` is not supported because:
    device=cpu (supported: {'cuda'})
    dtype=torch.float32 (supported: {torch.bfloat16, torch.float16})
    Operator wasn't built - see `python -m xformers.info` for more info
    triton is not available
`cutlassF` is not supported because:
    device=cpu (supported: {'cuda'})
`smallkF` is not supported because:
    max(query.shape[-1] != value.shape[-1]) > 32
    unsupported embed per head: 64
```

2、报错：No matching distribution found for tb-nightly
```shell
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```

# 鸣谢以下开源项目
* [Sadtalker](https://github.com/OpenTalker/SadTalker)  
* [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)  
* [Audiocraft](https://github.com/facebookresearch/audiocraft)  
* [StableDiffusion](https://github.com/Stability-AI/stablediffusion)