# DeCAN: Densely Connected Attention Networks
Taking inspiration from Densely Connected Convolution Networks, the Densely Connected Attention network (DeCAN) architecture introduces skip connections to the multi-head attention layers to encourage multi-level feature re-use and elicit the learning of high-rank key-value representations.

DeCAN comes in two flavours, the baseline implementation (or just 'DeCAN') which is 30% faster than a vanilla transformer while having slightly degraded evaluation performance, and the full implementation (or 'DeCAN+') which is 15% faster than a vanilla transformer but retains full evaluation performance! Both DeCAN and DeCAN+ have significantly smaller KV caches, with a GPT2-Small sized model using 12x less memory! Below is a table outlining the advantages of DeCAN and DeCAN+ over a vanilla transformer.

| Transformer Type | **Vanilla** | **DeCAN** | **DeCAN+** |
|------------------|:-----------:|:---------:|:----------:|
| KV Cache Size    |    Large    |   Small   |    Small   |
| Training Speed   |     Slow    |  Fastest  |    Fast    |
| Model Quality    |     Best    |  Degraded |    Best    |
