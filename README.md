# DeCAN: Densely Connected Attention Networks
Taking inspiration from Densely Connected Convolution Networks, the Densely Connected Attention network (DeCAN) architecture introduces skip connections to the multi-head attention layers to encourage multi-level feature re-use and elicit the learning of high-rank key-value representations.

DeCAN comes in two flavours, the baseline implementation (or just 'DeCAN') which is 30% faster than a vanilla transformer while having slightly degraded evaluation performance, and the full implementation (or 'DeCAN+') which is 15% faster than a vanilla transformer but retains full evaluation performance! Both DeCAN and DeCAN+ have significantly smaller KV caches, with a GPT2-Small sized model using 12x less memory!

<style>
	.coloured-table {}

	.coloured-table th { width: 25% }

	.coloured-table tr:nth-child(1) td:nth-child(2) { background: rgb( 255,  64, 64, .25 ); }
	.coloured-table tr:nth-child(1) td:nth-child(3) { background: rgb(  64, 255, 64, .25 ); }
	.coloured-table tr:nth-child(1) td:nth-child(4) { background: rgb(  64, 255, 64, .25 ); }

	.coloured-table tr:nth-child(2) td:nth-child(2) { background: rgb( 255,  64, 64, .25 ); }
	.coloured-table tr:nth-child(2) td:nth-child(3) { background: rgb(  64, 255, 64, .25 ); }
	.coloured-table tr:nth-child(2) td:nth-child(4) { background: rgb( 255, 255, 64, .25 ); }

	.coloured-table tr:nth-child(3) td:nth-child(2) { background: rgb(  64, 255, 64, .25 ); }
	.coloured-table tr:nth-child(3) td:nth-child(3) { background: rgb( 255, 255, 64, .25 ); }
	.coloured-table tr:nth-child(3) td:nth-child(4) { background: rgb(  64, 255, 64, .25 ); }
</style>

<div class="coloured-table">

| Transformer Type | **Vanilla** | **DeCAN** | **DeCAN+** |
|------------------|:-----------:|:---------:|:----------:|
| KV Cache Size    |    Large    |   Small   |    Small   |
| Training Speed   |     Slow    |  Fastest  |    Fast    |
| Model Quality    |     Best    |  Degraded |    Best    |

</div>