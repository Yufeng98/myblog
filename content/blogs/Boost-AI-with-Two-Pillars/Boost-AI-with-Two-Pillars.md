---
author: "Yufeng Gu"
title: "Boost AI with Two Pillars: Efficient Model and Faster Memory"
date: 2025-08-03T08:37:58+08:00
ShowToc: true
TocOpen: false
---



Transformer-based LLMs have revolutionized AI, but their impressive capabilities are extremly resource-intensive during deployment, coming with two key bottlenecks. First, LLMs have large paramerter sizes (10-1000B) and supports long context lengths (128K-1M) today, which necessitates substantial compute and memory resources. Second, the autoregressive decoding process generates output tokens one after another, requiring extensive memory bandwidth.

<!-- 
A key challenge is the *autoregressive decoding* process, where tokens are generated one by one, which cannot be parallelized across output tokens. This sequential generation means that unlike the initial *prefill* phase, which encodes the input and can use large batched matrix multiplies, the decode phase processes tokens one by one, resulting in matrix-vector multiplies (GEMV) that keep re-reading model weights. In practice, each new token may require the model to read **gigabytes (even terabytes) of parameters from memory**, creating a heavy memory-bandwidth burden. As a result, LLM inference is often *memory-bound*: model weights on the order of 10–1000 GB must be fetched repeatedly from GPU memory (or beyond) for each token. Even attempts to batch multiple user queries together (turning GEMV into larger GEMM operations) quickly hit he memory capacity limits, especially for long context scenarios. In short, today’s large transformers are bottlenecked by memory throughput during decoding – they generate amazing results, but *slowly*. -->

These challenges have prompted a wave of innovation from both AI researchers and computer architects. Numerous efforts are underway to boost LLM inference at two “pillars” supporting the next generation of AI: **(1) Efficient model optimizations**, which include techniques like quantization and sparsity to compress models or reduce computation, novel attention mechanisms and fast decoding methods; and **(2) Memory architecture optimizations**, which improve how data is stored, moved, and processed, from faster HBM to processing-in/near-memory (PIM/PNM) approaches. Considering both “pillars”, **Hardware/Software co-designs** are also crutial for LLM deployment, where algorithm developers and hardware designers collaborate to tailor solutions that bridge the gap between model and hardware, such as specialized attention kernels and inference frameworks. 

## Efficient Model Optimization Techniques

Modern AI models can be optimized to run faster and use less memory *without* fundamentally changing their outputs. These model-side optimizations are critical for deploying LLMs at both datacenter and edge scenarios. They include making the model weights (and activations) more compact via **quantization**, eliminating redundant computations via **sparsity and pruning**, using smarter **attention patterns** for long sequences, and speculative **decoding strategy**. Many of these techniques can dramatically cut down memory usage and/or computation.

### Quantization

Quantization reduces the precision of the numbers used to represent neural network parameters (and sometimes activations), for example using 8-bit or 4-bit integers instead of 16-bit or 32-bit floats. By quantizing model weights, LLMs can fit into the device with limited memory capacity. Meanwhile, quantization can significantly shrink its memory footprint and even speed up computation, since more of the model can fit in fast on-chip memory and integer math can be faster on some hardware. 

<!-- Recent advances show it’s possible to compress even 175-billion-parameter models down to 3–4 bits per weight with minimal loss in accuracy – a 2–4× speedup in GPT-class model inference has been reported when using high-end GPUs with quantized weights. -->

#### What to quantize? 

**weight-only quantization** quantizes the model’s learned parameters but still computes with high-precision activations. These approaches directly reduces model size and memory load. Other approaches also quantize activations (and even the *KV cache* in LLM decoders), achieving further gains by cutting memory and arithmetic for intermediate values at the cost of potential accuracy degragation.

<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/What-to-quantize.png" alt="Alt text" width="800">
  <figcaption style="color: gray; text-align: center;">Figure 1: Weight-only and Weight-Activation Quantization. Figure source: https://arxiv.org/abs/2410.04466</figcaption>
</figure>


#### When to quantize? 

There are two approaches: **quantization-aware training (QAT)**, where the model is trained (or fine-tuned) with quantization in mind, versus **post-training quantization (PTQ)**, where we take a pre-trained model and quantize it in one go. QAT tends to produce the best results, since the model can adjust to low precision during training. PTQ is cheaper due to the one-time quantization nature.

#### How to quantize?

Quantization can be catergorized as *Uniform* and *Non-uniform* approaches. Uniform quantization divides the entire range of values into equally sized intervals, which is simple but can lead to information loss when values are unevenly distributed. In contrast, non-uniform method divides the entire range of value into varying-sized intervals based on the value distribution. The uniform quantization further includes Symmetric and Asymmetric variants, with figure and formula shown as follows. 

<figure>
  <div style="display: flex; justify-content: center; gap: 10px;">
    <img src="../../Boost-AI-with-Two-Pillars/images/Asymatric-quantization.png" alt="Asymatric quantization" width="50%">
    <img src="../../Boost-AI-with-Two-Pillars/images/Symmetric-quantization.png" alt="Symatric quantization" width="50%">
  </div>
  <figcaption style="color: gray; text-align: center;">Figure 2: Asymatric and Symatric Quantization. Figure source: https://huggingface.co/blog/Isayoften/optimization-rush</figcaption>
</figure>

The quantization range is determined by the maximum absolute value of the data. The quantization process involves the scaling (S) and shifting (Z) stages. The scaling factor is determined by the range of both source and target ranges. Symmetric quantization skips shifting. The de-quantization process is implemented by reverse shifting and scaling.

**Asymmetric**

- $ S = \frac{r_{\text{max}} - r_{\text{min}}}{q_{\text{max}} - q_{\text{min}}} $
- $ Z = \left[ q_{\text{min}} - \frac{r_{\text{min}}}{S} \right] $
- $ X_{\text{quantized}} = \left[ \frac{X}{S} + Z \right] $
- $ X_{\text{dequantized}} = S \left( X_{\text{quantized}} - Z \right) $

---

**Symmetric**

- $ S = \frac{|r|_{\text{max}}}{2^{N-1} - 1} $
- $ Z = 0 $
- $ X_{\text{quantized}} = \left[ \frac{X}{S} \right] $
- $ X_{\text{dequantized}} = S X_{\text{quantized}} $


Non-uniform quantization is more flexible, borrowing the idea of floating point number. Dynamic tree quantization (DTQ) is a non-linear 8-bit quantization scheme designed to keep errors low for both very small and very large magnitudes. Instead of a fixed split between “exponent” and “fraction” bits, DTQ allows for adjustable split: (1) The first bit of the data type is reserved for a sign. (2) The number of subsequent zero bits indicates the magnitude of the exponent. (3) The first bit that is set to one indicates that all following values are reserved for (4) linear quantization.

<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/Dynamic-tree-quantization.png" alt="Alt text" width="400" style="margin: auto;">
  <figcaption style="color: gray; text-align: center;">Figure 3: Dynamic Tree Quantization. Figure source: https://ar5iv.labs.arxiv.org/html/2110.02861</figcaption>
</figure>

Outliers affect quantization accuracy. Tensors may have 0.01-0.1% of values with very large absolute values. Calculating the scaling factor with these outliers reduces the precision of the remaining values with small absolute. Below we introduces a few advanced approaches to address the outlier issue. 


**LLM.int8** keeps “outlier” activation channels and the corresponding weights in 16-bit to enable 8-bit inference with negligible loss. 

<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/LLM-int8.png" alt="Alt text" width="800">
  <figcaption style="color: gray; text-align: center;">Figure 4: LLM.int8 Quantization. Figure source: https://arxiv.org/abs/2208.07339</figcaption>
</figure>


**QLoRA** is a 4-bit quantization technique combined with low-rank adaptation (LoRA) for fine-tuning. During parameter efficient fine-tuning (PEFT), the forward pass goes through both the pretrained weights and a low-rank adaptor, while the backward pass is only applied on the low-rank adaptor. The trained adaptor is updated to the pretrained weights, therefore significantly reducing the trainable parameters. In QLoRA, the base model is quantized into 4-bit and loses accuracy, but the 16-bit LoRA fine-tuning process can compensate for the accuracy degradation during quantization.


<figure>
  <div style="display: flex; justify-content: center; gap: 10px;">
    <img src="../../Boost-AI-with-Two-Pillars/images/LoRA.png" alt="Asymatric quantization" width="30%">
    <img src="../../Boost-AI-with-Two-Pillars/images/QLoRA.png" alt="Symatric quantization" width="70%">
  </div>
  <figcaption style="color: gray; text-align: center;">Figure 5: LoRA and QLoRA. Figure source:  https://https://arxiv.org/abs/2106.09685 and https://arxiv.org/abs/2305.14314</figcaption>
</figure>


Other notable methods include **SmoothQuant** (which smooths out activation magnitude differences between layers to improve 8-bit activation quantization), 

**SmoothQuant** is an 8-bit weight, 8-bit activation (W8A8) post training quantization (PTQ) for LLMs. Based on the fact that weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers by offline migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation, as shown in the formula and figures as follows.

$$
\mathbf{Y} = \left( \mathbf{X} \, \mathrm{diag}(\mathbf{s})^{-1} \right) \cdot \left( \mathrm{diag}(\mathbf{s}) \, \mathbf{W} \right) = \hat{\mathbf{X}} \, \hat{\mathbf{W}}
$$


<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/SmoothQuant.png" alt="Alt text" width="400" style="margin: auto;">
  <figcaption style="color: gray; text-align: center;">Figure 6: SmoothQuant Quantization. Figure source: https://arxiv.org/abs/2211.10438 </figcaption>
</figure>

**AWQ (Activation-Aware Weight Quantization)** identifies a small fraction (\~1%) of “salient” weights that have outsized impact on activations and keeps those in higher precision (e.g. FP16), quantizing the rest. But this exerts challenges for mixed-precision execution on hardware. Instead of keeping in higher precision, AWQ’s hardware-friendly design multiplies these salient weights with a scaling factor (s>1) before quantization, reducing the accuracy degradation on which, as shown in the formula and figures as follows. 

$$
Q(\mathbf{w}) = \Delta \cdot \mathrm{Round}\left( \frac{\mathbf{w}}{\Delta} \right), 
\quad \Delta = \frac{\max(|\mathbf{w}|)}{2^{N-1}}
$$


where $N$ is the number of quantization bits, and $\Delta$ is the quantization scaler determined by the absolute maximum value. Now consider a weight element $w \in \mathbf{w}$, if we multiply $w$ with $s > 1$ and inversely scale $x$, we will have $Q(w \cdot s)(x/s)$, which is:  

$$
Q(w \cdot s) \cdot \frac{x}{s} 
= \Delta' \cdot \mathrm{Round}\left( \frac{w s}{\Delta'} \right) \cdot x \cdot \frac{1}{s}
$$

where $\Delta'$ is the new quantization scaler after applying $s$.


<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/AWQ.png" alt="Alt text" width="800">
  <figcaption style="color: gray; text-align: center;">Figure 7: Activation-aware Weight Quantization. Figure source: https://arxiv.org/abs/2306.00978</figcaption>
</figure>


**SpinQuant** introduces the idea of learning rotation matrices to transform weight space before quantization, narrowing the accuracy gap even at 4-bit weight+activation settings. Rotating activation or weight matrices helps remove outliers and benefits quantization. Paired rotation matrices can be merged into corresponding weight matrices, such as $R_1$ and $R_1^{-1}$. After absorption, no new parameters are introduced in the network without impacting the network's quality. The Hadamard matrix, such as $R_3$, can be inserted as the unabsorbed rotation matrix when low-bit KV cache quantization is required.


<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/SpinQuant.png" alt="Alt text" width="800">
  <figcaption style="color: gray; text-align: center;">Figure 8: SpinQuant Quantization. Figure source: https://arxiv.org/abs/2405.16406</figcaption>
</figure>

### Sparsity and Pruning


[Source](https://arxiv.org/abs/2006.05525): Knowledge Distillation: A Survey
[Source](https://arxiv.org/abs/2104.08378): Accelerating Sparse Deep Neural Networks


Another major avenue is making the model *sparse*: eliminating or skipping redundant computations. Deep neural networks have more parameters than needed, and many weights can be zeroed out (pruned) without much loss in accuracy. If done right, this means we don’t waste time multiplying by zeros. 

#### Structured sparsity

Structured sparsity in particular targets a regular pattern of zeros that hardware can exploit. A prime example is NVIDIA’s **2:4 structured sparsity**: in each group of 4 weight values, 2 are forced to zero, yielding a 50% sparse weight matrix that hardware can compress and accelerate. This tehnique effectively doubles tensor core throughput by skipping the zero multiplies.


<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/Structured-sparsity.png" alt="Alt text" width="600" style="margin:auto;">
  <figcaption style="color: gray; text-align: center;">Figure 9: 2:4 Structured Sparsity. Figure source: https://arxiv.org/abs/1811.03115</figcaption>
</figure>


<figure>
  <img src="../../Boost-AI-with-Two-Pillars/images/Structured-sparsity-compute.png" alt="Alt text" width="800">
  <figcaption style="color: gray; text-align: center;">Figure 10: Sparse Matrix-Multiply (SpMM) on NVIDIA's Tensor. Figure source: https://arxiv.org/abs/1811.03115</figcaption>
</figure>

#### Sparse Attention

Beyond weights, we can also introduce sparsity in the **attention computation** – a crucial strategy for long sequences. Full self-attention scales quadratically with sequence length, which is infeasible beyond a few thousands of tokens. **Sparse attention mechanisms** restrict each token to attending to only a subset of other tokens (e.g. a local window or random set), achieving linear or *O*(n·log n) complexity while often maintaining similar accuracy to full attention. Models like **Longformer** and **BigBird** pioneered this idea for long texts, using combinations of local windows and global summary tokens so that each token doesn’t attend to everything. These sparse-attention transformers can handle input sequences 10× longer with comparable accuracy to dense attention. In fact, Microsoft’s DeepSpeed library implemented a **Sparse Attention** suite that supports flexible block-sparse patterns and showed up to *6× faster* attention and 10× longer context lengths using these techniques. The DeepSpeed kernels allow mixing local, global, and random attention patterns efficiently, alleviating memory bottlenecks for long context lengths.

*Block-sparse attention* is another flavor: instead of individual query-key pairs being dropped, blocks of the attention matrix are pruned. A recent example is **XAttention (MIT Han Lab, 2025)**, which finds which blocks of the QK attention matrix can be skipped by using an **antidiagonal sum** as a cheap importance metric. By pruning away unimportant blocks, XAttention achieved up to **13.5× speedups** in attention computation on long contexts, without hurting accuracy on language and even video understanding tasks. Such block-sparse approaches are appealing because they map well to hardware and avoid the overhead of irregular sparsity.

Another type of structured sparsity is the *N\:M fine-grained sparsity* (like 2:4) mentioned above, which is also being explored for attention weights and activations in research. For instance, a dynamic *2:4 sparse attention* mechanism was shown to yield \~1.4–1.8× speedups over dense attention on A100 GPUs with minimal quality drop.

In summary, inducing sparsity – whether in the weight matrices or in the attention patterns – can drastically reduce the compute and memory needs. Industry hardware (NVIDIA Ampere/Hopper, etc.) already supports certain sparse patterns, making this a practical path to faster inference. We expect continued improvements in algorithms for selecting which weights or attention entries to prune (possibly guided by learning, as in some recent lottery ticket and N\:M mask learning studies), pushing the limits of sparsity without sacrificing model accuracy.

### **Speculative Decoding**

Beyond pruning or compressing existing models, researchers are also inventing new inference-time techniques and architectures that sidestep the fundamental latency issues of autoregressive decoding. Two notable directions are using *draft models for faster decoding* and restructuring transformer layers or heads to share work.

**Speculative decoding** is a recent technique that *trades extra compute for much less latency*. The idea is to run a small “draft” model to generate several tokens ahead, and then have the large model quickly verify or correct that draft in one go. Because the big model no longer generates every token sequentially by itself, several tokens can be obtained in parallel. Google first demonstrated speculative decoding in 2022, showing **2–3× faster generation** with no quality loss, since the large model’s output distribution is provably unchanged by this process. Essentially, it guarantees the same results as standard decoding, just faster – a rare free lunch. This method has since been adopted in production at Google (for example, speeding up Bard and Search AI results) and elsewhere. IBM researchers similarly reported that speculative decoding can significantly lower inference cost by using a cheaper model to accelerate a more expensive one. In practical terms, a user might perceive much snappier responses (latency cut in half or better) because the system is leveraging two models in parallel to generate the text. As LLMs grow even larger, such two-model setups that *facilitate parallel token generation* will be an important strategy to meet real-time application needs.

### Efficient Attention

#### Grouped-query Attention

#### Cross-layer Attention

Another line of research modifies the *transformer architecture* itself for efficiency. For example, some works investigate **cross-layer attention**, where instead of each layer attending only within itself, lower layers might directly attend to the outputs of higher layers or vice versa – potentially reducing redundant computations across layers. 

#### Multi-head Latent Attention

Similarly, **multi-head latent attention** aims to factorize or reduce the cost of having many attention heads, by finding a smaller latent representation that still captures the diversity of heads. (These latter ideas are still experimental, and specific techniques are under active study, so while we won’t delve into detailed citations here, they represent the forward-looking attempts to redesign transformers for efficiency.)

Lastly, a simple but powerful idea in inference serving is to optimize *how we batch and schedule decode steps*. Techniques like **continuous batching** and efficient scheduling (as implemented in libraries like HuggingFace’s text-generation-inference or DeepSpeed’s inference engine) ensure that whenever multiple requests are waiting for a next token, they are processed together on the GPU. This amortizes overhead and keeps utilization high. The **vLLM** serving system introduced PagedAttention (discussed more below) and also a scheduling algorithm to achieve near-perfect batching of concurrent decode steps, improving throughput substantially. These serve as software-level optimizations that complement the model tweaks described above.

In summary, model-centric optimizations – from quantization and sparsity to improved attention mechanisms and decoding algorithms – have proven essential for *taming* large transformers. For instance, combining 4-bit quantization with sparse attention and speculative decoding might enable an order of magnitude speedup in end-to-end generation. However, to unlock the next level of performance, we also must turn to the hardware side – ensuring the memory and compute infrastructure can supply data to these models as quickly as needed.

## **Memory Architecture Optimization**

Since LLM inference is largely a memory-bound problem, a parallel push is happening in improving memory systems for AI. These efforts range from using **faster memory technologies** (and more of it), to **unifying memory between devices** to avoid data copies, and even bringing compute *into* the memory chips themselves. The goal is to break the bottleneck imposed by current memory bandwidth limits, so that massive models can be served with lower latency and without requiring unrealistically large (and expensive) hardware clusters.

### **High-Bandwidth Memory and Unified Memory Solutions**

Today’s high-end AI accelerators already use **High Bandwidth Memory (HBM)** – specialized memory co-packaged with the GPU that offers huge bandwidth. For example, NVIDIA’s H100 GPU uses HBM3 memory delivering **up to 3 TB/s** of bandwidth, roughly double that of the previous generation A100. This is achieved by stacking memory dies with wide interfaces. Increasing memory bandwidth directly improves LLM decoding speeds, as more tokens can be generated per second before hitting the IO wall. Along with bandwidth, capacity per GPU is also growing (80GB or more HBM on high-end cards), which allows batching more requests or hosting larger models locally. In addition, GPU vendors provide features like memory compression and larger on-chip caches (e.g. 50MB L2 on H100) to reduce how often the GPU must reach out to HBM.

Still, a single GPU’s memory is finite. **Unified memory architectures** seek to make *other* memory available to accelerators in a seamless way. One approach is NVIDIA’s **unified virtual memory** which lets GPU code use CPU memory automatically, but the data transfer over PCIe is slow. A more recent development is the adoption of **CXL (Compute Express Link)** in data centers. CXL is an interconnect standard that enables coherent, high-bandwidth memory pooling between CPUs, GPUs, and memory expanders. Startups like **Panmnesia** have shown that using CXL-attached memory, one can effectively “add” tens or hundreds of GB of memory to a GPU’s pool, *unifying* the GPU’s HBM with an external memory device in the same address space. In their prototype, the GPU could access this external memory with sub-100 nanosecond latency – much faster than typical PCIe – enabling large models to run without being limited by on-board HBM size. This disaggregated memory approach means we don’t need as many GPUs just to meet memory requirements; instead, a few GPUs can be augmented with CXL memory expansion to handle model sizes that previously required model sharding across multiple GPUs. Companies like AMD are also building APUs (CPU+GPU on one package with shared memory) and leveraging CXL. The AMD Instinct MI300A, for instance, combines CPU and GPU with a unified memory, and connects to additional memory over CXL for more capacity. All these trends point to a future where **the memory hierarchy for AI is more flexible**: ultra-fast HBM for active working sets, and large pools of slower (but still fast) memory accessible via CXL for overflow, all seen by the programmer as one continuous memory space. This reduces the need to offload model weights to disk or manually manage data movement – the hardware and driver can page data in/out as needed (hence “paged” attention ideas at the software level align well with this concept).

### **Processing-In-Memory (PIM) and Near-Memory Computing**

Perhaps the most radical improvement to memory bandwidth is to **move the computation closer to where data lives**, so that instead of shuttling enormous weight matrices back and forth, we perform some operations directly in memory. This concept of processing-in-memory has made a comeback, driven by the AI workload. There are both commercial and academic developments here:

* On the commercial side, **UPMEM** introduced the first general-purpose PIM DIMMs. An UPMEM DDR4 module includes *hundreds of tiny RISC cores (DPUs) embedded inside the DRAM chips*, each core having access to a portion of the DRAM bank. These in-memory processors can run code near the data, achieving order-of-magnitude speedups on certain data-intensive tasks (like scanning or filtering large tables) while being 10× more energy-efficient than a CPU for those tasks. Although UPMEM’s DPUs are not specifically designed for matrix math, they prove the feasibility of integrating logic into commodity memory. Moving to AI-specific PIM: in 2021 Samsung announced **HBM-PIM (code-named Aquabolt-XL)**, which integrates AI processing engines into each 3D-stacked HBM memory bank. Their PIM architecture, sometimes called **FIMDRAM**, includes a simple SIMD MAC unit in each bank that can perform parallel multiply-accumulate operations on data *inside the memory*. This effectively turns memory into a combined storage+compute unit. For example, Samsung reported their HBM-PIM could achieve 1.2 TFLOPs of AI computation capability while reducing energy use by a large margin. Such PIM-enabled HBM can act as a smart cache for GPUs, offloading parts of neural network layers. Notably, these chips can operate in normal memory mode or PIM mode, so they are backward-compatible.

* In academic research, we see a proliferation of PIM-based accelerator proposals tailored to transformers. **AttAcc** (ASPLOS 2024) is one such design that *focuses on the attention mechanism*. It uses an HBM-based PIM architecture to perform the core attention computations (like Q·K and attention-weighted sum) directly in memory, thereby alleviating the data movement between GPU and memory for the attention layers. By doing so, AttAcc significantly speeds up batched attention and was shown to outperform a baseline GPU setup on transformer inference. Another project, **NeuPIM**, combined near-memory compute with neural network ASICs: essentially, each HBM stack got a built-in *TPU-like neural processor* next to the DRAM banks. This heterogeneous design keeps most matrix multiplications near the memory and only relies on a host CPU or minimal GPU for other tasks. NeuPIM demonstrated large gains in energy efficiency and throughput for LLM inference by cutting down those costly weight transfers. Extending the idea further, the recent **CENT** architecture proposes a *CXL-connected PIM cluster*: 32 memory devices, each with PIM compute, are linked via a CXL 3.0 switch to work together on LLM inference. The host CPU orchestrates this network of smart memory nodes. By using CXL’s fabric, CENT can scale out the PIM approach without a GPU, effectively acting as a GPU-free accelerator for large models. The **H2LLM** project (short for Hybrid Hierarchy for LLM, 2023) similarly explores using a two-level hierarchy of memory with compute – though details aside, the clear trend is thinking of *systems where memory modules do part of the “heavy lifting”* of neural network computation, especially for the memory-bound parts like attention or feed-forward layers with large weight matrices.

It’s worth noting that PIM is still emerging tech – challenges like integrating these into existing systems, programming them, and handling their limitations (e.g. simpler compute units, lower precision) are active research. But the promise is compelling: imagine a future GPU with not just 3 TB/s HBM, but with each HBM chip able to preprocess or multiply data on its own, delivering effectively >10× the usable bandwidth. In fact, one study showed a PIM-based LLM accelerator could be **4–6× faster** and more energy-efficient than even an NVIDIA A100 GPU for end-to-end inference. Real products may not be there yet, but companies like Samsung, SK Hynix, and start-ups are rapidly advancing PIM prototypes, so we may see such memory-centric acceleration in the coming years.

## **Hardware/Software Co-Design Innovations**

Closing the gap between theoretical model efficiency and real-world deployment often requires co-designing algorithms *with* hardware capabilities in mind. In the context of LLMs, this has led to specialized kernel libraries and inference frameworks that dramatically improve utilization and memory use, without changing the model’s architecture. We’ll highlight a few prominent examples of these cross-cutting innovations: **FlashAttention**, **PagedAttention**, and **disaggregated inference approaches**. Each of these bridges the divide between model and hardware in clever ways – by reordering operations, by treating memory like an OS would treat virtual memory, or by splitting workloads across different hardware resources – yielding substantial performance boosts.

### **FlashAttention: Exact Attention Faster than Ever**

A big breakthrough in 2022 was the introduction of **FlashAttention**. This is an algorithmic improvement to how Transformers compute their attention scores. Normally, computing attention on sequences is both compute- and memory-intensive (quadratic memory usage with sequence length). FlashAttention doesn’t approximate the result (it’s still exact), but it *reorders computations and leverages the GPU memory hierarchy* for efficiency. By tiling the attention calculation, FlashAttention keeps intermediate results in high-speed on-chip SRAM and dramatically reduces expensive reads/writes to GPU HBM memory. In terms of complexity, it brings the memory usage of attention down from *O*(n²) to *O*(n) with respect to sequence length. Practically, Tri Dao *et al.* showed this yields about a **2–3× speedup** on long sequences and allows much longer sequences to be processed on a given GPU without running out of memory. For example, using FlashAttention, they set new records in training long-context transformers (up to 16k or 64k tokens) and even unlocked new abilities (solving tasks that require 64k token context, which was infeasible before). From an engineering perspective, FlashAttention is a stellar example of co-design: it considers IO-awareness and optimal usage of on-chip memory explicitly. By doing more compute in shared memory and avoiding materializing large attention matrices, it not only speeds up training but also makes inference of LLMs more efficient (less memory per step, which can translate to higher batch sizes or longer prompts). Many libraries (e.g. PyTorch with xFormers, Hugging Face, and JAX) have integrated FlashAttention or similar kernels. In short, *the same GPU can handle longer sequences faster*, just thanks to a smarter algorithm that was designed with hardware constraints (memory bandwidth) in mind. This is co-design in action: no change to the model’s outputs, but a big change in how the model’s math is scheduled on hardware.

### **PagedAttention: Managing KV Cache like Virtual Memory**

Another ingenious development was motivated by the observation that during LLM decoding, a *lot* of memory is tied up storing the **key/value cache** – i.e. the hidden states from prior tokens that are needed for attention at each new generation step. For long outputs or many concurrent chats, this KV cache becomes huge, and existing systems often managed it in a naive way (one big contiguous array per request, which can’t be easily resized or shared). **PagedAttention** is a technique (introduced with the **vLLM** serving system) that treats the KV cache in an OS-like fashion. It **breaks the cache into fixed-size blocks (pages)** and uses a lookup indirection, so the blocks can be flexibly assigned, reused, or even shared between requests. This yields *near-zero fragmentation* and allows the memory from finished requests to be immediately recycled for new ones. In essence, PagedAttention eliminates the waste where maybe 40% of allocated KV memory was unused due to fragmentation or over-provisioning. The result is that **vLLM can achieve much higher batch sizes and throughput – 2–4× improvement** – at the same latency, compared to traditional systems. It decouples the memory for KV from the compute, meaning one GPU can serve many requests in parallel without running out of memory, as long as the average utilization is within limits. This idea, inspired by **virtual memory paging**, is a great example of software-hardware co-design: it doesn’t require new hardware, but it borrows a concept from computer architecture (paging and fragmentation management) to vastly improve hardware utilization. Notably, PagedAttention also makes it trivial to *share* parts of the KV cache across requests – for example, if multiple users prompt the same context, their initial keys/values can be computed once and reused – something that was very hard to do without a paging mechanism. Overall, PagedAttention addresses a practical memory bottleneck in LLM serving and has been adopted in efficient inference engines to allow larger context lengths and higher throughput on existing GPUs.

### **Disaggregated Prefill/Decode: Specialized Inference Pipelines**

When we look at an LLM service, as discussed, there are two distinct phases: the *prefill* (process the prompt) and the *decode* (generate tokens). They have different characteristics – prefill is heavy compute but done once, decode is lighter per step but done many times sequentially – and different importance for latency (time-to-first-token vs time-per-token). A one-size-fits-all approach on a single type of GPU often leads to suboptimal usage: the GPU might be underutilized during decode or we over-provision to meet strict latency on one part and waste capacity on the other. **Disaggregated inference** means separating these phases onto *different resources* (e.g. different GPU pools or nodes) and optimizing each independently. Research systems like **DistServe** (2024) demonstrated that by assigning prefill to one set of GPUs and decode to another, they could eliminate interference between the two and tune batch sizes and parallelism for each phase separately. For instance, prefill might be done on a GPU with high parallelism to crunch through the prompt quickly, while decode could be spread across more GPUs or CPU offloaded if needed, focusing on throughput per token. DistServe showed this approach can **serve significantly more requests under the same latency Service-Level Objectives**, improving throughput by up to *4.4×* in some cases. Similarly, the vLLM framework offers an *experimental disaggregated mode* where one server (or GPU) handles all prefill computations and then hands off the KV cache to another server specialized for decoding. NVIDIA has also noted this strategy in their inference best-practices, suggesting that separating prompt handling and generation can improve overall utilization and reduce costs. Essentially, this is a co-design at the system level: acknowledging the differing compute/memory patterns of phases and optimizing the deployment accordingly. It’s somewhat analogous to how a pipeline in a CPU might have different units for different stages – here we have different “engines” for different parts of the inference workload.

### **Disaggregating Model Layers (Attention vs FFN)**

In large models, not only can phases be disaggregated, but even within a single transformer layer, different subcomponents have different resource profiles. The self-attention part is relatively light on compute but heavy on memory (since it deals with the sequence and KV cache), whereas the feed-forward network (FFN) is a big matrix multiply, heavy on compute. Traditionally these execute in sequence on the same device. Recent research has explored splitting them: for example, running all the attention sublayers on one set of hardware and the FFN sublayers on another. This was explored in the context of **Mixture-of-Experts (MoE)** models by a system called **MegaScale-Infer**. In MoEs, the FFN part is especially large (multiple expert networks), making the imbalance even greater. MegaScale-Infer **disaggregates attention and FFN modules within each transformer layer**, allowing each to be scaled and parallelized independently. They even deploy them heterogeneously – e.g. attention could run on GPUs optimized for bandwidth, while FFNs (the experts) run on GPUs or TPUs optimized for dense compute. They introduce a “ping-pong” scheduling so that micro-batches of data shuttle between attention and FFN engines, always keeping both busy. This design achieved up to **1.9× higher throughput per GPU** in their experiments, by significantly boosting utilization (no more waiting idle while the other sublayer computes). Essentially, disaggregating at this finer granularity is like a form of *specialized parallelism*: treat the transformer not as monolithic but as a workflow of different tasks. By assigning the right hardware to each task and overlapping them, one can reduce bottlenecks. While this approach is complex, it might become relevant as models and deployments scale – especially for huge MoE models where experts could even live on separate GPU nodes. It’s another example where system design and model structure intersect to yield performance gains beyond just “throw more GPUs at it”.

---

**In conclusion**, the path toward faster and more efficient AGI (artificial general intelligence) systems is being built on these “two towers” – advances in *model optimization* and *memory/hardware optimization* – with a bridge of co-design connecting them. On the model side, techniques like quantization and sparsity have slashed the size and compute needs of enormous networks, making it feasible to run models like GPT-3 on a single GPU with minimal loss in quality. New attention mechanisms and decoding strategies promise to overcome the fundamental serial nature of generation, evidenced by speculative decoding yielding 2–3× faster outputs with identical results. On the hardware side, we see memory bandwidth – the longtime bane of neural networks – being attacked from all angles: HBM3 and beyond for raw throughput, clever memory management (PagedAttention) to use every byte efficiently, and processing-in-memory to cut down data movement by orders of magnitude. It’s telling that recent works call LLM inference *“memory-bound”* and respond by essentially turning memory itself into the new compute unit.

The synergy of these efforts is already yielding impressive results. For example, by combining 4-bit quantization (GPTQ/AWQ) with FlashAttention and vLLM’s PagedAttention, one report improved inference throughput on a 30B model by over **4×** while even *increasing* the maximum sequence length. And hardware like the H100 GPU, with its Transformer Engine and sparsity support, can train and infer models far faster than previous generation cards – up to 30× faster inference on large language models compared to its predecessor, thanks to a combination of FP8 precision and structured sparsity support.

As we move beyond 2025, we can expect these optimizations to continue and new ones to emerge. **The boundary between model and hardware is blurring**: training techniques are developed with certain hardware in mind, and conversely, hardware is being architected for specific AI workloads. Achieving AGI-level models will require not just algorithmic breakthroughs but also the engineering to run those algorithms efficiently. The community is well aware of this, which is why we see this rich landscape of optimizations from low-level memory tweaks to high-level model redesigns.

In summary, the quest for AGI is not just about making models *smarter*, but also making them *leaner and faster*. By standing on the two pillars of model and memory optimization – and by fostering collaboration between the “brains” (AI models) and the “brawn” (hardware) – we are steadily pushing the limits of what these systems can do, bringing the future of AI a little closer to the present.

**References**

* Li *et al.*, [Large Language Model Inference Acceleration: A Comprehensive Hardware Perspective](https://arxiv.org/abs/2410.04466), arXiv Preprint.
* Daniil Suhoi, [Efficient Deep Learning: A Comprehensive Overview of Optimization Techniques](https://huggingface.co/blog/Isayoften/optimization-rush), blog.
* Dettmers *et al.*, [8-bit Optimizers via Block-wise Quantization](https://ar5iv.labs.arxiv.org/html/2110.02861), ICLR 2022.
* Dettmers *et al.*, [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/2208.07339), NeurIPS 2022.
* Hu *et al.*, [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICRL 2022.
* Dettmers *et al.*, [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), NeurIPS 2023.
* Lin *et al.*, [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) MLSys 2024
* Xiao *et al.*, [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438), ICLR 2025.
* Stern *et al.*, [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115), arXiv Preprint.


* Leviathan *et al.*, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), ICML 2023.
* Sadhukhan *et al.*, [MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding](https://arxiv.org/abs/2408.11049), ICRL 2025.

a target LLM speculates itself with a sparsified version of its own KV cache, then it can achieve
acceptance rates higher than those of small draft models with a full KV cache.
