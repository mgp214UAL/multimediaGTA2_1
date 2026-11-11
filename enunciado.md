# Bit-Rate Control

**Autores:** [Vicente González Ruiz](https://vicente-gonzalez-ruiz.github.io/) y [Savíns Puertas Martín](https://hpca.ual.es/)  
**Fecha:** 14 de octubre de 2026

## Contents

1. [Description](#1-description)
   - [Impact of the network throughput](#11-impact-of-the-network-throughput)
   - [Compressing the audio data with zlib](#12-compressing-the-audio-data-with-zlib)
   - [Quantization](#13-quantization)
   - [Relationship between the QSS (Δ) and the quantization error](#14-relationship-between-the-qss-δ-and-the-quantization-error)
   - [(Bit-)Rate control and distortion](#15-bit-rate-control-and-distortion)
   - [The current implementations for the control of the bit-rate](#16-the-current-implementations-for-the-control-of-the-bit-rate)
2. [Deliverables](#2-deliverables)
3. [Resources](#3-resources)

---

## 1. Description

### 1.1 Impact of the network throughput

Along with latency and its variation (jitter), another important aspect of the transmission link used in an InterCom session is its **link throughput** (or bit-rate).

The bit-rate depends on:

- The maximum **capacity** of the link, which is closely related to its available bandwidth.
- The **congestion level**, which depends primarily on the network load.

In general, capacity can be considered constant over time: the bandwidth provided by the link does not vary. Throughput, however, changes over time and is difficult to predict because it depends on congestion, which in turn depends on the behavior of network users.

In this milestone, the goal is to measure the impact of link throughput on the Quality of Experience (QoE) provided by the current InterCom implementation, using [`echo_suppression.py`](https://github.com/). As in the procedure used to measure the impact of latency and jitter, [`tc`](https://man7.org/linux/man-pages/man8/tc.8.html) will be used to limit the amount of data an InterCom instance can send in a local environment, simulating a real network environment.

### 1.2 Compressing the audio data with zlib

Reducing the bit-rate requires compressing the data. The `pack()` and `unpack()` methods can respectively compress and decompress the chunks being processed.

The compression algorithm used is **DEFLATE**, a free data codec based on LZSS and Huffman coding. DEFLATE is implemented in Python's standard library through [`zlib`](https://docs.python.org/3/library/zlib.html).

To compare different approaches, the following modules implement different compression strategies:

1. **`DEFLATE_raw.py`**: compresses the raw chunks directly with DEFLATE.
2. **`DEFLATE_serial.py`**: concatenates the channels before compressing the chunk (see Figure 1). This reordering means that samples are no longer interleaved and slightly increases the correlation between consecutive samples. This should also improve the compression ratio.
3. **`DEFLATE_serial_reset.py`**: similar to `DEFLATE_serial.py`, but resets DEFLATE for each new chunk-channel, compressing each chunk-channel independently. The aim is to determine whether DEFLATE exploits redundancy between consecutive channels.
4. **`DEFLATE_byteplanes2.py`**: similar to `DEFLATE_serial.py` (the samples are de-interleaved), but generates two code streams: one for the LSB (Least Significant Byte) plane and another for the MSB (Most Significant Byte) plane. It assumes 16 bits per sample, represented as two 8-bit planes. The aim is to determine whether the MSB can be compressed more efficiently because it may contain runs of zeros, particularly during quiet audio sequences or after quantization.
5. **`DEFLATE_byteplanes3.py`**: similar to `DEFLATE_byteplanes2.py`, but uses three byte planes. This makes it possible to compress coefficients that require more than two bytes to represent.
6. **`DEFLATE_byteplanes4.py`**: uses four byte planes, since some coefficients may require four bytes.
7. **`DEFLATE_byteplanes2_interlaced.py`**: similar to `DEFLATE_byteplanes2.py`, but uses the raw chunks without concatenating the channels.

The number of UDP packets sent remains constant, although each packet can now have a variable length.

*Figure 1 in the original page illustrates the reordering of samples to create two independent channels.*

### 1.3 Quantization

At hardware level, audio samples are usually represented using **Pulse Code Modulation (PCM)**. The number of levels a PCM sample can represent depends on the number of bits per sample; in this case, 16 bits are used.

The Human Auditory System (HAS) has several sources of perceptual redundancy. One of them is the finite number of different volume levels that humans can distinguish. This property can be exploited to reduce transmission bit-rate by accepting a reduction in audio quality.

In most lossy compression schemes, **quantization** is the main source of distortion.

Two common approaches are:

- **Scalar Quantization (SQ):** reduces the number of discrete levels a signal can take.
- **Vector Quantization (VQ):** applies a similar process to tuples of samples simultaneously.

Scalar quantization is appropriate when samples are decorrelated or when their correlation will be exploited later by a signal transform or entropy-coding stage (DEFLATE in this case). Vector quantization offers only a marginal coding-efficiency benefit in this context and generally requires more computational resources.

Quantizers can also be classified as:

- **Uniform:** representation levels are distributed uniformly over the input-value range.
- **Non-uniform:** representation levels are denser in input-value intervals that occur more frequently.
- **Static:** the distribution of representation levels remains constant during quantization.
- **Adaptive:** quantizer parameters are dynamically adjusted to the characteristics of the input signal.

This milestone uses a **uniform, dead-zone, static scalar quantizer**. It can be implemented efficiently in software. Dead-zone quantizers tend to produce more zero-valued quantization indices, improving compression ratios. The trade-off is greater quantization noise for input values close to zero, which reduces the SNR for small signal amplitudes.

This is not necessarily problematic: when the signal amplitude is small and the noise is independent of that amplitude—as is often the case with electronic noise—the input SNR is already at its lowest for values close to zero. The quantizer therefore mainly replaces electronic noise with quantization noise.

Although it will not be exploited in this milestone, dead-zone quantizers are equivalent to encoding the signal using bit planes when the quantization step sizes are powers of two. This can be useful when designing progressive entropy-encoding schemes.

### 1.4 Relationship between the QSS (Δ) and the quantization error

The quantization error signal is:

\[
\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}
\]

where:

- \(\mathbf{x}\) is the original signal.
- \(\hat{\mathbf{x}}\) is the quantized signal.

Consider uniform scalar quantization with step size \(\Delta\). The quantization error can be modelled as a random variable uniformly distributed over each quantization cell:

\[
e_i \sim U\left(-\frac{\Delta}{2}, \frac{\Delta}{2}\right)
\]

Its probability density function (PDF) is:

\[
p(e) =
\begin{cases}
\frac{1}{\Delta}, & |e| \leq \frac{\Delta}{2} \\
0, & \text{otherwise}
\end{cases}
\]

The mean is:

\[
\mathbb{E}[e]
= \int_{-\Delta/2}^{\Delta/2} e\,p(e)\,de
= \frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2}e\,de
= 0
\]

Therefore, the quantization error has zero mean.

The variance, which provides a statistical estimate of the power of a zero-mean signal, is:

\[
\sigma^2 = \mathbb{E}[e^2]
= \int_{-\Delta/2}^{\Delta/2}e^2p(e)\,de
\]

Substituting the probability density function:

\[
\sigma^2
= \frac{1}{\Delta}\int_{-\Delta/2}^{\Delta/2}e^2\,de
\]

Evaluating the integral:

\[
\sigma^2
= \frac{1}{\Delta}
\left[\frac{e^3}{3}\right]_{-\Delta/2}^{\Delta/2}
= \frac{1}{\Delta}
\left(\frac{(\Delta/2)^3-(-\Delta/2)^3}{3}\right)
\]

Since:

\[
(\Delta/2)^3-(-\Delta/2)^3
= 2(\Delta/2)^3
\]

we obtain:

\[
\sigma^2
= \frac{2}{3\Delta}\left(\frac{\Delta}{2}\right)^3
= \frac{2}{3\Delta}\frac{\Delta^3}{8}
= \boxed{\frac{\Delta^2}{12}}
\]

Consequently, the variance of the quantization error is proportional to the square of the quantization step size. Equivalently, its standard deviation is proportional to the step size itself.

### 1.5 (Bit-)Rate control and distortion

The number of representation levels used by a quantizer depends on its step size, usually denoted by \(\Delta\).

- A larger \(\Delta\) means fewer representation levels.
- Fewer levels produce greater quantization error and therefore greater distortion.
- A larger \(\Delta\) generally reduces the output bit-rate.

This is the **rate-distortion trade-off** common to lossy compression: using more bits and smaller quantization intervals reduces distortion, while using fewer bits increases it.

In real-time transmission systems, the bit-rate can be controlled by modifying \(\Delta\) when congestion occurs, with the aim of minimizing data loss.

However, the relationship between \(\Delta\) and the resulting bit-rate is not necessarily direct. It depends on the entropy-coding stage (DEFLATE) and on signal characteristics such as variance and entropy. DEFLATE typically produces a variable-bit-rate code stream.

A rate-control algorithm based on quantization can be described using a **Rate-Distortion (RD) curve**:

- The X axis represents the received bit-rate.
- The Y axis represents the distortion in the reconstructed signal—in InterCom, the audio that is played.

Examples are available in the notebooks linked from the original page.

### 1.6 The current implementations for the control of the bit-rate

Bit-rate (BR) control through quantization is implemented in the `BR_Control*` classes found in the `BR_control*.py` modules.

These classes override the inherited `pack()` and `unpack()` methods. The chunks are already DEFLATE-encoded/decoded at this point.

Conceptually, `pack()` performs:

```python
def pack(chunk_number, chunk):
    quantized_chunk = quantize(chunk)  # (1)
    packed_chunk = Buffering.pack(chunk_number, quantized_chunk)  # (2)
    return packed_chunk  # (3)
```

And `unpack()` performs:

```python
def unpack(packed_chunk):
    (chunk_number, quantized_chunk) = Buffering.unpack(packed_chunk)  # (1)
    chunk = dequantize(quantized_chunk)  # (2)
    return (chunk_number, chunk)  # (3)
```

The milestone includes four implementations:

#### 1. `BR_control_no.py`

Uses a constant \(\Delta > 0\). There is no bit-rate control.

#### 2. `BR_control_add_lost.py`

Every second, it updates the quantization step as follows:

\[
\begin{cases}
\Delta \leftarrow \Delta + L - 1 & \text{always} \\
\Delta \leftarrow \Delta_{\min} & \text{if } \Delta < \Delta_{\min}
\end{cases}
\]

where \(L\) is the number of chunks lost (received, according to the page's wording) during the previous second.

This heuristic—and the following ones—assumes that the other participant loses, on average, the same number of chunks as the local participant.

#### 3. `BR_control_lost.py`

Every second, it updates the quantization step as follows:

\[
\begin{cases}
\Delta \leftarrow L - 1 & \text{always} \\
\Delta \leftarrow \Delta_{\min} & \text{if } \Delta < \Delta_{\min}
\end{cases}
\]

#### 4. `BR_control_conservative.py`

Every second, it applies the following rules:

\[
\begin{cases}
\Delta \leftarrow 2\Delta & \text{if } L > 2 \\
\Delta \leftarrow \frac{10}{11}\Delta & \text{always} \\
\Delta \leftarrow \Delta_{\min} & \text{if } \Delta < \Delta_{\min}
\end{cases}
\]

---

## 2. Deliverables

The results must be reported in a **Jupyter notebook**.

### 2.1 Which data ordering performs better?

Determine empirically which ordering of chunk data is most efficient from the perspective of lossless data compression. A smaller bit-rate means a higher compression ratio.

Use an audio sequence of your choice. Different audio files may produce different results.

### 2.2 Which BR control algorithm has the best RD curve?

For a specific audio sequence, generate the Rate-Distortion (RD) curve considering several simulated transmission environments. Use `tc` or another similar tool to simulate these environments.

Use **RMSE (Root Mean Square Error)** as the distortion measure between the sent and received audio signals.

Remember:

- The X axis must represent bit-rate, not \(\Delta\).
- The Y axis must represent distortion (RMSE).

---

## 3. Resources

1. M. Bosi and R. E. Goldberg, *Introduction to Digital Audio Coding and Standards*. Kluwer Academic Publishers, 2003.
2. Behrouz Forouzan, *Introduction to Data Communications and Networking*. McGraw-Hill, 2007.
3. Bert Hubert, Thomas Graf, Greg Maxwell, Remco van Mook, Martijn van Oosterhout, Paul B. Schroeder, Jasper Spaans, and Pedro Larroy, *Linux Advanced Routing & Traffic Control*. 2012.
4. Nelson M. and Gailly, *The Data Compression Book*. M&T Books, 1996.
5. K. Sayood, *Introduction to Data Compression*. Morgan Kaufmann, 2017.
6. Andrew S. Tanenbaum, *Computer Networks*. Prentice Hall, 2011.
7. D. S. Taubman and W. M. Marcellin, *JPEG2000: Image Compression Fundamentals, Standards and Practice*. Kluwer Academic Publishers, 2002.
8. M. Vetterli, J. Kovačević, and V. K. Goyal, *Foundations of Signal Processing*. Cambridge University Press, 2014.

---

## Notes from the original page

- The bit-rate is measured in bits per second, or as a multiple of the transmission capacity.
- Limiting the bit-rate can also cause chunk loss: if link capacity is below the audio bit-rate, chunks that cannot be buffered in retransmission nodes (routers and switches) will eventually be discarded. In that situation, the sender may contribute to, or even cause, link congestion.
- If `tc` or a similar tool is unavailable on the operating system, a real transmission environment can be used. The bit-rate must still be controlled to obtain the points of the RD curve.
- Some coefficients used in future InterCom improvements may require more than two bytes; coefficients can require up to four bytes.
- Quantizer decision intervals and representation levels can also be optimized according to other criteria, such as minimizing the rate-distortion value at a given point on the RD curve.
- De-quantization restores the original dynamic range, but does not necessarily restore the original signal. Exact restoration only occurs when \(\Delta = 1\).
- \(\Delta\) must always be greater than zero, independently of the bit-rate-control algorithm.
- Some audio samples are available in InterCom's `data` directory.

**Source:** https://tecnologias-multimedia.github.io/contents/BR_control/

> This Markdown document reorganizes the page into a readable Markdown structure. The original page also contains an illustration (Figure 1) and links to code, notebooks and references; those external resources are linked where identifiable, but the image itself is not embedded here.
