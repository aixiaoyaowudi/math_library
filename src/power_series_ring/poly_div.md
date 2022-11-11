给定 $f,h\in R\left[\left[x\right]\right]$，令 $g=1/f$，求 $q=hg=h/f \pmod{x^n}$

函数定义：$Div\left(f,h,dst,tmp_1,tmp_2,tmp_3\right)$

令 $g_0=g\pmod{x^{ n/2 }},h_0=h\pmod {x^{n/2}},q_0=q\pmod {x^{n/2}}$

即求 $q\bmod{x^{2n}}=q_0-\left(fq_0-h\right)g_0\bmod{x^{n}}$

调用 $\mathrm{Inv}\left(f,dst,tmp_1,tmp_2,tmp_3\right)$ 求出 $g_0$，并且 $tmp_3$ 中存储了 $\mathcal{F}_{n/2}\left(g_0\bmod{x^{n/4}}\right)$，花费 $4.5\mathrm{E}\left(n\right)$ 时间

令 $g_1=\left(g_0-g_0\bmod{x^{n/4}}\right)/x^{n/4}$

令 $h_1=\left(h_0-h_0\bmod{x^{n/4}}\right)/x^{n/4}$

花费 $0.5\mathrm{E}\left(n\right)$ 时间计算出 $\mathcal{F}_{n/2}\left(g_1\right)$，存储在 $tmp_3$ 的高 $n/2$ 位内。

花费 $1\mathrm{E}\left(n\right)$ 时间计算出 $\mathcal{F}_{n/2}\left(h_0\bmod{x^{n/4}}\right),\mathcal{F}_{n/2}\left(h_1\right)$，分别存储在 $tmp_2$ 的低 $n/2$ 和高 $n/2$ 位。

$q_0=\left(g_0\bmod{x^{n/4}}+g_1\times x^{n/4}\right)\times \left(h_0\bmod{x^{n/4}}+h_1\times x^{n/4}\right)\pmod {x^{n/2}}$，利用前置技巧，花费 $1\mathrm{E}\left(n\right)$ 时间计算出 $q_0$

乘法后的低位存储在 $tmp_1$ 的低 $n/2$ 位和高 $n/2$ 位。IDFT 后结果存在 $tmp_1$ 的低 $n/2$ 位。

接下来求 $\left(fq_0-h\right)g_0\bmod{x^{n}}$

花费 $3\mathrm{E}\left(n\right)$ 时间计算出 $fq_0$，取 $n/2\sim n-1$ 次项并减去 $h$ 的 $n/2\sim n-1$ 次项，作为 $\left(fq_0-h\right)/x^n$，设为 $t_0+t_1\times x^{n/4}$

$f$ 的 DFT 存在 $tmp_2$ 中，$q_0$ 先拷贝到 $dst$，然后原地 DFT，然后结果存到 $tmp_2$ 中 IDFT

花费 $1\mathrm{E}\left(n\right)$ 计算出 $\mathcal{F}_{n/2}\left(t_0\right),\mathcal{F}_{n/2}\left(t_1\right)$，存在 $tmp_2$ 中

花费 $1\mathrm{E}\left(n\right)$ 求出 $\left(\left(fq_0-h\right)/x^{n/2}\right)\times g_0\bmod{x^{n/2}}$，结果存在 $tmp_1$ 中

最后将结果取负存到 $dst$ 中

总共 $12\mathrm{E}\left(n\right)$ 时间
