## MetNet-3 - Pytorch (wip)

Implementation of <a href="https://blog.research.google/2023/11/metnet-3-state-of-art-neural-weather.html">MetNet 3</a>, SOTA neural weather model out of Google Deepmind, in Pytorch

The model architecture is pretty unremarkable. It is basically a U-net with a specific <a href="https://arxiv.org/abs/2204.01697">well performing vision transformer</a>. The most interesting thing about the paper may end up being the loss scaling in section 4.3.2

## Citations

```bibtex
@article{Andrychowicz2023DeepLF,
    title   = {Deep Learning for Day Forecasts from Sparse Observations},
    author  = {Marcin Andrychowicz and Lasse Espeholt and Di Li and Samier Merchant and Alexander Merose and Fred Zyda and Shreya Agrawal and Nal Kalchbrenner},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2306.06079},
    url     = {https://api.semanticscholar.org/CorpusID:259129311}
}
```
