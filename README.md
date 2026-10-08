<div align="center">
  <img src="Figures/Youtu/tencentyoutu.jpg" alt="AdaMARP Logo" width="200"/>
</div>

# AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing

## News

Our paper has been accepted to **Findings of ACL 2026**. We have open-sourced the
AdaMARP implementation, data construction pipelines, training scripts, evaluation
code, and datasets in this repository.

## Getting started

- [Code and reproducibility guide](AdaptiveRPMAS/README.md)
- [Paper on ACL Anthology](https://aclanthology.org/2026.findings-acl.1563/)
- [Project page](https://xuzhenhua55.github.io/AdaMARP/)

The release includes role-playing inference, data construction pipelines, training
scripts, Actor and Scene Manager evaluation, and datasets. See the
[code guide](AdaptiveRPMAS/README.md) to get started.

Datasets are stored with **Git LFS**. Install Git LFS before cloning, then run:

```bash
git lfs install
git lfs pull
```

## License

The code is licensed under [Apache-2.0](LICENSE). Datasets and third-party materials
are excluded from this code license and remain subject to their applicable terms.

## Citation

If you find the code, datasets and models in this repository useful, please cite:

```bibtex
@inproceedings{xu-etal-2026-adamarp,
    title = "{A}da{MARP}: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing",
    author = "Xu, Zhenhua  and
      Chen, Dongsheng  and
      Wang, Shuo  and
      Li, Jian  and
      Wang, Chengjie  and
      Han, Meng  and
      Wang, Yabiao",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1563/",
    doi = "10.18653/v1/2026.findings-acl.1563",
    pages = "31230--31283",
    ISBN = "979-8-89176-395-1",
    abstract = "LLM role-playing seeks to portray arbitrary characters in interactive narratives, yet existing systems often lack immersion and adapt ability. They typically under-model dynamic environment information and assume a largely static scene/cast, offering limited support for multi-character orchestration, scene transitions, and on-the-fly character introduction. We propose an adaptive multi-agent interaction framework dubbed AdaMARP, which featuring an immersive message format that interleaves [Thought], (Action), Environment, and Speech, and an explicit Scene Manager that controls role-playing via discrete actions (init{\_}scene, pick{\_}speaker, switch{\_}scene, add{\_}role, end) with rationales. To train these abilities, we construct AdaRPSet for the Actor Model and AdaSMSet for supervising or chestration decisions, and introduce AdaptiveBench for trajectory-level evaluation. Experiments across multiple backbones and scales show consistent gains: AdaRPSet improves character consistency, environment grounding, and narrative coherence{---}an 8B actor outperforming several commercial LLMs, while AdaSMSet enables smoother scene transitions and more natural role introductions, surpassing Claude Sonnet 4.5 with only 14B LLMs."
}
```
