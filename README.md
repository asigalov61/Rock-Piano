# Rock Piano

### "When all is one and one is all, that's what it is to be a rock and not to roll." ---Led Zeppelin, "Stairway To Heaven"

***

## Proof-Of-Concept Piano-Drums Music AI Model/Implementation

[![Open In Colab][colab-badge3]][colab-notebook3]

[colab-notebook3]: <https://colab.research.google.com/github/asigalov61/Rock-Piano/blob/main/Rock-Piano.ipynb>
[colab-badge3]: <https://colab.research.google.com/assets/colab-badge.svg>

***

### Model Applications

1) Semi-original Piano-Drums performance generation

2) Semi-original Piano-Drums continuations generation

3) Piano-conditioned Drums generation

4) Drums-conditioned Piano generation

5) Other possible uses

***

### Model Details

1) Trained upon ~500 Piano-Drums excerpts which were randomly selected from ~2392 western rock music masterpieces

2) Model SEQ: [channel, delta-start-time, pitch, duration, EOS]

***

### Model Use Tips:

1) Piano-conditioned Drums generation can be envoked by [EOS, 9] sequence
2) Drums-conditioned Piano generation can be envoked by [EOS, 0] sequence
3) To generate a note at the same time point use i.e. [EOS, 9, 0] sequence
4) To generate a note at the future time point use i.e. [EOS, 9, desired-delta-time] i.e. [EOS, 9, 7] sequence
5) Just in case, EOS == 500, Piano channel == 0, Drums channel == 9

***

### Citation

```bibtex
@inproceedings{lev2021rockpiano,
    title       = {Rock Piano},
    author      = {Aleksandr Lev},
    booktitle   = {GitHub},
    year        = {2021},
}
```

***

### Project Los Angeles

### Tegridy Code 2021

