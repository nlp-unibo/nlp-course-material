# Assignment 2 data

Derived from **MAFALDA** (Helwe et al., 2024), `gold_standard_dataset.jsonl`:
https://github.com/ChadiHelwe/MAFALDA

The span annotations are dropped and each text keeps the set of Level 1
categories of the fallacies it contains, following the `LEVEL_2_TO_LEVEL_1`
mapping of `src/evaluate.py` in the MAFALDA repository. The `nothing` and
`to clean` labels of the release are discarded. Texts with no remaining label
contain no fallacy.

| file | texts | no fallacy | credibility | emotion | logic |
|---|---|---|---|---|---|
| `a2_test.csv` | 156 | 50 | 42 | 28 | 76 |
| `demonstrations.csv` | 44 | 13 | 13 | 10 | 22 |

Columns: `mafalda_id` (line of the source file), `text`, `labels_level1` and
`labels_level2`, both semicolon-separated and empty when the text carries no
fallacy.

The two files are stratified over the Level 1 label sets, so every combination
appears in both.

`prepare_data.py`, in the private solutions repository, is the script that
produced these files.

## License

MAFALDA is released under **CC BY-SA 4.0**, and so are these derived files.

> Chadi Helwe, Tom Calamai, Pierre-Henri Paris, Chloé Clavel, Fabian Suchanek.
> 2024. MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and
> Classification. NAACL 2024, pages 4810-4845.
