# Public data used by `kg_omega.py`

| file | source | what |
|---|---|---|
| `metaqa_kb.txt` | MetaQA (Zhang et al., AAAI 2018), WikiMovies KB, mirror `huggingface.co/datasets/camazlucas/MetaQA` (`kb/kb.txt`) | 134,741 `subject|relation|object` triples |
| `metaqa_2-hop_test.txt`, `metaqa_3-hop_test.txt` | same mirror, `{2,3}-hop/vanilla/qa_test.txt` | 14,872 and 14,274 test questions with gold answers |
| `hetionet-edges.sif.gz`, `hetionet-nodes.tsv` | Hetionet v1.0 (Himmelstein et al., eLife 2017), `github.com/hetio/hetionet/hetnet/tsv/` | 47,031 nodes, 2,250,197 typed edges; CC0 |

| `wikimovies_wiki.txt` | WikiMovies (Miller et al., EMNLP 2016), `movieqa/knowledge_source/wiki.txt` from `movieqa.tar.gz` (Internet Archive copy of the original bAbI host; sha256 checked) | 18,127 movie articles; provenance documents for `e2e_factiso.py` |

`fetch_data.sh` re-downloads all six files. Hetionet is the Medical domain of the
GragPoison benchmark (Liang et al., S&P 2026); MetaQA is the standard multi-hop KGQA workload.
