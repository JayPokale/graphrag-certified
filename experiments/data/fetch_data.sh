#!/usr/bin/env bash
# Re-download the public data behind results_kgomega.json.  ~25 MB total.
set -euo pipefail
cd "$(dirname "$0")"
HF=https://huggingface.co/datasets/camazlucas/MetaQA/resolve/main
curl -sL -o metaqa_kb.txt        "$HF/kb/kb.txt"
curl -sL -o metaqa_2-hop_test.txt "$HF/2-hop/vanilla/qa_test.txt"
curl -sL -o metaqa_3-hop_test.txt "$HF/3-hop/vanilla/qa_test.txt"
HN=https://github.com/hetio/hetionet/raw/main/hetnet/tsv
curl -sL -o hetionet-edges.sif.gz "$HN/hetionet-v1.0-edges.sif.gz"
curl -sL -o hetionet-nodes.tsv    "$HN/hetionet-v1.0-nodes.tsv"
# WikiMovies articles (Miller et al. 2016): provenance text for e2e_factiso.py.  The original
# host is gone; the Internet Archive copy of movieqa.tar.gz is byte-identical to the release.
WM='https://web.archive.org/web/2019id_/http://www.thespermwhale.com/jaseweston/babi/movieqa.tar.gz'
curl -sL -o movieqa.tar.gz "$WM"
echo "ed062b49922b602ebee6073f58951bf38c4772a8b53d46682f3ff80ed57de948  movieqa.tar.gz" | sha256sum -c -
tar xzf movieqa.tar.gz movieqa/knowledge_source/wiki.txt 2>/dev/null
mv movieqa/knowledge_source/wiki.txt wikimovies_wiki.txt && rm -rf movieqa movieqa.tar.gz
ls -la
