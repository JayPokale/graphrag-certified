#!/usr/bin/env bash
# run_llm_measurement.sh -- measure eps' and kappa against a served model.
#
# The one experiment the paper ships a harness for rather than a result (Sec. 10).
# Inference happens on the SERVER behind $API_BASE, so this client needs no GPU --
# only python3, numpy, scipy and network access.
#
#   export LLM_API_KEY=...            # never put the key on the command line
#   tmux new -s llm
#   ./run_llm_measurement.sh
#
# Resume after a disconnect with:  tmux attach -t llm
set -euo pipefail

API_BASE="${API_BASE:?set API_BASE=http://<host>:<port>/v1}"
API_MODEL="${API_MODEL:-qwen-122b}"
CONCURRENCY="${CONCURRENCY:-16}"
QUERIES="${QUERIES:-60}"
OUT="${OUT:-results_llm.json}"
LOG="${LOG:-llm_run_$(date +%Y%m%d_%H%M%S).log}"

cd "$(dirname "$0")"

# Use the venv if present. It exists to keep a broken ~/.local numpy off sys.path.
VENV="${VENV:-$HOME/.venvs/graphrag-certified}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi
export PYTHONNOUSERSITE=1

if ! python3 -c "import numpy, scipy" 2>/dev/null; then
  echo "numpy/scipy not importable. Run ./setup_env.sh first." >&2
  python3 -c "import numpy" 2>&1 | tail -2 >&2 || true
  exit 1
fi

if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "LLM_API_KEY is not set. Run:  export LLM_API_KEY=<token>" >&2
  exit 1
fi

echo "endpoint : $API_BASE"
echo "model    : $API_MODEL"
echo "log      : $LOG"
echo

# 1. Reachability. Fail here rather than 40 minutes in.
echo "== checking endpoint =="
code=$(curl -s -o /tmp/_llm_probe.json -w '%{http_code}' --max-time 60 \
  "$API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LLM_API_KEY" \
  -d "{\"model\":\"$API_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"say OK\"}],\"max_tokens\":5}")
if [[ "$code" != "200" ]]; then
  echo "endpoint returned HTTP $code:" >&2; cat /tmp/_llm_probe.json >&2; exit 1
fi
echo "ok: $(head -c 200 /tmp/_llm_probe.json)"; rm -f /tmp/_llm_probe.json
echo

# 2. Probe: show raw replies. qwen-122b emits a "Thinking Process" preamble, so a small
#    max_tokens truncates every reply before the verdict and the whole sweep is void.
echo "== probe: raw replies =="
python3 run_on_gpu.py \
  --api-base "$API_BASE" --api-model "$API_MODEL" \
  --api-max-tokens "${MAX_TOKENS:-512}" ${API_EXTRA:+--api-extra "$API_EXTRA"} \
  --probe 3
echo
read -r -p "Do the replies above parse? [y/N] " ans
[[ "$ans" == [yY]* ]] || { echo "stopping; tune MAX_TOKENS or API_EXTRA" >&2; exit 1; }
echo

# 3. Smoke test: real endpoint, tiny sweep. Confirms prompts parse before the long run.
echo "== smoke test (about 100 calls) =="
python3 run_on_gpu.py \
  --api-base "$API_BASE" --api-model "$API_MODEL" \
  --queries 3 --cells 16 --loads 8 --payloads none targeted \
  --api-max-tokens "${MAX_TOKENS:-512}" ${API_EXTRA:+--api-extra "$API_EXTRA"} \
  --concurrency "$CONCURRENCY" --out /tmp/_llm_smoke.json 2>&1 | tee "smoke_$LOG"
rm -f /tmp/_llm_smoke.json
echo
echo "If the unparseable rate above is not ~0%, STOP and inspect the endpoint's"
echo "reply format before trusting the full run."
echo

# 4. Full sweep. ~28,800 calls at the defaults; budget 1-3 hours.
echo "== full sweep =="
time python3 run_on_gpu.py \
  --api-base "$API_BASE" --api-model "$API_MODEL" \
  --queries "$QUERIES" \
  --cells 16 64 --loads 8 24 --payloads none generic targeted \
  --api-max-tokens "${MAX_TOKENS:-512}" ${API_EXTRA:+--api-extra "$API_EXTRA"} \
  --concurrency "$CONCURRENCY" --out "$OUT" 2>&1 | tee "$LOG"

echo
echo "wrote $OUT"
echo "Report kappa_max and eps' per payload. Then update Sec. 10, which currently"
echo "states this measurement was NOT made, and add the numbers to audit_paper.py."
