"""e2e_table.py -- the LaTeX rows of Table e2e-attacks, generated from results_e2e.json.
audit_paper.py imports this, so the table in the paper is checked character for character."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"gragpoison": "shared-relation", "kepo": "modification", "mincut": "min-cut adaptive",
         "injection": "prompt injection"}


def fmt(x):
    return "---" if x is None else "%.1f" % x


def rows(E=None):
    E = E or json.load(open(os.path.join(HERE, "results_e2e.json")))
    A = E["summary"]["attacks"]
    out = []
    for k in ("gragpoison", "kepo", "mincut", "injection"):
        for b in (1, 3):
            m, h = A.get("metaqa/%s/b%d" % (k, b)), A.get("hetionet/%s/b%d" % (k, b))
            out.append("%s & %d & %s & %s & %s & %s \\\\" % (
                NAMES[k] if b == 1 else "", b, fmt(m["defended_flip"]), fmt(m["undefended_flip"]),
                fmt(m["defended_success"]), fmt(h["defended_success"] if h else None)))
    return out


if __name__ == "__main__":
    print("\n".join(rows()))
