"""A GBNF grammar that makes a malformed citation impossible to generate.

Constrained decoding is the same move as `make_chunk` slicing its own text: it
converts a rule that would otherwise be checked after the fact into one the
generator cannot break. The grammar pins the JSON shape AND the citation index
range, so `citation_indices` can only ever name a hit that was actually shown.

`ws` is bounded (`{{0,4}}`), not `*`. An unbounded `[ \\t\\n]*` was tried first and
reproduced against the real model: at `temperature=0.0`, grammar-masked greedy
decoding got stuck in a whitespace loop immediately after the opening `{{`,
consuming the entire `max_tokens` budget on nothing but tabs and never reaching
`"sentences"` -- a genuine degenerate attractor, not a slow-but-working case
(`finish_reason == "length"`, 512/512 completion tokens, all whitespace). Bounding
the repeat count removes the attractor -- there is no longer a token position
where "emit more whitespace" stays legal indefinitely -- while still tolerating
whatever incidental formatting whitespace a model emits between tokens.
"""

_TEMPLATE = """\
root       ::= "{{" ws "\\"sentences\\"" ws ":" ws sentences ws "}}"
sentences  ::= "[" ws sentence (ws "," ws sentence)* ws "]"
sentence   ::= "{{" ws "\\"text\\"" ws ":" ws string ws "," ws \
"\\"citations\\"" ws ":" ws citations ws "}}"
citations  ::= "[" ws (index (ws "," ws index)*)? ws "]"
index      ::= {indices}
string     ::= "\\"" char* "\\""
char       ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]
ws         ::= [ \\t\\n]{{0,4}}
"""


def build_grammar(n_hits: int) -> str:
    """GBNF allowing citation indices `1..n_hits` and nothing else."""
    if n_hits < 1:
        raise ValueError(
            "a grammar needs at least one citable hit; with nothing retrieved the "
            "caller must refuse instead of generating"
        )
    indices = " | ".join(f'"{i}"' for i in range(1, n_hits + 1))
    return _TEMPLATE.format(indices=indices)
