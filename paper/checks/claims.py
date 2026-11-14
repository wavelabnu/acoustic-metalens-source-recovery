"""claims.py -- read what the manuscript SAYS, so no number is transcribed by hand.

Why this file exists
--------------------
`numbers_ledger.py` used to print its computed value next to a hardcoded string
("PAPER Sec. Results: rank 4 / 5 / 8 / 8 ; M = 41 and 52").  Nobody updated those
strings when the text was fixed, so by 2026-09-02 the ledger was reporting false
mismatches: it still claimed the paper printed the pre-2026-08-24 dB ladder and
M = 52, while main.tex and supplement.tex already carried the corrected values.

A checker that carries its own copy of the thing it checks is the same inverted
provenance the 2026-08-17 dB-ladder fix was about, one level up.  So: the ONLY
source for "what the paper says" is the .tex, and the ONLY source for "what is
true" is the `direct` package.  This module is the bridge, and it holds neither.

Two rules make it validatable rather than decorative:

1.  A pattern that does not match is an ERROR, not a pass.  If someone rewrites
    a sentence, the check fails loudly with the pattern that went stale, instead
    of silently checking nothing.

2.  The tolerance is READ OFF the manuscript, never chosen here.  A value written
    "$2.5\\times$" claims one decimal, so it is satisfied by anything within half
    a unit of the last quoted digit (2.45-2.55) and by nothing else.  Writing
    more digits in the paper automatically tightens the test.

Usage
-----
    from claims import Manuscript, Ledger

    main = Manuscript('main.tex')
    led  = Ledger()
    led.check('weakest-channel margin, coupled',
              computed=52.53,
              quoted=main.number(r'\\$41\\$ for\\s+independent drivers and \\$(NUM)\\$ when'))
    raise SystemExit(led.report())
"""
from __future__ import annotations

import pathlib
import re

PAPER = pathlib.Path(__file__).resolve().parent.parent

# A LaTeX-ish number: optional sign, digits, optional decimals.  Patterns passed
# to Manuscript.number() write NUM where the value sits; it expands to this and
# becomes the capture group.
_NUM = r'([-+]?\d+(?:\.\d+)?)'


class ClaimNotFound(LookupError):
    """A pattern matched nothing -- the sentence moved, so the check is stale."""


def split_review_notes(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Separate visible author-review notes from scientific text, preserving lines.

    Balanced arguments allow math/macros in notes. Malformed markup fails rather
    than silently swallowing a scientific claim or changing word-count scope.
    """
    pattern = re.compile(r'\\reviewnote\s*(?=\{)')
    pieces, notes, start = [], [], 0
    for match in pattern.finditer(text):
        if match.start() < start:
            continue
        end, args = match.end(), []
        for _ in range(2):
            while end < len(text) and text[end].isspace():
                end += 1
            if end == len(text) or text[end] != '{':
                raise ValueError('reviewnote needs two braced arguments')
            begin, depth = end + 1, 1
            end += 1
            while end < len(text) and depth:
                if text[end] == '\\':
                    end += 2
                    continue
                depth += {'{': 1, '}': -1}.get(text[end], 0)
                end += 1
            if depth:
                raise ValueError('unclosed reviewnote argument')
            args.append(text[begin:end-1])
        pieces.extend((text[start:match.start()], '\n' * text[match.start():end].count('\n')))
        notes.append(tuple(args))
        start = end
    return ''.join(pieces) + text[start:], notes


class Manuscript:
    """One .tex file, searched for the numbers it prints."""

    def __init__(self, name: str):
        self.name = name
        self.path = PAPER / name
        self.text, _ = split_review_notes(self.path.read_text(encoding='utf-8'))

    def _search(self, pattern: str):
        # A literal space in a pattern should match a line break too: LaTeX
        # reflows, and a pattern that breaks because a sentence moved to a new
        # line is a false alarm.  Done before NUM is expanded so the number
        # pattern itself is untouched.
        expanded = re.sub(r' +', r'\\s+', pattern).replace('NUM', _NUM)
        m = re.search(expanded, self.text, re.S)
        if m is None:
            raise ClaimNotFound(
                f'{self.name}: no match for\n      {pattern}\n'
                '    The sentence carrying this claim was rewritten or removed. '
                'Re-anchor the pattern, or drop the check if the claim is gone.')
        return m

    def number(self, pattern: str) -> 'Quoted':
        """The single NUM in `pattern`, with the precision it was written at."""
        m = self._search(pattern)
        if m.re.groups != 1:
            raise ValueError(f'pattern must contain exactly one NUM, got {m.re.groups}')
        return Quoted(m.group(1), self, pattern)

    def numbers(self, pattern: str) -> list['Quoted']:
        """Every NUM in `pattern`, in order, each with its own precision."""
        m = self._search(pattern)
        return [Quoted(g, self, pattern) for g in m.groups()]

    def line_of(self, pattern: str) -> int:
        """1-based line number where `pattern` matches -- for error messages."""
        return self.text.count('\n', 0, self._search(pattern).start()) + 1

    def count(self, pattern: str) -> int:
        return len(re.findall(pattern.replace('NUM', _NUM), self.text, re.S))


class Quoted:
    """A number as the manuscript writes it, carrying its own precision.

    `2.5` claims one decimal and so tolerates +-0.05; `2.50` claims two and
    tolerates +-0.005.  The manuscript therefore sets the test's strictness,
    and nothing here does.
    """

    def __init__(self, literal: str, doc: Manuscript, pattern: str):
        self.literal = literal
        self.value = float(literal)
        self.doc = doc
        self.pattern = pattern
        self.decimals = len(literal.split('.')[1]) if '.' in literal else 0

    @property
    def tolerance(self) -> float:
        """Half a unit in the last digit the manuscript actually wrote."""
        return 0.5 * 10.0 ** (-self.decimals)

    def accepts(self, computed: float) -> bool:
        # the 1e-9 slack is float representation, not a physics allowance:
        # it keeps 8.95 vs "9.0" (exactly half a unit) from failing on binary
        # rounding alone.
        return abs(computed - self.value) <= self.tolerance * (1 + 1e-9)

    def __repr__(self):
        return self.literal


class Ledger:
    """Collects checks, prints a table, and decides the exit code."""

    def __init__(self, title: str = ''):
        self.rows: list[tuple] = []
        self.failures = 0
        self.errors = 0
        if title:
            self.header(title)

    @staticmethod
    def header(t: str):
        print('\n' + '=' * 78)
        print(t)
        print('=' * 78)

    @staticmethod
    def info(label: str, text: str):
        print(f'  INFO   {label:<44s} {text}')

    def check(self, label: str, computed: float, quoted: Quoted, unit: str = ''):
        """Assert the computed value rounds to what the manuscript prints."""
        if quoted.accepts(computed):
            status = 'ok  '
        else:
            status = 'FAIL'
            self.failures += 1
        self.rows.append((status, label))
        delta = computed - quoted.value
        print(f'  {status}   {label:<44s} code {computed:>10.3f}{unit}   '
              f'paper {quoted.literal:>8s}{unit}   d={delta:+.3f}  (tol +-{quoted.tolerance:g})')
        if status == 'FAIL':
            print(f'         ^ {quoted.doc.name}:{quoted.doc.line_of(quoted.pattern)}')

    def check_seq(self, label: str, computed: list[float], quoted: list[Quoted],
                  unit: str = ''):
        """Same, element by element, for a list the paper prints in a row."""
        if len(computed) != len(quoted):
            self.fail(label, f'code has {len(computed)} values, paper prints {len(quoted)}')
            return
        bad = [(i, c, q) for i, (c, q) in enumerate(zip(computed, quoted)) if not q.accepts(c)]
        status = 'ok  ' if not bad else 'FAIL'
        if bad:
            self.failures += 1
        self.rows.append((status, label))
        print(f'  {status}   {label:<44s} code ' +
              ' '.join(f'{c:g}' for c in computed) + unit)
        print(f'         {"":<44s} paper ' +
              ' '.join(q.literal for q in quoted) + unit)
        for i, c, q in bad:
            print(f'         ^ position {i}: code {c:g} vs paper {q.literal} '
                  f'(tol +-{q.tolerance:g}), {q.doc.name}:{q.doc.line_of(q.pattern)}')

    def check_ints(self, label: str, computed, quoted, note: str = ''):
        """Exact comparison, for counts and sets where rounding is meaningless."""
        computed, quoted = list(computed), list(quoted)
        ok = computed == quoted
        status = 'ok  ' if ok else 'FAIL'
        if not ok:
            self.failures += 1
        self.rows.append((status, label))
        print(f'  {status}   {label:<44s} code {computed}   paper {quoted}'
              + (f'   {note}' if note else ''))

    def fail(self, label: str, why: str):
        self.failures += 1
        self.rows.append(('FAIL', label))
        print(f'  FAIL   {label:<44s} {why}')

    def error(self, label: str, exc: Exception):
        """A pattern went stale.  Louder than a failure: the check ran on nothing."""
        self.errors += 1
        self.rows.append(('ERR ', label))
        print(f'  ERR    {label:<44s} {exc}')

    def guarded(self, label: str, fn):
        """Run one check, turning a stale pattern into a reported ERR."""
        try:
            fn()
        except ClaimNotFound as exc:
            self.error(label, exc)

    def report(self) -> int:
        n = len(self.rows)
        ok = sum(1 for s, _ in self.rows if s == 'ok  ')
        print('\n' + '-' * 78)
        print(f'  {ok}/{n} checks passed'
              + (f', {self.failures} FAILED' if self.failures else '')
              + (f', {self.errors} stale pattern(s)' if self.errors else ''))
        if self.failures or self.errors:
            print('  Failing claims, in manuscript order:')
            for s, label in self.rows:
                if s != 'ok  ':
                    print(f'    {s}  {label}')
        print('-' * 78)
        return 1 if (self.failures or self.errors) else 0
