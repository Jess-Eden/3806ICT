import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

@dataclass
class Atom:
    name: str

    def __str__(self):
        return self.name


@dataclass
class Predicate:
    name: str
    args: List[Any]

    def __str__(self):
        arg_strings = []
        for a in self.args:
            arg_strings.append(str(a))
        joined_args = ', '.join(arg_strings)
        return f"{self.name}({joined_args})"


@dataclass
class Var:
    name: str

    def __str__(self):
        return self.name


@dataclass
class FuncTerm:
    name: str
    args: List[Any] = field(default_factory=list)

    def __str__(self):
        if self.args:
            arg_strings = []
            for a in self.args:
                arg_strings.append(str(a))
            joined_args = ', '.join(arg_strings)
            return f"{self.name}({joined_args})"
        else:
            return self.name


@dataclass
class Neg:
    formula: Any

    def __str__(self):
        inner = str(self.formula)
        if isinstance(self.formula, (And, Or, Implies)):
            return f"\u00ac({inner})"
        else:
            return f"\u00ac{inner}"


@dataclass
class And:
    left: Any
    right: Any

    def __str__(self):
        return f"({self.left} \u2227 {self.right})"


@dataclass
class Or:
    left: Any
    right: Any

    def __str__(self):
        return f"({self.left} \u2228 {self.right})"


@dataclass
class Implies:
    left: Any
    right: Any

    def __str__(self):
        return f"({self.left} \u2192 {self.right})"


@dataclass
class Forall:
    var: str
    formula: Any

    def __str__(self):
        return f"\u2200{self.var}.{self.formula}"


@dataclass
class Exists:
    var: str
    formula: Any

    def __str__(self):
        return f"\u2203{self.var}.{self.formula}"


class Sequent:
    antecedent: List[Any]
    succedent: List[Any]

    def __init__(self, antecedent: List[Any] = None, succedent: List[Any] = None):
        if antecedent is not None:
            self.antecedent = antecedent
        else:
            self.antecedent = []

        if succedent is not None:
            self.succedent = succedent
        else:
            self.succedent = []

    def __str__(self) -> str:
        lhs_items = []
        for f in self.antecedent:
            lhs_items.append(str(f))
        lhs = ', '.join(lhs_items)

        rhs_items = []
        for f in self.succedent:
            rhs_items.append(str(f))
        rhs = ', '.join(rhs_items)

        if lhs:
            return f"{lhs} ⊢ {rhs}"
        else:
            return f"⊢ {rhs}"

    def canonical(self) -> str:
        lhs_items = []
        for f in self.antecedent:
            lhs_items.append(str(f))
        lhs = ','.join(sorted(lhs_items))

        rhs_items = []
        for f in self.succedent:
            rhs_items.append(str(f))
        rhs = ','.join(sorted(rhs_items))

        return f"{lhs}|{rhs}"


@dataclass
class ProofNode:
    sequent: Sequent
    rule: str = ''
    premises: List['ProofNode'] = field(default_factory=list)


TT_FORALL  = 'FORALL'
TT_EXISTS  = 'EXISTS'
TT_NOT     = 'NOT'
TT_AND     = 'AND'
TT_OR      = 'OR'
TT_IMPLIES = 'IMPLIES'
TT_LPAREN  = 'LPAREN'
TT_RPAREN  = 'RPAREN'
TT_DOT     = 'DOT'
TT_COMMA   = 'COMMA'
TT_IDENT   = 'IDENT'
TT_EOF     = 'EOF'

_RE = re.compile(
    r"(?P<FORALL>FORALL\b|\u2200)|(?P<EXISTS>EXISTS\b|\u2203)"
    r"|(?P<NOT>NOT\b|\u00ac)|(?P<AND>AND\b|\u2227)|(?P<OR>OR\b|\u2228)"
    r"|(?P<IMPLIES>->|\u2192)|(?P<LPAREN>\()|(?P<RPAREN>\))"
    r"|(?P<DOT>\.)|(?P<COMMA>,)|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<SKIP>[ \t\r]+)", re.UNICODE)


class LexError(Exception):
    pass


class ParseError(Exception):
    pass


def tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        m = _RE.match(text, pos)
        if not m:
            raise LexError(f"Unexpected char {text[pos]!r} at col {pos+1}")
        kind = m.lastgroup
        value = m.group()
        pos = m.end()
        if kind != 'SKIP':
            tokens.append((kind, value))
    tokens.append((TT_EOF, '<EOF>'))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos]

    def _match(self, *tt):
        return self.tokens[self.pos][0] in tt

    def _consume(self, exp=None):
        tok = self.tokens[self.pos]
        if exp and tok[0] != exp:
            raise ParseError(f"Expected {exp!r}, got {tok[0]!r} ({tok[1]!r})")
        self.pos += 1
        return tok

    def parse_formula(self):
        return self._impl()

    def _impl(self):
        L = self._disj()
        if self._match(TT_IMPLIES):
            self._consume()
            return Implies(L, self._impl())
        return L

    def _disj(self):
        L = self._conj()
        while self._match(TT_OR):
            self._consume()
            L = Or(L, self._conj())
        return L

    def _conj(self):
        L = self._unary()
        while self._match(TT_AND):
            self._consume()
            L = And(L, self._unary())
        return L

    def _unary(self):
        if self._match(TT_NOT):
            self._consume()
            return Neg(self._unary())
        return self._quant()

    def _quant(self):
        if self._match(TT_FORALL):
            self._consume()
            _, v = self._consume(TT_IDENT)
            if self._match(TT_DOT):
                self._consume()
            return Forall(v, self._unary())
        if self._match(TT_EXISTS):
            self._consume()
            _, v = self._consume(TT_IDENT)
            if self._match(TT_DOT):
                self._consume()
            return Exists(v, self._unary())
        return self._atom()

    def _atom(self):
        if self._match(TT_LPAREN):
            self._consume()
            f = self.parse_formula()
            self._consume(TT_RPAREN)
            return f
        if self._match(TT_IDENT):
            _, name = self._consume()
            if self._match(TT_LPAREN):
                self._consume()
                args = []
                if not self._match(TT_RPAREN):
                    args.append(self._term())
                    while self._match(TT_COMMA):
                        self._consume()
                        args.append(self._term())
                self._consume(TT_RPAREN)
                return Predicate(name, args)
            return Atom(name)
        tok = self._peek()
        raise ParseError(f"Unexpected token {tok[0]!r} ({tok[1]!r})")

    def _term(self):
        _, name = self._consume(TT_IDENT)
        if self._match(TT_LPAREN):
            self._consume()
            args = []
            if not self._match(TT_RPAREN):
                args.append(self._term())
                while self._match(TT_COMMA):
                    self._consume()
                    args.append(self._term())
            self._consume(TT_RPAREN)
            return FuncTerm(name, args)
        return Var(name)


def parse(text: str) -> Any:
    tokens = tokenize(text)
    p = Parser(tokens)
    ast = p.parse_formula()
    if p._peek()[0] != TT_EOF:
        raise ParseError(f"Unexpected trailing token {p._peek()!r}")
    return ast


def _st(t, v, rep):
    if isinstance(t, Var):
        if t.name == v:
            return rep
        else:
            return t
    if isinstance(t, FuncTerm):
        new_args = []
        for a in t.args:
            new_args.append(_st(a, v, rep))
        return FuncTerm(t.name, new_args)
    return t


def subst(f, v, rep):
    if isinstance(f, Atom):
        return f

    if isinstance(f, Var):
        if f.name == v:
            return rep
        else:
            return f

    if isinstance(f, FuncTerm):
        new_args = []
        for a in f.args:
            new_args.append(_st(a, v, rep))
        return FuncTerm(f.name, new_args)

    if isinstance(f, Predicate):
        new_args = []
        for a in f.args:
            new_args.append(_st(a, v, rep))
        return Predicate(f.name, new_args)

    if isinstance(f, Neg):
        return Neg(subst(f.formula, v, rep))

    if isinstance(f, And):
        return And(subst(f.left, v, rep), subst(f.right, v, rep))

    if isinstance(f, Or):
        return Or(subst(f.left, v, rep), subst(f.right, v, rep))

    if isinstance(f, Implies):
        return Implies(subst(f.left, v, rep), subst(f.right, v, rep))

    if isinstance(f, Forall):
        if f.var == v:
            return f
        else:
            return Forall(f.var, subst(f.formula, v, rep))

    if isinstance(f, Exists):
        if f.var == v:
            return f
        else:
            return Exists(f.var, subst(f.formula, v, rep))

    return f


def collect_terms(seq: Sequent) -> List[Any]:
    terms: List[Any] = []
    seen: Set[str] = set()

    def _term(t):
        s = str(t)
        if s not in seen:
            seen.add(s)
            terms.append(t)
        if isinstance(t, FuncTerm):
            for a in t.args:
                _term(a)

    def _formula(f, bound=frozenset()):
        if isinstance(f, Predicate):
            for a in f.args:
                if not (isinstance(a, Var) and a.name in bound):
                    _term(a)
        elif isinstance(f, Neg):
            _formula(f.formula, bound)
        elif isinstance(f, (And, Or, Implies)):
            _formula(f.left, bound)
            _formula(f.right, bound)
        elif isinstance(f, (Forall, Exists)):
            _formula(f.formula, bound | {f.var})

    for f in seq.antecedent + seq.succedent:
        _formula(f)
    return terms


_fc = 0


def _fresh() -> Var:
    global _fc
    _fc += 1
    return Var(f"c{_fc}")


def _reset_fresh():
    global _fc
    _fc = 0


def _contract(seq: Sequent) -> Sequent:
    def dedup(lst):
        seen: Set[str] = set()
        out = []
        for f in lst:
            k = str(f)
            if k not in seen:
                seen.add(k)
                out.append(f)
        return out

    return Sequent(dedup(seq.antecedent), dedup(seq.succedent))


def _body_terms(formula: Any) -> Set[str]:
    """Collect string representations of all sub-terms inside a formula."""
    result: Set[str] = set()

    def _walk(f):
        if isinstance(f, Var):
            result.add(str(f))
        elif isinstance(f, FuncTerm):
            result.add(str(f))
            for a in f.args:
                _walk(a)
        elif isinstance(f, Predicate):
            for a in f.args:
                _walk(a)
        elif isinstance(f, Neg):
            _walk(f.formula)
        elif isinstance(f, (And, Or, Implies)):
            _walk(f.left)
            _walk(f.right)
        elif isinstance(f, (Forall, Exists)):
            _walk(f.formula)

    _walk(formula)
    return result


def _smart_order(candidates: List[Any], body: Any) -> List[Any]:
    body_strs = _body_terms(body)
    return ([t for t in candidates if str(t) in body_strs] +
            [t for t in candidates if str(t) not in body_strs])


def _closes_immediately(seq: Sequent) -> bool:
    """Return True if seq is closeable by id or ⊥L without any rule application."""
    ant_strs = {str(f) for f in seq.antecedent}
    for f in seq.succedent:
        if str(f) in ant_strs:
            return True
    for f in seq.antecedent:
        if isinstance(f, Atom) and f.name in ('False', 'Bot', '\u22a5'):
            return True
    return False


_improved_calls = 0


def _reset_counters():
    global _improved_calls
    _improved_calls = 0


MAX_DEPTH_IDS = 40
MAX_CALLS     = 500_000


def _improved_search(
        seq:     Sequent,
        depth:   int,
        used:    Dict[str, FrozenSet[str]],   # Algorithm 2: per-formula used terms
        visited: FrozenSet[str]               # Improvement 2: path-based loop detection
) -> Optional[ProofNode]:
    global _improved_calls
    _improved_calls += 1
    if _improved_calls > MAX_CALLS or depth <= 0:
        return None

    # IMPROVEMENT 1: contraction
    seq = _contract(seq)

    # IMPROVEMENT 2: path-based loop detection (only when quantifiers present)
    if any(isinstance(f, (Forall, Exists)) for f in seq.antecedent + seq.succedent):
        key = seq.canonical()
        if key in visited:
            return None
        visited = visited | {key}

    node = ProofNode(seq)
    ant, suc = seq.antecedent, seq.succedent

    def one(rule, new_seq):
        child = _improved_search(new_seq, depth - 1, used, visited)
        if child:
            node.rule = rule
            node.premises = [child]
            return node
        return None

    def two(rule, s1, s2):
        # IMPROVEMENT 4: if either branch closes immediately, skip full search
        # for that branch and only recurse into the non-trivial one
        c1_trivial = _closes_immediately(s1)
        c2_trivial = _closes_immediately(s2)
        if c1_trivial and c2_trivial:
            n1 = ProofNode(s1, 'id', [])
            n2 = ProofNode(s2, 'id', [])
            node.rule = rule
            node.premises = [n1, n2]
            return node
        if c1_trivial:
            c2 = _improved_search(s2, depth - 1, used, visited)
            if c2:
                n1 = ProofNode(s1, 'id', [])
                node.rule = rule
                node.premises = [n1, c2]
                return node
            return None
        if c2_trivial:
            c1 = _improved_search(s1, depth - 1, used, visited)
            if c1:
                n2 = ProofNode(s2, 'id', [])
                node.rule = rule
                node.premises = [c1, n2]
                return node
            return None
        c1 = _improved_search(s1, depth - 1, used, visited)
        c2 = _improved_search(s2, depth - 1, used, visited)
        if c1 and c2:
            node.rule = rule
            node.premises = [c1, c2]
            return node
        return None

    # --- Closing rules ---
    ant_strs = {str(f) for f in ant}
    for f in suc:
        if str(f) in ant_strs:
            node.rule = 'id'
            return node
    for f in ant:
        if isinstance(f, Atom) and f.name in ('False', 'Bot', '\u22a5'):
            node.rule = '\u22a5L'
            return node

    # --- Invertible rules (apply eagerly) ---
    for i, f in enumerate(ant):
        if isinstance(f, And):
            new_ant = ant[:i] + [f.left, f.right] + ant[i+1:]
            return one('\u2227L', Sequent(new_ant, suc[:]))

    for i, f in enumerate(suc):
        if isinstance(f, Or):
            new_suc = suc[:i] + [f.left, f.right] + suc[i+1:]
            return one('\u2228R', Sequent(ant[:], new_suc))

    for i, f in enumerate(suc):
        if isinstance(f, Implies):
            new_suc = suc[:i] + [f.right] + suc[i+1:]
            return one('\u2192R', Sequent(ant + [f.left], new_suc))

    for i, f in enumerate(ant):
        if isinstance(f, Neg):
            new_ant = ant[:i] + ant[i+1:]
            return one('\u00acL', Sequent(new_ant, suc + [f.formula]))

    for i, f in enumerate(suc):
        if isinstance(f, Neg):
            new_suc = suc[:i] + suc[i+1:]
            return one('\u00acR', Sequent(ant + [f.formula], new_suc))

    # Eigenvariable rules (fresh constant -- invertible)
    for i, f in enumerate(suc):
        if isinstance(f, Forall):
            y = _fresh()
            new_suc = suc[:i] + [subst(f.formula, f.var, y)] + suc[i+1:]
            return one('\u2200R', Sequent(ant[:], new_suc))

    for i, f in enumerate(ant):
        if isinstance(f, Exists):
            y = _fresh()
            new_ant = ant[:i] + [subst(f.formula, f.var, y)] + ant[i+1:]
            return one('\u2203L', Sequent(new_ant, suc[:]))

    # --- Branching rules ---
    for i, f in enumerate(suc):
        if isinstance(f, And):
            rest = suc[:i] + suc[i+1:]
            return two('\u2227R', Sequent(ant[:], rest + [f.left]), Sequent(ant[:], rest + [f.right]))

    for i, f in enumerate(ant):
        if isinstance(f, Or):
            rest = ant[:i] + ant[i+1:]
            return two('\u2228L', Sequent(rest + [f.left], suc[:]), Sequent(rest + [f.right], suc[:]))

    for i, f in enumerate(ant):
        if isinstance(f, Implies):
            rest = ant[:i] + ant[i+1:]
            return two('\u2192L', Sequent(rest[:], suc + [f.left]), Sequent(rest + [f.right], suc[:]))

    # --- ∀L: Algorithm 2 used-term tracking + Improvement 3 smart ordering ---
    for i, f in enumerate(ant):
        if isinstance(f, Forall):
            fkey = str(f)
            used_for_f = used.get(fkey, frozenset())
            existing = _smart_order(
                [t for t in collect_terms(seq) if str(t) not in used_for_f],
                f.formula)
            for t in existing:
                new_used = dict(used)
                new_used[fkey] = used_for_f | {str(t)}
                new_ant = ant[:i] + [subst(f.formula, f.var, t)] + ant[i+1:]
                c = _improved_search(Sequent(new_ant, suc[:]), depth - 1, new_used, visited)
                if c:
                    node.rule = '\u2200L'
                    node.premises = [c]
                    return node
            # all existing exhausted: try one fresh term
            if not existing or all(str(t) in used_for_f for t in collect_terms(seq)):
                t = _fresh()
                if str(t) not in used_for_f:
                    new_used = dict(used)
                    new_used[fkey] = used_for_f | {str(t)}
                    new_ant = ant[:i] + [subst(f.formula, f.var, t)] + ant[i+1:]
                    c = _improved_search(Sequent(new_ant, suc[:]), depth - 1, new_used, visited)
                    if c:
                        node.rule = '\u2200L'
                        node.premises = [c]
                        return node
            return None  # loop: all terms tried for this formula

    # --- ∃R: Algorithm 2 used-term tracking + Improvement 3 smart ordering ---
    for i, f in enumerate(suc):
        if isinstance(f, Exists):
            fkey = str(f)
            used_for_f = used.get(fkey, frozenset())
            existing = _smart_order(
                [t for t in collect_terms(seq) if str(t) not in used_for_f],
                f.formula)
            for t in existing:
                new_used = dict(used)
                new_used[fkey] = used_for_f | {str(t)}
                new_suc = suc[:i] + [subst(f.formula, f.var, t)] + suc[i+1:]
                c = _improved_search(Sequent(ant[:], new_suc), depth - 1, new_used, visited)
                if c:
                    node.rule = '\u2203R'
                    node.premises = [c]
                    return node
            # all existing exhausted: try one fresh term
            if not existing or all(str(t) in used_for_f for t in collect_terms(seq)):
                t = _fresh()
                if str(t) not in used_for_f:
                    new_used = dict(used)
                    new_used[fkey] = used_for_f | {str(t)}
                    new_suc = suc[:i] + [subst(f.formula, f.var, t)] + suc[i+1:]
                    c = _improved_search(Sequent(ant[:], new_suc), depth - 1, new_used, visited)
                    if c:
                        node.rule = '\u2203R'
                        node.premises = [c]
                        return node
            return None  # loop: all terms tried for this formula

    return None


def _proof_depth(node: ProofNode) -> int:
    if not node.premises:
        return 0
    return 1 + max(_proof_depth(p) for p in node.premises)


def prove(formula: Any):
    _reset_counters()
    _reset_fresh()
    result = _improved_search(Sequent([], [formula]), MAX_DEPTH_IDS, {}, frozenset())
    depth = _proof_depth(result) if result else MAX_DEPTH_IDS
    return result, _improved_calls, depth


def prove_sequent(seq: Sequent):
    """Prove an arbitrary sequent -- used by tptp_parser with antecedents."""
    _reset_counters()
    _reset_fresh()
    result = _improved_search(seq, MAX_DEPTH_IDS, {}, frozenset())
    depth = _proof_depth(result) if result else MAX_DEPTH_IDS
    return result, _improved_calls, depth


def _count_steps(node: ProofNode) -> int:
    return 1 + sum(_count_steps(p) for p in node.premises)


def main() -> None:
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'input.txt'

    try:
        lines = open(filename, encoding='utf-8').readlines()
    except FileNotFoundError:
        print(f"Error: file {filename!r} not found.")
        sys.exit(1)

    proved = 0
    failed = 0
    total_depth = 0
    total_calls = 0
    total_start = time.perf_counter()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        print('─' * 140)

        try:
            ast = parse(line)
            print(f"  Formula  :  {ast}")

            t0 = time.perf_counter()
            result, calls, depth = prove(ast)
            elapsed = (time.perf_counter() - t0) * 1000

            if result:
                steps = _count_steps(result)
                print(f"  Status   :  PROVED  ({steps} steps, depth {depth}, {calls} calls, {elapsed:.2f}ms)")
                proved += 1
                total_depth += depth
                total_calls += calls
            else:
                print(f"  Status   :  not provable as a tautology  ({calls} calls, {elapsed:.2f}ms)")
                failed += 1

        except (LexError, ParseError) as exc:
            print(f"  Status   :  Parse error - {exc}")
            failed += 1

    avg_depth = total_depth / proved if proved else 0
    avg_calls = total_calls / proved if proved else 0
    total_time = time.perf_counter() - total_start

    print(f"\n{'='*66}")
    print(f"  {proved} proved   |   {failed} not provable / errors")
    print(f"  Avg depth : {avg_depth:.1f}   |   Avg calls : {avg_calls:.1f}")
    print(f"  Time      : {total_time:.2f}s")
    print(f"{'='*66}")


if __name__ == '__main__':
    main()