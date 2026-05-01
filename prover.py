import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set


# AST Nodes

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
        return self.name + "(" + joined_args + ")"


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
            return self.name + "(" + joined_args + ")"
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


@dataclass
class Sequent:
    antecedent: List[Any]   # formulas on the LEFT  of |-
    succedent:  List[Any]   # formulas on the RIGHT of |-

    def __str__(self) -> str:
        lhs_parts = []
        for f in self.antecedent:
            lhs_parts.append(str(f))
        lhs = ', '.join(lhs_parts)

        rhs_parts = []
        for f in self.succedent:
            rhs_parts.append(str(f))
        rhs = ', '.join(rhs_parts)

        if lhs:
            return lhs + " \u22a2 " + rhs
        else:
            return "\u22a2 " + rhs


# Lexer

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
    r"|(?P<SKIP>[ \t\r]+)",
    re.UNICODE)


class LexError(Exception):
    pass


class ParseError(Exception):
    pass


def tokenize(text: str) -> list:
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


# Recursive-Descent Parser
# Precedence (low -> high):  ->  |  \/  |  /\  |  NOT  |  Qx.  |  atom

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


# Proof-tree node

@dataclass
class ProofNode:
    sequent:  Sequent
    rule:     str               = ''
    premises: List['ProofNode'] = field(default_factory=list)


# Substitution

def _st(t: Any, v: str, rep: Any) -> Any:
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


def subst(f: Any, v: str, rep: Any) -> Any:
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
    seen: set = set()

    def _term(t: Any):
        s = str(t)
        if s not in seen:
            seen.add(s)
            terms.append(t)
        if isinstance(t, FuncTerm):
            for a in t.args:
                _term(a)

    def _formula(f: Any, bound: frozenset = frozenset()):
        if isinstance(f, Predicate):
            for a in f.args:
                if not (isinstance(a, Var) and a.name in bound):
                    _term(a)
        elif isinstance(f, Neg):
            _formula(f.formula, bound)
        elif isinstance(f, (And, Or, Implies)):
            _formula(f.left, bound)
            _formula(f.right, bound)
        elif isinstance(f, Forall):
            _formula(f.formula, bound | {f.var})
        elif isinstance(f, Exists):
            _formula(f.formula, bound | {f.var})

    for f in seq.antecedent + seq.succedent:
        _formula(f)
    return terms


# Fresh constant generator

_fc: int = 0


def _fresh() -> Var:
    global _fc
    _fc += 1
    return Var(f"t{_fc}")


def _reset_fresh() -> None:
    global _fc
    _fc = 0


# Backward Proof Search

MAX_DEPTH = 40
MAX_CALLS = 500_000   # abort if search exceeds this many calls

_calls: int = 0


def _contract(seq: Sequent) -> Sequent:
    """Remove duplicate formulas from each side (contraction)."""
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


def _reset_calls() -> None:
    global _calls
    _calls = 0


def _search(
        seq:   Sequent,
        depth: int,
        used:  dict = None   # maps formula_str -> frozenset of used term strs
) -> Optional[ProofNode]:
    global _calls
    _calls += 1
    if _calls > MAX_CALLS or depth <= 0:
        return None

    # Contraction: remove duplicates before processing
    seq = _contract(seq)

    if used is None:
        used = {}

    node = ProofNode(seq)
    ant, suc = seq.antecedent, seq.succedent

    def one(rule: str, new_seq: Sequent) -> Optional[ProofNode]:
        child = _search(new_seq, depth - 1, used)
        if child:
            node.rule = rule
            node.premises = [child]
            return node
        return None

    def two(rule: str, s1: Sequent, s2: Sequent) -> Optional[ProofNode]:
        c1 = _search(s1, depth - 1, used)
        c2 = _search(s2, depth - 1, used)
        if c1 and c2:
            node.rule = rule
            node.premises = [c1, c2]
            return node
        return None

    # Closing rules

    # id: same formula on both sides
    ant_strs = set()
    for f in ant:
        ant_strs.add(str(f))

    for f in suc:
        if str(f) in ant_strs:
            node.rule = "id"
            return node

    # ⊥L: False / bottom on the left closes any branch
    for f in ant:
        if isinstance(f, Atom):
            if f.name in ("False", "Bot", "⊥"):
                node.rule = "⊥L"
                return node

    # Linear invertible rules (apply the first one found)

    # AndL
    for i, f in enumerate(ant):
        if isinstance(f, And):
            new_ant = ant[:i] + [f.left, f.right] + ant[i+1:]
            return one('∧L', Sequent(new_ant, suc[:]))

    # OrR
    for i, f in enumerate(suc):
        if isinstance(f, Or):
            new_suc = suc[:i] + [f.left, f.right] + suc[i+1:]
            return one('VR', Sequent(ant[:], new_suc))

    # ImpR
    for i, f in enumerate(suc):
        if isinstance(f, Implies):
            new_suc = suc[:i] + [f.right] + suc[i+1:]
            return one('->R', Sequent(ant + [f.left], new_suc))

    # NegL
    for i, f in enumerate(ant):
        if isinstance(f, Neg):
            new_ant = ant[:i] + ant[i+1:]
            return one('¬L', Sequent(new_ant, suc + [f.formula]))

    # NegR
    for i, f in enumerate(suc):
        if isinstance(f, Neg):
            new_suc = suc[:i] + suc[i+1:]
            return one('¬R', Sequent(ant + [f.formula], new_suc))

    # ForallR
    for i, f in enumerate(suc):
        if isinstance(f, Forall):
            y = _fresh()
            new_suc = suc[:i] + [subst(f.formula, f.var, y)] + suc[i+1:]
            return one('∀R', Sequent(ant[:], new_suc))

    # ExistsL
    for i, f in enumerate(ant):
        if isinstance(f, Exists):
            y = _fresh()
            new_ant = ant[:i] + [subst(f.formula, f.var, y)] + ant[i+1:]
            return one('∃L', Sequent(new_ant, suc[:]))

    # Branching rules

    # AndR
    for i, f in enumerate(suc):
        if isinstance(f, And):
            rest = suc[:i] + suc[i+1:]
            return two('∧R', Sequent(ant[:], rest + [f.left]), Sequent(ant[:], rest + [f.right]))

    # OrL
    for i, f in enumerate(ant):
        if isinstance(f, Or):
            rest = ant[:i] + ant[i+1:]
            return two('VL', Sequent(rest + [f.left], suc[:]), Sequent(rest + [f.right], suc[:]))

    # ImplyL
    for i, f in enumerate(ant):
        if isinstance(f, Implies):
            rest = ant[:i] + ant[i+1:]
            return two('->L', Sequent(rest[:], suc + [f.left]), Sequent(rest + [f.right], suc[:]))

    # ∀L -- Algorithm 2: try unused existing terms first, then fresh
    for i, f in enumerate(ant):
        if isinstance(f, Forall):
            fkey = str(f)
            used_for_f = used.get(fkey, frozenset())

            # Step 1: Try existing terms first
            all_terms = collect_terms(seq)
            unused_terms = []
            for t in all_terms:
                if str(t) not in used_for_f:
                    unused_terms.append(t)

            for t in unused_terms:
                new_used = dict(used)
                new_used[fkey] = used_for_f | {str(t)}
                instantiated = subst(f.formula, f.var, t)
                new_ant = list(ant)
                new_ant[i] = instantiated
                child = _search(Sequent(new_ant, list(suc)), depth - 1, new_used)
                if child is not None:
                    node.rule = "∀L"
                    node.premises = [child]
                    return node

            # Step 2: If all existing terms exhausted, try a fresh one
            all_terms_used = True
            for t in all_terms:
                if str(t) not in used_for_f:
                    all_terms_used = False
                    break

            if not used_for_f or all_terms_used:
                t = _fresh()
                if str(t) not in used_for_f:
                    new_used = dict(used)
                    new_used[fkey] = used_for_f | {str(t)}
                    instantiated = subst(f.formula, f.var, t)
                    new_ant = list(ant)
                    new_ant[i] = instantiated
                    child = _search(Sequent(new_ant, list(suc)), depth - 1, new_used)
                    if child is not None:
                        node.rule = "∀L"
                        node.premises = [child]
                        return node

            # Step 3: No progress possible → stop this branch
            return None  # loop detected / all terms exhausted

    # ∃R -- Algorithm 2: try unused existing terms first, then fresh
    for i, f in enumerate(suc):
        if isinstance(f, Exists):
            fkey = str(f)
            used_for_f = used.get(fkey, frozenset())

            # Collect all terms in the current sequent
            all_terms = collect_terms(seq)

            # Filter terms we haven't tried yet
            unused_terms = []
            for t in all_terms:
                if str(t) not in used_for_f:
                    unused_terms.append(t)

            # Try each unused term
            for t in unused_terms:
                new_used = dict(used)
                updated_used_set = set(used_for_f)
                updated_used_set.add(str(t))
                new_used[fkey] = updated_used_set
                instantiated_formula = subst(f.formula, f.var, t)
                new_suc = list(suc)
                new_suc[i] = instantiated_formula
                child = _search(Sequent(list(ant), new_suc), depth - 1, new_used)
                if child is not None:
                    node.rule = "∃R"
                    node.premises = [child]
                    return node

            # All existing terms exhausted: introduce a fresh term
            existing_terms = collect_terms(seq)
            all_terms_used = True
            for t in existing_terms:
                if str(t) not in used_for_f:
                    all_terms_used = False
                    break

            if not used_for_f or all_terms_used:
                t = _fresh()
                if str(t) not in used_for_f:
                    new_used = dict(used)
                    new_used[fkey] = used_for_f | {str(t)}
                    new_suc = suc[:i] + [subst(f.formula, f.var, t)] + suc[i+1:]
                    c = _search(Sequent(ant[:], new_suc), depth - 1, new_used)
                    if c is not None:
                        node.rule = "∃R"
                        node.premises = [c]
                        return node

            return None  # loop detected: all terms tried for this formula

    return None  # no rule applicable


def _proof_depth(node: ProofNode) -> int:
    if not node.premises:
        return 0
    return 1 + max(_proof_depth(p) for p in node.premises)


def prove(formula: Any):
    """Attempt to find a closed LK derivation for  |- formula."""
    _reset_fresh()
    _reset_calls()
    result = _search(Sequent([], [formula]), MAX_DEPTH)
    depth = _proof_depth(result) if result else MAX_DEPTH
    return result, _calls, depth


def prove_sequent(seq: Sequent):
    """Prove an arbitrary sequent (used by tptp_parser with antecedents)."""
    _reset_fresh()
    _reset_calls()
    result = _search(seq, MAX_DEPTH)
    depth = _proof_depth(result) if result else MAX_DEPTH
    return result, _calls, depth


def _count_steps(node: ProofNode) -> int:
    total = 1
    for premise in node.premises:
        total += _count_steps(premise)
    return total


def main() -> None:
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "input.txt"

    try:
        lines = open(filename, encoding='utf-8').readlines()
    except FileNotFoundError:
        print(f"Error: file {filename} not found.")
        sys.exit(1)

    proved = 0
    failed = 0
    total_depth = 0
    total_calls = 0
    total_start = time.perf_counter()

    for _, raw in enumerate(lines, 1):
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