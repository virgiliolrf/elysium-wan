#!/usr/bin/env python3
"""Generate first-order mutants of Python source, restricted to given lines.

    python3 agents-architetura/tools/gates/mutate.py <file> <line,line,...>   ->   JSON list of mutants on stdout

Each mutant: {"line", "col", "op", "source"} where source is the whole mutated file. Operators are the
classic set: comparison flips and boundary shifts, arithmetic swaps, boolean connective swaps, negation
removal, boolean constant flips, integer constant +1, and return-value replacement by None.
Stdlib only, so it runs under any Python >= 3.9 without the project's environment.
"""
import ast
import copy
import json
import sys

CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
       ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn, ast.NotIn: ast.In}
CMP_NEG = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt}
BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Mult,
       ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult, ast.BitAnd: ast.BitOr, ast.BitOr: ast.BitAnd, ast.LShift: ast.RShift,
       ast.RShift: ast.LShift}
BOOL = {ast.And: ast.Or, ast.Or: ast.And}


def mutations(node):
    """Yield (op_description, apply) pairs; apply(node) mutates a deep copy in place."""
    if isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            for table in (CMP, CMP_NEG):
                new = table.get(type(op))
                if new:
                    yield f'{type(op).__name__}->{new.__name__}', (lambda n, i=i, new=new: n.ops.__setitem__(i, new()))
    elif isinstance(node, (ast.BinOp, ast.AugAssign)):
        new = BIN.get(type(node.op))
        if new:
            yield f'{type(node.op).__name__}->{new.__name__}', (lambda n, new=new: setattr(n, 'op', new()))
    elif isinstance(node, ast.BoolOp):
        new = BOOL[type(node.op)]
        yield f'{type(node.op).__name__}->{new.__name__}', (lambda n, new=new: setattr(n, 'op', new()))
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield 'remove-not', None  # handled by the parent replacement below
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            yield f'{node.value}->{not node.value}', (lambda n: setattr(n, 'value', not n.value))
        elif isinstance(node.value, int):
            yield f'{node.value}->{node.value + 1}', (lambda n: setattr(n, 'value', n.value + 1))
    elif isinstance(node, ast.Return) and node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
        yield 'return->None', (lambda n: setattr(n, 'value', ast.Constant(value=None)))


def path_to(tree, target):
    """Index path from tree to target, so the same node can be found in a deep copy."""
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is target:
                return parent, field, None
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if item is target:
                        return parent, field, i
    return None


def find_same(tree_copy, tree, node):
    originals = list(ast.walk(tree))
    copies = list(ast.walk(tree_copy))
    return copies[originals.index(node)]


def main():
    path, lines = sys.argv[1], {int(x) for x in sys.argv[2].split(',') if x}
    with open(path, encoding='utf-8') as fh:
        source = fh.read()
    tree = ast.parse(source)
    out = []
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', None)
        if line not in lines:
            continue
        for desc, apply in mutations(node):
            mutant = copy.deepcopy(tree)
            twin = find_same(mutant, tree, node)
            if apply is None:  # remove-not: replace `not x` by `x` in its parent
                loc = path_to(mutant, twin)
                if loc is None:
                    continue
                parent, field, idx = loc
                if idx is None:
                    setattr(parent, field, twin.operand)
                else:
                    getattr(parent, field)[idx] = twin.operand
            else:
                apply(twin)
            try:
                text = ast.unparse(ast.fix_missing_locations(mutant))
            except Exception:  # an operator that cannot be rendered is skipped, not fatal
                continue
            out.append({'line': line, 'col': node.col_offset, 'op': desc, 'source': text + '\n'})
    json.dump(out, sys.stdout)


if __name__ == '__main__':
    main()
