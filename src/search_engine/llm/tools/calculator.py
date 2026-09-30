import ast
import operator

from langchain_core.tools import BaseTool, tool

from search_engine.llm.context import AnswerContext

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POWER = 100  # keeps "9 ** 9 ** 9" from hanging the request


def build(context: AnswerContext) -> BaseTool:
    @tool
    def calculator(expression: str) -> str:
        """Work out an arithmetic expression exactly, e.g. "143 - 78" or "(120 + 95) / 2" or "12 / 100 * 143".
        Use it for every sum, difference, percentage or average instead of calculating in your head. Supports
        + - * / // % ** and parentheses, with numbers taken from the context passages."""
        try:
            return str(evaluate(expression))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as error:
            return f"Could not calculate {expression!r}: {error}"

    return calculator


def evaluate(expression: str) -> int | float:
    """Arithmetic only: the expression is parsed, never executed, so it cannot run code."""
    result = _evaluate(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval").body)
    return round(result, 10) if isinstance(result, float) else result


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")
