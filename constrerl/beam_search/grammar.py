import re

from lark import ParseTree, Token, Tree
from lark import Lark

GBNF_GRAMMAR = r"""
start     : rule+
rule        : identifier "::=" expression

expression  : term ( "|" term )*
term        : factor+
counts      : "{" SIGNED_NUMBER "}"
counts_range: "{" SIGNED_NUMBER "," SIGNED_NUMBER "}"
multiplicity: "*" | "+" | "?" | counts | counts_range
factor      : primary ( multiplicity )?
primary     : identifier
              | literal
              | char_class
              | "(" expression ")"

identifier  : /[a-zA-Z_-][a-zA-Z0-9_-]*/
literal      : ESCAPED_STRING
char_class  : "[" /[^\]]/+ "]"
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
"""
GBNF_PARSER = Lark(GBNF_GRAMMAR)


class GrammarMatcher:
    def __init__(self, grammar: ParseTree):
        self.grammar = grammar
        self.rules = self._extract_rules(grammar)

    def _extract_rules(self, grammar: ParseTree):
        rules: dict[str, Tree] = {}
        for rule in grammar.children:
            rule_name = rule.children[0].children[0].value
            rule_expr = rule.children[1]
            rules[rule_name] = rule_expr
        return rules

    def test_grammar(self, test_string: str, rule_name: str):
        # recursively match the test string against the grammar rule
        rule = self.rules[rule_name]
        return self._match_rule(test_string, rule)

    def _match_literal(
        self, test_string: str, literal: str, position: int, strict: bool = False
    ):
        literal_value = literal.strip('"')
        do_strict = True
        if len(literal_value) > len(test_string[position:]):
            do_strict = False
        if do_strict:
            if test_string[position:].startswith(literal_value):
                return True, position + len(literal_value)
        else:
            if literal_value.startswith(test_string[position:]):
                return True, position + len(literal_value)
        return False, position

    def _match_char_class(self, test_string: str, char_class: str, position: int):
        char_class_value = char_class.strip("[]")
        # build regex for char class
        regex = re.compile(f"[{char_class_value}]")
        if regex.match(test_string[position]):
            return True, position + 1
        else:
            return False, position

    def _handle_multiplicity(
        self, test_string: str, primary: Tree, multiplicity: str, position: int
    ):
        expected_count_range = (0, float("inf"))
        if multiplicity == "*":
            expected_count_range = (0, float("inf"))
        elif multiplicity == "+":
            expected_count_range = (1, float("inf"))
        elif multiplicity == "?":
            expected_count_range = (0, 1)
        elif multiplicity.startswith("{") and multiplicity.endswith("}"):
            counts = multiplicity.strip("{}").split(",")
            if len(counts) == 1:
                expected_count_range = (int(counts[0]), int(counts[0]))
            elif len(counts) == 2:
                expected_count_range = (int(counts[0]), int(counts[1]))
        count = 0
        current_pos = position
        while count < expected_count_range[1]:
            success, new_pos = self._match_rule(test_string, primary, current_pos)
            if not success:
                break
            count += 1
            current_pos = new_pos
        if expected_count_range[0] <= count <= expected_count_range[1]:
            return True, current_pos
        else:
            return False, position

    def _match_rule(self, test_string: str, rule: Tree, position: int = 0):
        # match the rule against the test string starting from the given position
        # return a tuple of (match_successful: bool, new_position: int)
        if position >= len(test_string):
            return True, position
        if rule.data == "expression":
            for term in rule.children:
                success, new_pos = self._match_rule(test_string, term, position)
                if success:
                    return True, new_pos
            return False, position
        elif rule.data == "term":
            current_pos = position
            for factor in rule.children:
                success, new_pos = self._match_rule(test_string, factor, current_pos)
                if not success:
                    return False, position
                current_pos = new_pos
            return True, current_pos
        elif rule.data == "factor":
            primary = rule.children[0]
            multiplicity = rule.children[1] if len(rule.children) > 1 else None
            success, new_pos = self._match_rule(test_string, primary, position)
            if not success:
                return False, position
            if multiplicity is None or len(multiplicity.children) == 0:
                return success, new_pos
            return self._handle_multiplicity(
                test_string, primary, multiplicity.children[0].value, new_pos
            )
        elif rule.data == "primary":
            # it's an identifier or literal or char_class
            token = rule.children[0]
            typ = token.data.value if isinstance(token, Tree) else token.type
            value = (
                token.children[0].value
                if isinstance(token.children[0], Token)
                else None
            )
            if typ == "identifier":
                # match the identifier by looking up the corresponding rule
                return self._match_rule(test_string, self.rules[value], position)
            elif typ == "literal":
                return self._match_literal(test_string, value, position)
            elif typ == "char_class":
                value = "".join(child.value for child in token.children)
                return self._match_char_class(test_string, value, position)
            elif typ == "expression":
                return self._match_rule(test_string, token, position)
        else:
            raise ValueError(f"Unknown rule type: {rule.data}")
        return False, position


def test_against_grammar(
    test_string: str, str_grammar: ParseTree, root_rule_name: str = "root"
):
    # greedily match the test string against the grammar, starting from the root rule
    # if the match does not find a partial match, return False
    # if the match finds a partial match, return True and the remaining string
    # if the match finds a full match, return True and an empty string
    # the match should prioritize longer matches over shorter matches
    matcher = GrammarMatcher(str_grammar)
    # start matching from the root rule
    return matcher.test_grammar(test_string, root_rule_name)
