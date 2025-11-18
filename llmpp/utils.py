import json
import logging
import re
from tempfile import NamedTemporaryFile

DEFAULT_LOGGER_NAME = "llmpp"

PTB_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


BRACKET_ESCAPE_MAPPING = {
    "(": "-LRB-",
    ")": "-RRB-"
}


def escape_brackets(target: str) -> str:
    for c, r in BRACKET_ESCAPE_MAPPING.items():
        target = target.replace(c, r)
    return target


def create_logger(logger_name=DEFAULT_LOGGER_NAME, log_file_path=None, is_dummy=False):
    if is_dummy:
        logger = logging.getLogger(f"{logger_name}-dummy")
        handler = logging.NullHandler()
        logger.addHandler(handler)
        return logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file_path:
        handler = logging.FileHandler(log_file_path, mode="w", encoding="utf8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def create_system_role_replaced_tempfiles(jsonl_path):
    with open(jsonl_path, "r", encoding="utf8") as fin:
        train_dataset = [json.loads(_) for _ in fin]
    with NamedTemporaryFile("w", encoding="utf8", newline="\n", delete=False) as fout:
        for record in train_dataset:
            new_messages = []
            system_content = ""
            for m in record["messages"]:
                if m["role"] == "system":
                    system_content += m["content"] + "\n"
                    continue
                if system_content:
                    assert m["role"] == "user"
                    m["content"] = system_content + m["content"]
                    system_content = ""
                new_messages.append(m)
            assert not system_content
            json.dump({"messages": new_messages}, fout, ensure_ascii=False)
            print(file=fout)
        return fout.name


def select_last_bracketing_line(content: str) -> str | None:
    for line in reversed(content.strip().split("\n")):
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            return line
    else:
        return ""


def select_tsv_part(content: str, tsv_index: int = -1, min_columns: int = 3) -> list[list[str]]:
    rows = [line.split("\t") for line in content.split("\n")]
    begin_of_tsv = None
    tsv_parts = []
    for _, r in enumerate(rows):
        if len(r) < min_columns:
            if begin_of_tsv is not None:
                tsv_parts.append(rows[begin_of_tsv:_])
                begin_of_tsv = None
        elif begin_of_tsv is None:
            begin_of_tsv = _
    if begin_of_tsv is not None:
        tsv_parts.append(rows[begin_of_tsv:])
    if 0 <= tsv_index < len(tsv_parts) or -len(tsv_parts) <= tsv_index < 0:
        return tsv_parts[tsv_index]
    else:
        return None


def list_tsv_blocks(messages: list[dict[str, str]], target_field: str, use_first_user_turn: bool = True):
    tsv_blocks = []
    first_user_turn = use_first_user_turn
    for message in messages:
        if first_user_turn and message["role"] == "user":
            tsv_block = select_tsv_part(message["content"])
            if tsv_block:
                tsv_blocks.append(tsv_block)
            first_user_turn = False
        elif message["role"] == "assistant":
            tsv_block = []
            for line in message[target_field].split("\n"):
                if "\t" in line:
                    tsv_block.append(line.split("\t"))
                elif tsv_block:
                    tsv_blocks.append(tsv_block)
                    tsv_block = []
            if tsv_block:
                tsv_blocks.append(tsv_block)
    return tsv_blocks


def yield_constituent_units(text):
    unit = ""
    for c in text:
        if c in " \n\r\t":
            if unit:
                yield unit
                unit = ""
        elif c in "()":
            if unit:
                yield unit
                unit = ""
            yield c
        else:
            unit += c
    assert not unit, f"invalid tree: {text}"


def constituent_tree(units: list[str], no_terminal=False) -> list[str | list]:
    assert units.pop(0) == "(", f"bad sequence: {units}"
    constituent = [units.pop(0)]
    assert constituent[0] not in ["(", ")"] , f"bad sequence: {units}"
    while units[0] != ")":
        if units[0] == "(":
            units, subtree = constituent_tree(units, no_terminal=no_terminal)
            constituent.append(subtree)
        else:
            word = units.pop(0)
            if len(constituent) > 1 and isinstance(constituent[-1], str):
                constituent[-1] += " " + word
            elif no_terminal:
                constituent.append([word])
            else:
                constituent.append(word)
    return units[1:], constituent


def constituent_trees(units: list[str], no_terminal=False):
    while units:
        assert units[0] == "("
        units, tree = constituent_tree(units[1:], no_terminal=no_terminal)
        assert units[0] == ")"
        units = units[1:]
        yield tree


def tree_to_text(tree: list[str|list], add_root: bool = False, opener="(", closer=")") -> str:
    bracketed = f"{opener}{' '.join(_ if isinstance(_, str) else tree_to_text(_) for _ in tree)}{closer}"
    if add_root:
        return f"( {bracketed})"
    else:
        return bracketed


def recover_bracketing(text: str) -> str:
    balance = []
    balanced_count = 0
    b = 0
    for i, c in enumerate(text):
        if c == "(":
            b += 1
        elif c == ")":
            b -= 1
        balance.append(b)
        if b <= 0:
            balanced_count += 1
    if b == 0 and balanced_count == 1:
        return text
    skipped = 0
    recovered = ""
    for i, b in enumerate(balance):
        if b + skipped > 0:
            recovered += text[i]
        else:
            skipped += 1
    if balance and balance[-1] + skipped > 0:
        recovered += ")" * (balance[-1] + skipped)
    return recovered


def recover_word(gold_text: str, pred_text: str) -> str:
    gold_words = [m.group(1) for m in re.finditer(r"(?= )([^()]+)(?=\))", gold_text)]
    pred_words = [(m.start(1), m.end(1)) for m in re.finditer(r"(?= )([^()]+)(?=\))", pred_text)]
    if len(gold_words) != len(pred_words):
        return pred_text
    recovered = ""
    prev_end = 0
    for g_word, (p_start, p_end) in zip(gold_words, pred_words):
        recovered += pred_text[prev_end:p_start] + g_word
        prev_end = p_end
    recovered += pred_text[prev_end:]
    return recovered
