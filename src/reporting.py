"""Shared plain-text report formatting without optional table dependencies."""


def markdown_table(table):
    """Turn a small DataFrame into a Markdown table without extra packages."""
    lines = [
        "| " + " | ".join(str(column) for column in table.columns) + " |",
        "|" + "|".join(["---"] * len(table.columns)) + "|",
    ]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)
