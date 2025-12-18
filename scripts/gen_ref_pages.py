"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# 扫描 DASMatrix 目录
src = Path("DASMatrix")

# 添加静态页面到导航
nav["首页"] = "index.md"
nav["快速开始"] = "quickstart.md"
nav["贡献指南"] = "contributing.md"
nav["API 文档", "概述"] = "api/index.md"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # 排除私有模块和测试（如果有）
    if any(part.startswith("_") for part in parts):
        continue

    nav[("API 文档", "API 参考") + parts] = full_doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
