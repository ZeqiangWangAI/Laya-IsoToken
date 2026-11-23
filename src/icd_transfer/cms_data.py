import csv
import io
import re
import zipfile
from pathlib import Path

from .codebook import Code, write_codebook
from .transfer_data import save_json


def table5_rows(path, member=None):
    with zipfile.ZipFile(path) as archive:
        text_files = [name for name in archive.namelist() if name.endswith(".txt")]
        if text_files:
            text = archive.read(member or text_files[0]).decode("cp1252")
            rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
        else:
            import xlrd
            name = next(name for name in archive.namelist() if name.endswith(".xls"))
            sheet = xlrd.open_workbook(file_contents=archive.read(name)).sheet_by_index(0)
            rows = [sheet.row_values(i) for i in range(sheet.nrows)]
    return [{"code": str(row[0]).strip().zfill(3), "mdc": str(row[3]).strip(),
             "kind": str(row[4]).strip(), "description": row[5].strip()}
            for row in rows if row and re.fullmatch(r"\d{1,3}", str(row[0]).strip())]


def manual_rows(path, member, nested=None):
    with zipfile.ZipFile(path) as archive:
        if nested:
            archive = zipfile.ZipFile(io.BytesIO(archive.read(nested)))
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    return [{"code": line[:3], "mdc": line[4:6].strip(), "kind": line[8].strip(),
             "description": line[11:].strip()}
            for line in lines if re.match(r"^\d{3}\s", line)]


def prepare_cms(args):
    root, output = Path(args.input), Path(args.output)
    versions = {int(path.name[2:6]) - 1983: table5_rows(path)
                for path in sorted((root / "cms-historical").glob("FY*-*.zip"))}
    raw = root / "cms-raw"
    versions[32] = table5_rows(raw / "fy2015-fr-table-5.zip", "FY_2015_CN_Table 5.txt")
    manuals = [
        (33, "fy2016-cms-1632-fr-ms-drg.zip", "fulltext/full_appendix_A.txt",
         "ICD10_Definitions_Manual_MSDRG_v33.0_TEXT.zip"),
        (34, "fy2017-cms-1655-fr-ms-drg.zip", "appendix_A.txt", None),
        (35, "fy2018-cms-1677-fr-ms-drg-definitions-manual.zip", "appendix_A.txt", None),
        (36, "fy2019-cms-1694-fr-ms-drg-definitions-manual.zip", "appendix_A.txt", None),
        (37, "fy2020-cms-1716-fr-ms-drg-definitions-manual.zip", "appendix_A.txt",
         "MSDRGv37.0ICD10_R1_DefinitionsManual_TEXT.zip"),
    ]
    versions.update({version: manual_rows(raw / filename, member, nested)
                     for version, filename, member, nested in manuals})
    if set(versions) != set(range(25, 38)):
        raise ValueError("CMS codebooks require every MS-DRG edition from 25 to 37")
    output.mkdir(parents=True, exist_ok=False)
    for version, rows in sorted(versions.items()):
        write_cms_version(output, version, rows)
    save_json(output / "metadata.json", {
        "source": "CMS annual Table 5 and MS-DRG Definitions Manual Appendix A",
        "role": "target_public_codebooks", "versions": {str(v): len(rows) for v, rows in versions.items()},
        "fiscal_years": [2008, 2020], "model_fit_allowed": False,
        "case_edition_policy": "MIMIC has no annual grouper edition; do not infer from shifted dates",
    })


def write_cms_version(output, version, rows):
    if len({row["code"] for row in rows}) != len(rows) or not rows:
        raise ValueError(f"Invalid CMS v{version} code table")
    system = f"ms_drg_v{version}"
    codes = [Code(f"{system}:{row['code']}", system, row["code"], row["description"]) for row in rows]
    write_codebook(codes, output / f"ms-drg-v{version}.json")
    save_json(output / f"ms-drg-v{version}-structure.json", rows)
