# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "attrs==25.3.0",
#     "cattrs==25.2.0",
#     "clumper==0.2.15",
#     "marimo",
#     "psycopg2==2.9.10",
#     "psycopg2-binary==2.9.10",
#     "rich==14.1.0",
#     "sqlalchemy==2.0.43",
#     "sqlalchemy-utils==0.42.0",
#     "srsly==2.5.1",
# ]
# ///

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="columns", app_title="Extract DD Try 1")


@app.cell(column=0)
def _():
    # Keeping: Might be useful if we ever return to instance extraction
    from pathlib import Path

    import marimo as mo
    import srsly
    from cattrs import structure

    from dr_ingest.qa import ensure_extracted, list_tarballs
    from dr_ingest.qa.schemas import (
        ModelAnswerOutput,
        QuestionOutputData,
        TaskOutputData,
    )
    from dr_ingest.qa.transform import (
        build_file_metadata,
        extract_question_payloads,
        model_output_keys,
        preview_agg_metrics,
    )

    return (
        Path,
        ModelAnswerOutput,
        QuestionOutputData,
        TaskOutputData,
        ensure_extracted,
        list_tarballs,
        build_file_metadata,
        extract_question_payloads,
        preview_agg_metrics,
        model_output_keys,
        mo,
        srsly,
        structure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _(Path):
    repo_dir = "/Users/daniellerothermel/drotherm/repos/datadec/"
    raw_downloads_dir = Path(repo_dir, "data", "raw_downloads")
    instance_ds_snapshots = Path(
        raw_downloads_dir,
        "datasets--allenai--DataDecide-eval-instances",
        "snapshots",
        "23f3b2e186ca6c39026e3efa00e4af397680c075",
        "models",
    )
    extracted_dir = Path(raw_downloads_dir, "instances_extracted")
    return extracted_dir, instance_ds_snapshots


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""## Extraction""")
    return


@app.cell(hide_code=True)
def _(instance_ds_snapshots, list_tarballs):
    data = "c4"
    params = "4M"
    seed = 2
    c4_4m_2_tard_files = list_tarballs(instance_ds_snapshots, data, params, seed)
    print(
        f"Num tar'd: {len(c4_4m_2_tard_files)}, first example: {c4_4m_2_tard_files[0].parts[-4:]}"
    )
    return c4_4m_2_tard_files, data, params, seed


@app.cell(hide_code=True)
def _(
    Path,
    c4_4m_2_tard_files,
    data,
    ensure_extracted,
    extracted_dir,
    params,
    seed,
):
    newly_extracted = ensure_extracted(
        c4_4m_2_tard_files[0], Path(extracted_dir, f"{data}_{params}_{seed}")
    )
    print(f"Newly extracted: {newly_extracted.parts[-2:]}")
    return (newly_extracted,)


@app.cell(hide_code=True)
def _(newly_extracted):
    extracted_files = list(newly_extracted.glob("*"))
    print("First 5 extracted:")
    for f in extracted_files[:5]:
        print(f.parts[-2:])
    return (extracted_files,)


@app.cell
def _(QuestionOutputData, parsed_per_answer, structure):
    ppa_structured = [structure(ppa, QuestionOutputData) for ppa in parsed_per_answer]
    return (ppa_structured,)


@app.cell
def _(ppa_structured):
    ppa_structured[0]
    return


@app.cell
def _(TaskOutputData, full_file_info, parsed_per_answer, structure):
    taskout_structured = structure(
        {**full_file_info[0], "question_outputs": parsed_per_answer},
        TaskOutputData,
    )
    return (taskout_structured,)


@app.cell
def _(taskout_structured):
    taskout_structured
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""## Loading""")
    return


@app.cell(hide_code=True)
def _(extracted_files):
    curr_file = extracted_files[0]
    print(f">> Current file: {curr_file.parts[-3:]}")
    curr_step = int(curr_file.parts[-2].split("-")[-1])
    curr_task = curr_file.stem
    print(f">> {curr_task=} {curr_step=}")
    return curr_file, curr_step, curr_task


@app.cell(hide_code=True)
def _(curr_file, srsly):
    curr_jsonl = list(srsly.read_jsonl(curr_file))
    print(f"Num entries: {len(curr_jsonl)}")
    return (curr_jsonl,)


@app.cell(hide_code=True)
def _(build_file_metadata, curr_jsonl, curr_step, curr_task, data, params, seed):
    full_file_info = build_file_metadata(
        curr_jsonl,
        data=data,
        params=params,
        seed=seed,
        task=curr_task,
        step=curr_step,
    )
    print("Full File Info", full_file_info)
    return (full_file_info,)


@app.cell(hide_code=True)
def _(curr_jsonl, preview_agg_metrics):
    # Get the agg metrics
    agg_metrics = preview_agg_metrics(curr_jsonl)
    if agg_metrics:
        print("Agg Metrics", agg_metrics[0])
    return


@app.cell(hide_code=True)
def _(curr_jsonl, extract_question_payloads):
    parsed_per_answer = extract_question_payloads(curr_jsonl)
    if parsed_per_answer:
        print("Parsed example", parsed_per_answer[0])
    return (parsed_per_answer,)


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(extract_question_payloads, curr_jsonl, model_output_keys):
    # Get the model output keys
    keys = model_output_keys(extract_question_payloads(curr_jsonl))
    print(">> Ordered Model Pred Keys")
    for k in keys:
        print(f"   - {k}")
    return


@app.cell
def _():
    return


@app.cell(column=3)
def _():
    return


if __name__ == "__main__":
    app.run()
