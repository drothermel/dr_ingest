import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import duckdb
    from dr_ingest.motherduck.connector import open_motherduck_connection

    DATASET = "https://raw.githubusercontent.com/andkret/MotherDuck-DuckDB-Course/main/311_Elevator_Service_Requests_.csv"
    return DATASET, duckdb, mo, open_motherduck_connection


@app.cell
def _(open_motherduck_connection):
    conn = open_motherduck_connection()
    conn
    return


@app.cell
def _(duckdb):
    def get_tables():
        return [r[0] for r in duckdb.execute("SHOW TABLES").fetchall()]
    return (get_tables,)


@app.cell
def _(duckdb):
    def get_dbs():
        return [r[0] for r in duckdb.execute("SHOW DATABASES").fetchall()]
    return (get_dbs,)


@app.cell
def _(DATASET, duckdb, get_tables):
    if "elevator_requests" not in get_tables():
        duckdb.execute(f"""
            CREATE VIEW elevator_requests AS
            SELECT *
            FROM read_csv_auto("{DATASET}", HEADER=True);
        """)
    duckdb.sql("SHOW TABLES")
    return


@app.cell
def _(duckdb, elevator_requests):
    duckdb.execute(f"""
        SELECT
          strftime('%Y-%m', "Created Date") as month,
          COUNT(*) AS complaints
        FROM elevator_requests
        WHERE "Borough" = 'MANHATTAN'
        GROUP BY month
        ORDER BY month;
    """).df()
    return


@app.cell
def _(duckdb, get_dbs):
    if "sample_data" not in get_dbs():
        duckdb.sql("ATTACH 'md:_share/sample_data/23b0d623-1361-421d-ae77-62d701d471e6'")
    return


@app.cell
def _(duckdb):
    duckdb.sql("""
        SELECT
            strftime('%Y', created_date) as year,
            COUNT(*) AS complaints
        FROM sample_data.nyc.service_requests
        WHERE 
            complaint_type ILIKE 'elevator' AND 
            borough = 'MANHATTAN'
        GROUP BY year
        ORDER BY year;
    """)
    return


@app.cell
def _(mo):
    the_plan = mo.sql(
        f"""
        EXPLAIN ANALYZE
            SELECT
                strftime('%Y', created_date) as year,
                COUNT(*) AS complaints
            FROM sample_data.nyc.service_requests
            WHERE 
                complaint_type ILIKE 'elevator' AND 
                borough = 'MANHATTAN'
            GROUP BY year
            ORDER BY year;
        """,
        output=False,
    )
    print(the_plan["explain_value"][0])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
