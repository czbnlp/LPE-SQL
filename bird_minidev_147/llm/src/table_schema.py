import sqlite3
import glob
import os
import re
import pandas as pd
db_table_map = {
    "debit_card_specializing": [
        "customers",
        "gasstations",
        "products",
        "transactions_1k",
        "yearmonth",
    ],
    "student_club": [
        "major",
        "member",
        "attendance",
        "budget",
        "event",
        "expense",
        "income",
        "zip_code",
    ],
    "thrombosis_prediction": ["Patient", "Examination", "Laboratory"],
    "european_football_2": [
        "League",
        "Match",
        "Player",
        "Player_Attributes",
        "Team",
        "Team_Attributes",
    ],
    "formula_1": [
        "circuits",
        "seasons",
        "races",
        "constructors",
        "constructorResults",
        "constructorStandings",
        "drivers",
        "driverStandings",
        "lapTimes",
        "pitStops",
        "qualifying",
        "status",
        "results",
    ],
    "superhero": [
        "alignment",
        "attribute",
        "colour",
        "gender",
        "publisher",
        "race",
        "superpower",
        "superhero",
        "hero_attribute",
        "hero_power",
    ],
    "codebase_community": [
        "posts",
        "users",
        "badges",
        "comments",
        "postHistory",
        "postLinks",
        "tags",
        "votes",
    ],
    "card_games": [
        "cards",
        "foreign_data",
        "legalities",
        "rulings",
        "set_translations",
        "sets",
    ],
    "toxicology": ["molecule", "atom", "bond", "connected"],
    "california_schools": ["satscores", "frpm", "schools"],
    "financial": [
        "district",
        "account",
        "client",
        "disp",
        "card",
        "loan",
        "order",
        "trans",
    ],
}


def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [
        max(len(str(value[i])) for value in values + [column_names])
        for i in range(len(column_names))
    ]

    # Print the column names
    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )
    # print(header)
    # Print the values
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + "\n" + rows
    return final_output


def generate_schema_prompt_sqlite(db_path, num_rows=None):
    # extract create ddls
    """
    :param root_place:
    :param db_name:
    :return:
    """
    full_schema_prompt_list = []
    conn = sqlite3.connect(db_path)
    # Create a cursor object
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schemas = {}
    for table in tables:
        if table == "sqlite_sequence":
            continue
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
                table[0]
            )
        )
        create_prompt = cursor.fetchone()[0]
        schemas[table[0]] = create_prompt
        if num_rows:
            cur_table = table[0]
            if cur_table in ["order", "by", "group"]:
                cur_table = "`{}`".format(cur_table)

            cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
            column_names = [description[0] for description in cursor.description]
            values = cursor.fetchall()
            rows_prompt = nice_look_table(column_names=column_names, values=values)
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                num_rows, cur_table, num_rows, rows_prompt
            )
            schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

    for k, v in schemas.items():
        full_schema_prompt_list.append(v)

    schema_prompt = "\n\n".join(full_schema_prompt_list)

    return schema_prompt

from langchain.sql_database import SQLDatabase
def get_database_schema(DB_URI: str) -> str:
    """Get the database schema from the database URI

    Args:
        DB_URI (str): Database URI

    Returns:
        str: Database schema
    """
    db = SQLDatabase.from_uri("sqlite:///"+DB_URI)
    db._sample_rows_in_table_info = 3
    return db.get_table_info_no_throw()

def table_descriptions_parser(database_dir):
    csv_files = glob.glob(f"{database_dir}/*.csv")
    # Iterate over the CSV files
    db_descriptions = ""
    for file_path in csv_files:
        table_name: str = os.path.basename(file_path).replace(".csv", "")
        db_descriptions += f"Table: {table_name}\n"
        table_df = pd.read_csv(file_path, encoding='latin-1')
        for _,row in table_df.iterrows():
           try:
                if pd.notna(row.iloc[2]):
                    col_description = re.sub(r'\s+', ' ', str(row.iloc[2]))  # noqa: E501
                    val_description = re.sub(r'\s+', ' ', str(row.iloc[4]))
                    db_descriptions += f"Column {row.iloc[0]}: column description -> {col_description}, value description -> {val_description}\n"  # noqa: E501
           except Exception as e:
                print(e)
                db_descriptions += "No column description"
        db_descriptions += "\n"
    return db_descriptions

def generate_schema_prompt(sql_dialect, db_path=None, num_rows=None):
    if sql_dialect == "SQLite":
        schema_prompt = "\n### Schema of the database with sample rows and column descriptions:\n"+generate_schema_prompt_sqlite(db_path, num_rows)
        db_descriptions = db_path[:db_path.rfind('/')+1]+"database_description"
        column_description = table_descriptions_parser(db_descriptions)
        return schema_prompt+'\n\n'+column_description+'\n\n'
    else:
        raise ValueError("Unsupported SQL dialect: {}".format(sql_dialect))
    
# path = "/data/qqt/1615_c/text-to-sql/bird_minidev/llm/data/dev_databases/california_schools/california_schools.sqlite"
# print(generate_schema_prompt('SQLite',path))
