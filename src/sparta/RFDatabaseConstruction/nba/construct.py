import os
import hydra
import chardet
import psycopg2
import pandas as pd
from io import StringIO
from psycopg2 import sql
from omegaconf import DictConfig
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME

TEAM_INFO_KEYS = {
    "TEAM-NAME": "Name",
    "TEAM-AST": "Number of team assists",
    "TEAM-FG3_PCT": "Percentage of 3 points",
    "TEAM-FG_PCT": "Percentage of field goals",
    "TEAM-LOSSES": "Losses",
    "TEAM-PTS": "Total points",
    "TEAM-PTS_QTR1": "Points in 1st quarter",
    "TEAM-PTS_QTR2": "Points in 2nd quarter",
    "TEAM-PTS_QTR3": "Points in 3rd quarter",
    "TEAM-PTS_QTR4": "Points in 4th quarter",
    "TEAM-REB": "Rebounds",
    "TEAM-TOV": "Turnovers",
    "TEAM-WINS": "Wins",
}

PLAYER_INFO_KEYS = {
    "PLAYER_NAME": "Name",
    "AST": "Assists",
    "BLK": "Blocks",
    "DREB": "Defensive rebounds",
    "FG3A": "3-pointers attempted",
    "FG3M": "3-pointers made",
    "FG3_PCT": "3-pointer percentage",
    "FGA": "Field goals attempted",
    "FGM": "Field goals made",
    "FG_PCT": "Field goal percentage",
    "FTA": "Free throws attempted",
    "FTM": "Free throws made",
    "FT_PCT": "Free throw percentage",
    "MIN": "Minutes played",
    "OREB": "Offensive rebounds",
    "PF": "Personal fouls",
    "PTS": "Points",
    "REB": "Total rebounds",
    "START_POSITION": "Position",
    "STL": "Steals",
    "TO": "Turnovers",
}


def map_types(df):
    dtypedict = {}
    for i, j in zip(df.columns, df.dtypes):
        if "bool" in str(j):
            dtypedict.update({i: "BOOLEAN"})
        if "object" in str(j):
            max_len = df[i].astype(str).map(len).max()
            dtypedict.update({i: f"VARCHAR({max_len})"})
        if "float" in str(j):
            dtypedict.update({i: "float"})
        if "int" in str(j):
            dtypedict.update({i: "INT"})
        if "datetime64" in str(j):
            dtypedict.update({i: "TIMESTAMP"})
    return dtypedict


def get_file_paths(dir_path):
    file_paths = []

    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

    return file_paths


def construct_source_tables(dir_path, db_name, port):
    file_paths = get_file_paths(dir_path)
    for file_path in file_paths:
        file_name = os.path.basename(file_path).split(".")[0]
        rawdata = open(file_path, "rb").read()
        result = chardet.detect(rawdata)
        encoding = result["encoding"]

        if file_name == "top_player_career":
            df = pd.read_csv(file_path, encoding=encoding, delimiter=";")
        else:
            df = pd.read_csv(file_path, encoding=encoding, delimiter=",")

        conn = psycopg2.connect(
            database=db_name,
            user="postgres",
            password="postgres",
            host="localhost",
            port=port,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        table_name = file_name
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        fields = ", ".join([f"{col} {type}" for col, type in map_types(df).items()])
        create_table_query = f"CREATE TABLE {table_name} ({fields})"
        cursor.execute(create_table_query)

        sio = StringIO()
        sio.write(df.to_csv(index=None, header=None))
        sio.seek(0)
        copy_query = f"""
            COPY {table_name} FROM STDIN WITH 
            CSV 
            DELIMITER ',' 
            QUOTE '"'
            ESCAPE '"'
        """
        with conn.cursor() as c:
            c.copy_expert(copy_query, sio)
            conn.commit()

        cursor.close()
        conn.close()


def create_stat_table(dbname, port):
    con = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cursor = con.cursor()

    cursor.execute("DROP TABLE IF EXISTS prep_team_stats;")
    cursor.execute(
        f"""
                    CREATE TABLE prep_team_stats (
                    "Name" VARCHAR(500),
                    "Number of team assists" INTEGER,
                    "Percentage of 3 points" INTEGER,
                    "Percentage of field goals" INTEGER,
                    "Losses" INTEGER,
                    "Total points" INTEGER,
                    "Points in 1st quarter" INTEGER,
                    "Points in 2nd quarter" INTEGER,
                    "Points in 3rd quarter" INTEGER,
                    "Points in 4th quarter" INTEGER,
                    "Rebounds" INTEGER,
                    "Turnovers" INTEGER,
                    "Wins" INTEGER,
                    "summary" TEXT
                    );
                   """
    )

    cursor.execute("DROP TABLE IF EXISTS prep_person_stats;")
    cursor.execute(
        f"""
                    CREATE TABLE prep_person_stats (
                    "Name" VARCHAR(500),
                    "Assists" INTEGER,
                    "Blocks" INTEGER,
                    "Defensive rebounds" INTEGER,
                    "3-pointers attempted" INTEGER,
                    "3-pointers made" INTEGER,
                    "3-pointer percentage" INTEGER,
                    "Field goals attempted" INTEGER,
                    "Field goals made" INTEGER,
                    "Field goal percentage" INTEGER,
                    "Free throws attempted" INTEGER,
                    "Free throws made" INTEGER,
                    "Free throw percentage" INTEGER,
                    "Minutes played" INTEGER,
                    "Offensive rebounds" INTEGER,
                    "Personal fouls" INTEGER,
                    "Points" INTEGER,
                    "Total rebounds" INTEGER,
                    "Position" VARCHAR(50),
                    "Steals" INTEGER,
                    "Turnovers" INTEGER,
                    "summary" TEXT
                    );
                   """
    )

    cursor.execute("DROP TABLE IF EXISTS prep_game_information;")
    cursor.execute(
        f"""
                    CREATE TABLE prep_game_information (
                    "GAME-PLACE" VARCHAR(500),
                    "GAME-WEEKDAY" VARCHAR(500),
                    "GAME-STADIUM" VARCHAR(500),
                    "summary" TEXT
                    );
                   """
    )

    cursor.execute("DROP TABLE IF EXISTS prep_team_info;")
    cursor.execute(
        f"""
                    CREATE TABLE prep_team_info (
                    "HOME-CITY" VARCHAR(500),
                    "VIS-CITY" VARCHAR(500),
                    "HOME-CONFERENCE" VARCHAR(500),
                    "VIS-CONFERENCE" VARCHAR(500),
                    "HOME-DIVISION" VARCHAR(500),
                    "VIS-DIVISION" VARCHAR(500),
                    "summary" TEXT
                    );
                   """
    )

    cursor.execute("DROP TABLE IF EXISTS prep_next_info;")
    cursor.execute(
        f"""
                    CREATE TABLE prep_next_info (
                    "NEXT-HOME-OPPONENT" VARCHAR(500),
                    "NEXT-VIS-OPPONENT" VARCHAR(500),
                    "NEXT-HOME-STADIUM" VARCHAR(500),
                    "NEXT-VIS-STADIUM" VARCHAR(500),
                    "NEXT-HOME-PLACE" VARCHAR(500),
                    "NEXT-VIS-PLACE" VARCHAR(500),
                    "NEXT-HOME-WEEKDAY" VARCHAR(500),
                    "NEXT-VIS-WEEKDAY" VARCHAR(500),
                    "summary" TEXT
                    );
                   """
    )

    con.commit()
    cursor.close()
    con.close()


def insert_one_line(dbname, data_string, txt_string, port):
    con = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cursor = con.cursor()
    lines = data_string.split("<NEWLINE>")
    section = None
    team_headers = None
    player_headers = None
    game_information_headers = None
    team_info_headers = None
    next_info_headers = None

    for line in lines:
        if "Team:" in line:
            section = "Team"
            continue
        elif "Player:" in line:
            section = "Player"
            continue
        elif "Game_info:" in line:
            section = "game_information"
            continue
        elif "Team_info:" in line:
            section = "Team_info"
            continue
        elif "Next_info:" in line:
            section = "Next_info"
            continue

        def add_name(values, header):
            header = [value for value in values if value != ""]
            if len(header) == 0:
                return ["Name"]
            else:
                return ["Name"] + header

        values = [v.strip() for v in line.split("|")][1:-1]
        if section == "Team" and not team_headers:
            team_headers = add_name(values, team_headers)
        elif section == "Player" and not player_headers:
            player_headers = add_name(values, player_headers)
        elif section == "game_information" and not game_information_headers:
            game_information_headers = [value for value in values if value != ""]
        elif section == "Team_info" and not team_info_headers:
            team_info_headers = [value for value in values if value != ""]
        elif section == "Next_info" and not next_info_headers:
            next_info_headers = [value for value in values if value != ""]
        else:

            def construct_data_dict(header, values):
                exclude_keys = "game_id"
                filtered_keys = [key for key in header if key != exclude_keys]
                filtered_values = [
                    values[i] for i, key in enumerate(header) if key != exclude_keys
                ]
                data = dict(zip(filtered_keys, filtered_values))
                data["summary"] = txt_string.strip()

                return data

            if section == "Team":
                table = "prep_team_stats"
                data = construct_data_dict(team_headers, values)
            elif section == "Player":
                table = "prep_person_stats"
                data = construct_data_dict(player_headers, values)
            elif section == "game_information":
                table = "prep_game_information"
                data = construct_data_dict(game_information_headers, values)
            elif section == "Team_info":
                table = "prep_team_info"
                data = construct_data_dict(team_info_headers, values)
            elif section == "Next_info":
                table = "prep_next_info"
                data = construct_data_dict(next_info_headers, values)

            columns = ", ".join(['"' + col + '"' for col in data.keys()])
            placeholders = ", ".join(["%s"] * len(data.values()))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            for key, value in data.items():
                if value == "":
                    data[key] = None
            cursor.execute(query, list(data.values()))

    con.commit()
    cursor.close()
    con.close()


def insert_stat_table(grounding_data_dir_path, dbname, port):

    for split in ["train", "valid", "test"]:

        with open(f"{grounding_data_dir_path}/{split}.data", "r") as data_file, open(
            f"{grounding_data_dir_path}/{split}.txt", "r"
        ) as data_txts:
            for data_line, data_txt in zip(data_file, data_txts):
                insert_one_line(dbname, data_line, data_txt, port)


def generate_alter_statements(table_name, column_mapping):
    statements = []
    for key, value in column_mapping.items():
        stmt = f'ALTER TABLE {table_name} RENAME COLUMN "{value}" TO "{key}";'
        statements.append(stmt)
    return statements


def change_col_name(dbname, port):
    con = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cursor = con.cursor()

    team_statements = generate_alter_statements("prep_team_stats", TEAM_INFO_KEYS)
    player_statements = generate_alter_statements("prep_person_stats", PLAYER_INFO_KEYS)

    for stmt in team_statements:
        cursor.execute(stmt)
    for stmt in player_statements:
        cursor.execute(stmt)
    con.commit()

    print("change column names all finished")

    cursor.close()
    con.close()


def add_summary_id(dbname, port):

    con = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cursor = con.cursor()
    tables = [
        "prep_team_stats",
        "prep_person_stats",
        "prep_game_information",
        "prep_team_info",
        "prep_next_info",
    ]

    for table in tables:
        cursor.execute(
            f"""
                    ALTER TABLE {table} DROP COLUMN IF EXISTS summary_id;
                    """
        )
        con.commit()
        cursor.execute(
            f"""
                    ALTER TABLE {table} ADD COLUMN summary_id int;
                    """
        )
        con.commit()
        cursor.execute(
            f"""
                        UPDATE {table}
                        SET summary_id = rotowire_entries.id
                        FROM rotowire_entries
                        WHERE {table}.summary = rotowire_entries.summary;
                        """
        )
        con.commit()

    print("add columns all finished")
    for table in tables:
        cursor.execute(
            f"""
                    ALTER TABLE {table} DROP COLUMN "summary";
                    """
        )
        con.commit()

    print("drop summary columns all finished")
    cursor.close()
    con.close()


def construct_grounding_tables(grounding_data_dir_path, db_name, port):
    create_stat_table(db_name, port)
    insert_stat_table(grounding_data_dir_path, db_name, port)
    change_col_name(db_name, port)
    add_summary_id(db_name, port)


def link_two_databases(sql_file_path, dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cur = conn.cursor()
    with open(sql_file_path, "r") as file:
        sql_script = file.read()
    cur.execute(sql_script)
    conn.commit()
    cur.close()
    conn.close()


def rename_columns_in_db(table_name, cursor):
    """
    Rename columns in a given table based on specific criteria.
    """
    cursor.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'"
    )
    columns = [row[0] for row in cursor.fetchall()]

    for column in columns:
        if column == "summary_id":
            continue
        new_column = column.replace("-", "_").lower()
        if new_column != column:
            if new_column == "to":
                new_column = "turnover"
            cursor.execute(
                f'ALTER TABLE {table_name} RENAME COLUMN "{column}" TO {new_column}'
            )
            print(f"Renamed column {column} to {new_column}")


def create_btree_indexes(cur):
    """
    Create a B-tree index on every column of every table in a PostgreSQL database.

    :param database_config: A dictionary containing database connection details
    """
    cur.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public' 
    """
    )
    tables = [row[0] for row in cur.fetchall()]

    for table in tables:
        cur.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name='{table}';"
        )
        columns = [row[0] for row in cur.fetchall()]

        for column in columns:
            index_name = f"idx_{table}_{column}"
            cur.execute(f"DROP INDEX IF EXISTS {index_name};")

            try:
                cur.execute(
                    f"CREATE INDEX {index_name} ON {table} USING btree({column});"
                )
                print(f"Created index {index_name}")
            except Exception as e:
                print(f"Error creating index {index_name}. Reason: {e}")


def create_hash_indexes_on_foreign_keys(cur):
    """
    Create a hash index on every foreign key column in a PostgreSQL database.

    :param database_config: A dictionary containing database connection details
    """
    cur.execute(
        """
        SELECT kcu.column_name, kcu.table_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
        ON tc.constraint_name = kcu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
    """
    )

    for column_name, table_name in cur.fetchall():
        index_name = f"idx_hash_{table_name}_{column_name}"
        cur.execute(f"DROP INDEX IF EXISTS {index_name};")

        try:
            cur.execute(
                f"CREATE INDEX {index_name} ON {table_name} USING hash({column_name});"
            )
            print(f"Created hash index {index_name}")
        except Exception as e:
            print(f"Error creating hash index {index_name}. Reason: {e}")


def rename_columns(sql_file_path, dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname = {dbname} user=postgres password=postgres port = {port}"
    )
    cur = conn.cursor()
    tables = ["game_info", "next_info", "person_stats", "team_stats", "team_in_game"]
    for table in tables:
        rename_columns_in_db(table, cur)

    with open(sql_file_path, "r") as file:
        sql_script = file.read()
    cur.execute(sql_script)

    cur.execute(
        """
                CREATE TABLE game_information_1 AS
                SELECT 
                    game_summary.summary_id AS summary_id,
                    home_team_id, 
                    visitor_team_id, 
                    game_place, 
                    game_weekday, 
                    game_stadium
                FROM 
                    game_summary
                JOIN 
                    nba_game_home_away_record 
                ON 
                    game_summary.game_id = nba_game_home_away_record.game_id
                JOIN 
                    game_information 
                ON 
                    game_information.summary_id = game_summary.summary_id;
                """
    )
    cur.execute("DROP TABLE IF EXISTS game_information;")
    cur.execute("DROP TABLE IF EXISTS next_game_information;")
    cur.execute("ALTER TABLE game_summary DROP CONSTRAINT game_summary_game_id_fk;")
    cur.execute("ALTER TABLE game_information_1 RENAME TO game_information;")
    cur.execute("ALTER TABLE game_information ADD PRIMARY KEY (summary_id);")
    cur.execute(
        "ALTER TABLE game_information ADD CONSTRAINT game_information_summary_id_fk FOREIGN KEY (summary_id) REFERENCES summary(summary_id);"
    )
    cur.execute(
        "ALTER TABLE game_information ADD CONSTRAINT home_team_id_fk FOREIGN KEY (home_team_id) REFERENCES nba_team_information(team_id);"
    )
    cur.execute(
        "ALTER TABLE game_information ADD CONSTRAINT visitor_team_id_fk FOREIGN KEY (visitor_team_id) REFERENCES nba_team_information(team_id);"
    )

    create_btree_indexes(cur)
    print("-- btree finished")

    create_hash_indexes_on_foreign_keys(cur)
    print("-- hash finished")

    conn.commit()
    cur.close()
    conn.close()


def add_date(dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname={dbname} user=postgres password=postgres port={port}"
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='game_information' and column_name='game_date';"
    )
    if not cur.fetchone():
        cur.execute("ALTER TABLE game_information ADD COLUMN game_date DATE;")

    select_query = """
    SELECT nba_game_home_away_record.game_date_est, summary.summary_id, summary.summary
    FROM nba_game_home_away_record
    JOIN game_summary ON nba_game_home_away_record.game_id = game_summary.game_id
    JOIN summary ON summary.summary_id = game_summary.summary_id;
    """

    cur.execute(select_query)
    rows = cur.fetchall()
    for row in rows:
        game_date_est, summary_id, summary_text = row
        formatted_date = game_date_est

        formatted_date_str = formatted_date.strftime("On %d, %B, %Y,")
        first_sentence = summary_text.split(".")[0]
        updated_first_sentence = (
            formatted_date_str + " " + first_sentence[0].lower() + first_sentence[1:]
        )
        updated_summary = updated_first_sentence + summary_text[len(first_sentence) :]

        update_summary_query = """
        UPDATE summary
        SET summary = %s
        WHERE summary_id = %s;
        """
        cur.execute(update_summary_query, (updated_summary, summary_id))

        update_game_information_query = """
        UPDATE game_information
        SET game_date = %s
        WHERE summary_id = %s;
        """
        cur.execute(update_game_information_query, (formatted_date, summary_id))

    column_drop_query = """
    ALTER TABLE nba_game_home_away_record
    DROP COLUMN game_date_est;
    """
    cur.execute(column_drop_query)

    conn.commit()
    cur.close()
    conn.close()

    print("-- add date finished --")


def delete_duplicate_name(dbname, port, use_cascade=True):
    conn = psycopg2.connect(
        f"host=localhost dbname={dbname} user=postgres password=postgres port={port}"
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        """
        SELECT con.conname, con.conrelid::regclass AS table_name,
               pg_get_constraintdef(con.oid) as constraint_definition
        FROM pg_constraint con
        JOIN pg_namespace nsp ON con.connamespace = nsp.oid
        WHERE con.contype IN ('f', 'p')
        AND nsp.nspname = 'public'
        AND EXISTS (
            SELECT 1 
            FROM information_schema.table_constraints 
            WHERE constraint_name = con.conname
        );
    """
    )

    constraints = cur.fetchall()

    for conname, table_name, constraint_def in constraints:
        if use_cascade:
            drop_sql = (
                f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {conname} CASCADE;"
            )
        else:
            drop_sql = f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {conname};"

        try:
            cur.execute(drop_sql)
            print(f"Successfully dropped constraint {conname} on {table_name}")
        except Exception as e:
            print(f"Error dropping constraint {conname} on {table_name}: {str(e)}")

    print("All primary key and foreign key constraints have been dropped.")

    cur.execute(
        """
    UPDATE player_game_stats
    SET player_name = nba_player_information.name
    FROM nba_player_information
    WHERE player_game_stats.player_id = nba_player_information.player_id
    AND player_game_stats.player_name <> nba_player_information.name;
    """
    )

    cur.execute(
        """
    DELETE FROM nba_draft_combine_stats
    WHERE player_id IN (
    SELECT player_id FROM nba_player_information
    WHERE name IN (
        SELECT name FROM nba_player_information
        GROUP BY name
        HAVING COUNT(player_id) > 1
    )
    AND player_id NOT IN ('hendege02', 'willire02', 'dunlemi02', 'jamesmi02', 'tayloje03', 'johnsch04', 'leeda02'));
    """
    )

    cur.execute(
        """
    DELETE FROM nba_player_award
    WHERE player_id IN (
    SELECT player_id FROM nba_player_information
    WHERE name IN (
        SELECT name FROM nba_player_information
        GROUP BY name
        HAVING COUNT(player_id) > 1
    )
    AND player_id NOT IN ('hendege02', 'willire02', 'dunlemi02', 'jamesmi02', 'tayloje03', 'johnsch04', 'leeda02'));
    """
    )

    cur.execute(
        """
    DELETE FROM nba_player_affiliation
    WHERE player_id IN (
    SELECT player_id FROM nba_player_information
    WHERE name IN (
        SELECT name FROM nba_player_information
        GROUP BY name
        HAVING COUNT(player_id) > 1
    )
    AND player_id NOT IN ('hendege02', 'willire02', 'dunlemi02', 'jamesmi02', 'tayloje03', 'johnsch04', 'leeda02'));
    """
    )

    cur.execute(
        """
    UPDATE nba_champion_history
    SET mvp_player = (
        SELECT name
        FROM nba_player_information
        WHERE nba_champion_history.mvp_player = nba_player_information.player_id
    );
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD COLUMN western_champion_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_champion_history
    SET western_champion_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE nba_champion_history.western_champion = nba_team_information.team_id
    );
    """
    )

    cur.execute(
        "ALTER TABLE nba_champion_history DROP COLUMN western_champion CASCADE;"
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD COLUMN eastern_champion_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_champion_history
    SET eastern_champion_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE nba_champion_history.eastern_champion = nba_team_information.team_id
    );
    """
    )

    cur.execute("ALTER TABLE nba_champion_history DROP COLUMN eastern_champion;")

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD COLUMN nba_champion_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_champion_history
    SET nba_champion_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE nba_champion_history.nba_champion = nba_team_information.team_id
    );
    """
    )

    cur.execute("ALTER TABLE nba_champion_history DROP COLUMN nba_champion;")

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD COLUMN nba_vice_champion_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_champion_history
    SET nba_vice_champion_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE nba_champion_history.nba_vice_champion = nba_team_information.team_id
    );
    """
    )

    cur.execute("ALTER TABLE nba_champion_history DROP COLUMN nba_vice_champion;")

    cur.execute(
        """
    ALTER TABLE game_information
    ADD COLUMN home_team_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE game_information
    SET home_team_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE game_information.home_team_id = nba_team_information.team_id
    );
    """
    )

    cur.execute("ALTER TABLE game_information DROP COLUMN home_team_id;")

    cur.execute(
        """
    ALTER TABLE game_information
    ADD COLUMN visitor_team_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE game_information
    SET visitor_team_name = (
        SELECT nickname
        FROM nba_team_information
        WHERE game_information.visitor_team_id = nba_team_information.team_id
    );
    """
    )

    cur.execute("ALTER TABLE game_information DROP COLUMN visitor_team_id;")

    cur.execute(
        """
    DELETE FROM nba_player_information
    WHERE player_id IN (
    SELECT player_id FROM nba_player_information
    WHERE name IN (
        SELECT name FROM nba_player_information
        GROUP BY name
        HAVING COUNT(player_id) > 1
    )
    AND player_id NOT IN ('hendege02', 'willire02', 'dunlemi02', 'jamesmi02', 'tayloje03', 'johnsch04', 'leeda02'));
    """
    )

    cur.execute(
        """
    UPDATE nba_player_affiliation
    SET team = (
        SELECT nickname
        FROM nba_team_information
        WHERE nba_player_affiliation.team_id = nba_team_information.team_id
    );
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_affiliation
    ADD COLUMN player_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_player_affiliation
    SET player_name = (
        SELECT name
        FROM nba_player_information
        WHERE nba_player_affiliation.player_id = nba_player_information.player_id
    );
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_award
    ADD COLUMN player_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_player_award
    SET player_name = (
        SELECT name
        FROM nba_player_information
        WHERE nba_player_award.player_id = nba_player_information.player_id
    );
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_draft_combine_stats
    ADD COLUMN player_name VARCHAR(24);
    """
    )

    cur.execute(
        """
    UPDATE nba_draft_combine_stats
    SET player_name = (
        SELECT name
        FROM nba_player_information
        WHERE nba_draft_combine_stats.player_id = nba_player_information.player_id
    );
    """
    )

    cur.execute("ALTER TABLE player_game_stats DROP COLUMN player_id;")
    cur.execute("ALTER TABLE nba_player_information DROP COLUMN player_id CASCADE;")
    cur.execute("ALTER TABLE nba_player_award DROP COLUMN player_id;")
    cur.execute("ALTER TABLE nba_player_affiliation DROP COLUMN player_id CASCADE;")
    cur.execute("ALTER TABLE nba_draft_combine_stats DROP COLUMN player_id;")
    cur.execute("ALTER TABLE team_game_stats DROP COLUMN team_id;")
    cur.execute("ALTER TABLE nba_team_information DROP COLUMN team_id CASCADE;")
    cur.execute("ALTER TABLE nba_team_information DROP COLUMN team_name CASCADE;")
    cur.execute("ALTER TABLE nba_player_affiliation DROP COLUMN team_id;")

    cur.execute("ALTER TABLE game_information RENAME TO nba_game_information;")
    cur.execute("ALTER TABLE team_game_stats RENAME TO nba_team_game_stats;")
    cur.execute("ALTER TABLE player_game_stats RENAME TO nba_player_game_stats;")

    cur.execute(
        """
    DELETE FROM nba_player_game_stats
    WHERE summary_id NOT IN (
        SELECT summary_id FROM nba_game_information
    );
    """
    )

    cur.execute(
        """
    DELETE FROM nba_team_game_stats
    WHERE summary_id NOT IN (
        SELECT summary_id FROM nba_game_information
    );
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_affiliation
    RENAME COLUMN team TO team_name;
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_team_information
    RENAME COLUMN nickname TO team_name;
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT year_pk PRIMARY KEY (year);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_information
    ADD CONSTRAINT name_pk PRIMARY KEY (name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_affiliation
    ADD CONSTRAINT nba_player_affiliation_pk PRIMARY KEY (season, team_name, player_name, salary);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_award
    ADD CONSTRAINT nba_player_award_pk PRIMARY KEY (season, award, player_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_draft_combine_stats
    ADD CONSTRAINT nba_draft_combine_stats_pk PRIMARY KEY (player_name, season);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_team_information
    ADD CONSTRAINT nba_team_information_pk PRIMARY KEY (team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_game_stats
    ADD CONSTRAINT nba_player_game_stats_pk PRIMARY KEY (summary_id, player_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_team_game_stats
    ADD CONSTRAINT nba_team_game_stats_pk PRIMARY KEY (summary_id, team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_game_information
    ADD CONSTRAINT nba_game_information_pk PRIMARY KEY (summary_id);
    """
    )

    cur.execute(
        """
    ALTER TABLE summary
    ADD CONSTRAINT summary_pk PRIMARY KEY (summary_id);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_game_stats
    ADD CONSTRAINT nba_player_game_stats_summary_id_fk FOREIGN KEY (summary_id) REFERENCES nba_game_information(summary_id);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_game_stats
    ADD CONSTRAINT nba_player_game_stats_player_name_fk FOREIGN KEY (player_name) REFERENCES nba_player_information(name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_team_game_stats
    ADD CONSTRAINT nba_team_game_stats_summary_id_fk FOREIGN KEY (summary_id) REFERENCES nba_game_information(summary_id);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_team_game_stats
    ADD CONSTRAINT nba_team_game_stats_team_name_fk FOREIGN KEY (team_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_game_information
    ADD CONSTRAINT nba_game_information_home_team_name_fk FOREIGN KEY (home_team_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_game_information
    ADD CONSTRAINT nba_game_information_visitor_team_name_fk FOREIGN KEY (visitor_team_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_affiliation
    ADD CONSTRAINT nba_player_affiliation_player_name_fk FOREIGN KEY (player_name) REFERENCES nba_player_information(name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_affiliation
    ADD CONSTRAINT nba_player_affiliation_team_name_fk FOREIGN KEY (team_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_award
    ADD CONSTRAINT nba_player_award_player_name_fk FOREIGN KEY (player_name) REFERENCES nba_player_information(name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_draft_combine_stats
    ADD CONSTRAINT nba_draft_combine_stats_player_name_fk FOREIGN KEY (player_name) REFERENCES nba_player_information(name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT nba_champion_history_nba_champion_name_fk FOREIGN KEY (nba_champion_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT nba_champion_history_nba_vice_champion_name_fk FOREIGN KEY (nba_vice_champion_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT nba_champion_history_western_champion_name_fk FOREIGN KEY (western_champion_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT nba_champion_history_eastern_champion_name_fk FOREIGN KEY (eastern_champion_name) REFERENCES nba_team_information(team_name);
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    ADD CONSTRAINT nba_champion_history_mvp_player_fk FOREIGN KEY (mvp_player) REFERENCES nba_player_information(name);
    """
    )

    cur.execute("DROP TABLE IF EXISTS nba_game_home_away_record;")

    cur.execute(
        """
    ALTER TABLE nba_champion_history
    RENAME COLUMN mvp_player TO mvp_player_name;
    """
    )

    cur.execute(
        """
    ALTER TABLE nba_player_information
    RENAME COLUMN name TO player_name;
    """
    )

    cur.close()
    conn.close()


def define_auxiliary(sql_file_path, dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname={dbname} user=postgres password=postgres port={port}"
    )
    conn.autocommit = True
    cur = conn.cursor()
    with open(sql_file_path, "r") as file:
        sql_script = file.read()

    cur.execute(sql_script)
    conn.commit()
    cur.execute(
        """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
    """
    )
    table_names = [row[0] for row in cur.fetchall()]

    for table_name in table_names:
        drop_table1 = f"DROP TABLE IF EXISTS {table_name}_1 CASCADE;"
        create_table1 = f"""
            CREATE TABLE {table_name}_1 AS
            SELECT nextval('tuid_seq')::tuid_t AS tuid, t.*
            FROM {table_name} AS t;
        """
        alter_table1_pk = f"ALTER TABLE {table_name}_1 ADD PRIMARY KEY (tuid);"
        cur.execute(drop_table1)
        cur.execute(create_table1)
        cur.execute(alter_table1_pk)
        query_columns = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name.lower()}_1'
            AND table_schema = 'public'   -- 만약 public 스키마에 만들었다면
            ORDER BY ordinal_position
        """
        cur.execute(query_columns)
        columns = [row[0] for row in cur.fetchall() if row[0] != "tuid"]
        drop_table2 = f"DROP TABLE IF EXISTS {table_name}_2 CASCADE;"
        cur.execute(drop_table2)

        columns_annot = ",\n    ".join(
            [f"ARRAY[nextval('annot_seq')]::pset_t AS {col}" for col in columns]
        )
        create_table2 = f"""
            CREATE TABLE {table_name}_2 AS
            SELECT
                d.tuid,
                {columns_annot}
            FROM {table_name}_1 AS d;
        """
        alter_table2_pk = f"ALTER TABLE {table_name}_2 ADD PRIMARY KEY (tuid);"

        cur.execute(create_table2)
        cur.execute(alter_table2_pk)

    cur.close()
    conn.close()


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    args = cfg.sparta.db
    source_data_dir_path = os.path.join(cfg.root_dir_path, args.source_data_dir_path)
    grounding_data_dir_path = os.path.join(
        cfg.root_dir_path, args.grounding_data_dir_path
    )
    join_file_path = os.path.join(cfg.root_dir_path, args.join_file_path)
    rename_file_path = os.path.join(cfg.root_dir_path, args.rename_file_path)
    aux_file_path = os.path.join(cfg.root_dir_path, args.aux_file_path)
    db_name = args.db_name
    port = args.port

    print("start construct source tables")
    construct_source_tables(source_data_dir_path, db_name, port)
    print("construct source tables finished")

    print("start construct grounding tables")
    construct_grounding_tables(grounding_data_dir_path, db_name, port)
    print("construct grounding tables finished")

    print("start link two databases")
    link_two_databases(join_file_path, db_name, port)
    print("link two databases finished")

    print("start rename columns")
    rename_columns(rename_file_path, db_name, port)
    print("rename columns finished")

    print("start add date")
    add_date(db_name, port)
    print("add date finished")

    print("start delete duplicate name")
    delete_duplicate_name(db_name, port)
    print("delete duplicate name finished")

    print("start define auxiliary function")
    define_auxiliary(aux_file_path, db_name, port)
    print("define auxiliary function finished")
    print("reference fact database construction finished")


if __name__ == "__main__":
    main()
