import pandas as pd
import psycopg2
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME
from omegaconf import DictConfig
import hydra
import os


# ====== 1. Create database ======
def create_database(port):
    conn = psycopg2.connect(
        host="localhost",
        dbname="postgres",
        user="postgres",
        password="postgres",
        port=port,
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute("DROP DATABASE IF EXISTS imdb;")
    cursor.execute("CREATE DATABASE imdb;")
    cursor.close()
    conn.close()
    print("[Step 1] Database 'imdb' created successfully.")


# ====== 2. Create tables ======
def create_tables(port):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    schema_sql = """
    DROP TABLE IF EXISTS movie CASCADE;
    CREATE TABLE movie (
      id VARCHAR(10),
      title VARCHAR(200),
      year INT,
      date_published DATE,
      duration INT,
      country VARCHAR(250),
      worlwide_gross_income VARCHAR(30),
      languages VARCHAR(200),
      production_company VARCHAR(200)
    );

    DROP TABLE IF EXISTS genre CASCADE;
    CREATE TABLE genre (
        movie_id VARCHAR(10),
        genre VARCHAR(20)
    );

    DROP TABLE IF EXISTS director_mapping CASCADE;
    CREATE TABLE director_mapping (
        movie_id VARCHAR(10),
        name_id VARCHAR(10)
    );

    DROP TABLE IF EXISTS role_mapping CASCADE;
    CREATE TABLE role_mapping (
        movie_id VARCHAR(10),
        name_id VARCHAR(10),
        category VARCHAR(10)
    );

    DROP TABLE IF EXISTS names CASCADE;
    CREATE TABLE names (
      id VARCHAR(10),
      name VARCHAR(100),
      height INT,
      date_of_birth DATE,
      known_for_movies VARCHAR(100)
    );

    DROP TABLE IF EXISTS ratings CASCADE;
    CREATE TABLE ratings (
        movie_id VARCHAR(10),
        avg_rating DECIMAL(3,1),
        total_votes INT,
        median_rating INT
    );
    """
    for q in schema_sql.strip().split(";"):
        q = q.strip()
        if q:
            cursor.execute(q)

    cursor.close()
    conn.close()
    print("[Step 2] All tables created successfully.")


# ====== 3. Insert data from Excel ======
def insert_data_from_excel(port, excel_path):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    dfs = pd.read_excel(excel_path, sheet_name=None)

    def insert_dataframe(table_name, df):
        if "date_of_birth" in df.columns:
            df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

        df = df.astype(object).where(pd.notnull(df), None)
        columns = list(df.columns)
        col_str = ",".join(columns)
        placeholders = ",".join(["%s"] * len(columns))
        insert_sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholders})"
        print(f"[Step 3] Inserting {len(df)} rows into {table_name}...")

        for idx, row in df.iterrows():
            try:
                cursor.execute(insert_sql, tuple(row))
            except psycopg2.errors.NumericValueOutOfRange as e:
                print(
                    f"Row index {idx} in table {table_name} caused error: {row.to_dict()}"
                )
                raise

    for sheet_name, data in dfs.items():
        table = sheet_name.lower()
        insert_dataframe(table, data)

    cursor.close()
    conn.close()
    print("[Step 3] All data inserted successfully.")


# ====== 4. Modify knownformovie column ======
def modify_known_for_movies(port):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("[Step 4] Modifying known_for_movies and creating known_for_movie...")

    # 1. Keep only the first movie_id in known_for_movies
    cursor.execute(
        """
        UPDATE names
        SET known_for_movies = split_part(known_for_movies, ',', 1);
    """
    )

    # 2. Add new column known_for_movie (title)
    cursor.execute(
        """
        ALTER TABLE names ADD COLUMN known_for_movie VARCHAR(200);
    """
    )

    # 3. Update known_for_movie by joining movie_id -> title
    cursor.execute(
        """
        UPDATE names n
        SET known_for_movie = m.title
        FROM movie m
        WHERE n.known_for_movies = m.id;
    """
    )

    # 4. Drop the old known_for_movies column
    cursor.execute(
        """
        ALTER TABLE names DROP COLUMN known_for_movies;
    """
    )

    print("[Step 4] known_for_movies cleaned and replaced with known_for_movie!")

    cursor.close()
    conn.close()


def join_with_name(port):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("[Step 5] Cleaning duplicates and joining with name...")

    # 1. Delete duplicate names
    cursor.execute(
        """
        DELETE FROM names
        WHERE name IN (
            SELECT name
            FROM names
            GROUP BY name
            HAVING COUNT(*) > 1
        );
    """
    )

    # 2. Delete unmapped rows in role_mapping and director_mapping
    cursor.execute(
        """
        DELETE FROM role_mapping
        WHERE name_id NOT IN (SELECT id FROM names);
    """
    )
    cursor.execute(
        """
        DELETE FROM director_mapping
        WHERE name_id NOT IN (SELECT id FROM names);
    """
    )

    # 3. Add name column to both tables
    cursor.execute("ALTER TABLE director_mapping ADD COLUMN name VARCHAR(100);")
    cursor.execute("ALTER TABLE role_mapping ADD COLUMN name VARCHAR(100);")

    # 4. Update name column by joining with names
    cursor.execute(
        """
        UPDATE director_mapping d
        SET name = n.name
        FROM names n
        WHERE d.name_id = n.id;
    """
    )
    cursor.execute(
        """
        UPDATE role_mapping r
        SET name = n.name
        FROM names n
        WHERE r.name_id = n.id;
    """
    )

    # 5. change column name from id to name_id in names table
    cursor.execute("ALTER TABLE names RENAME COLUMN id TO name_id;")

    print("[Step 5] join_with_name completed successfully!")

    cursor.close()
    conn.close()


def join_with_movie_title(port):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("[Step 6] Cleaning duplicates in movie and joining with movie_title...")

    # 1. Delete duplicate titles in movie table
    cursor.execute(
        """
        DELETE FROM movie
        WHERE title IN (
            SELECT title
            FROM movie
            GROUP BY title
            HAVING COUNT(*) > 1
        );
    """
    )

    # 2. Remove rows in 4 tables that no longer match movie.id
    cursor.execute(
        """
        DELETE FROM director_mapping
        WHERE movie_id NOT IN (SELECT id FROM movie);
    """
    )
    cursor.execute(
        """
        DELETE FROM role_mapping
        WHERE movie_id NOT IN (SELECT id FROM movie);
    """
    )
    cursor.execute(
        """
        DELETE FROM ratings
        WHERE movie_id NOT IN (SELECT id FROM movie);
    """
    )
    cursor.execute(
        """
        DELETE FROM genre
        WHERE movie_id NOT IN (SELECT id FROM movie);
    """
    )

    # 3. Add movie_title column to 4 tables
    cursor.execute("ALTER TABLE director_mapping ADD COLUMN movie_title VARCHAR(200);")
    cursor.execute("ALTER TABLE role_mapping ADD COLUMN movie_title VARCHAR(200);")
    cursor.execute("ALTER TABLE ratings ADD COLUMN movie_title VARCHAR(200);")
    cursor.execute("ALTER TABLE genre ADD COLUMN movie_title VARCHAR(200);")

    # 4. Update movie_title based on movie_id from movie table
    cursor.execute(
        """
        UPDATE director_mapping d
        SET movie_title = m.title
        FROM movie m
        WHERE d.movie_id = m.id;
    """
    )
    cursor.execute(
        """
        UPDATE role_mapping r
        SET movie_title = m.title
        FROM movie m
        WHERE r.movie_id = m.id;
    """
    )
    cursor.execute(
        """
        UPDATE ratings ra
        SET movie_title = m.title
        FROM movie m
        WHERE ra.movie_id = m.id;
    """
    )
    cursor.execute(
        """
        UPDATE genre g
        SET movie_title = m.title
        FROM movie m
        WHERE g.movie_id = m.id;
    """
    )

    # 5. Drop movie_id column from 4 tables
    cursor.execute("ALTER TABLE director_mapping DROP COLUMN movie_id;")
    cursor.execute("ALTER TABLE role_mapping DROP COLUMN movie_id;")
    cursor.execute("ALTER TABLE ratings DROP COLUMN movie_id;")
    cursor.execute("ALTER TABLE genre DROP COLUMN movie_id;")

    # 6. Drop id column from movie table
    cursor.execute("ALTER TABLE movie DROP COLUMN id;")

    print("[Step 6] join_with_movie_title completed successfully!")

    cursor.close()
    conn.close()


def construct_key_constraint(port):
    conn = psycopg2.connect(
        host="localhost", dbname="imdb", user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("[Step 7] Adding primary keys...")

    # --- Add primary keys ---
    cursor.execute("ALTER TABLE movie ADD PRIMARY KEY (title);")
    cursor.execute("ALTER TABLE genre ADD PRIMARY KEY (movie_title, genre);")
    cursor.execute("ALTER TABLE ratings ADD PRIMARY KEY (movie_title);")
    cursor.execute("ALTER TABLE role_mapping ADD PRIMARY KEY (movie_title, name);")
    cursor.execute("ALTER TABLE director_mapping ADD PRIMARY KEY (movie_title, name);")
    cursor.execute("ALTER TABLE names ADD PRIMARY KEY (name);")

    print("[Step 7] Adding foreign keys...")

    # --- Add foreign keys ---
    cursor.execute(
        """
        ALTER TABLE genre
        ADD CONSTRAINT fk_genre_movie
        FOREIGN KEY (movie_title) REFERENCES movie(title) ON DELETE CASCADE;
    """
    )
    cursor.execute(
        """
        ALTER TABLE ratings
        ADD CONSTRAINT fk_ratings_movie
        FOREIGN KEY (movie_title) REFERENCES movie(title) ON DELETE CASCADE;
    """
    )
    cursor.execute(
        """
        ALTER TABLE role_mapping
        ADD CONSTRAINT fk_role_movie
        FOREIGN KEY (movie_title) REFERENCES movie(title) ON DELETE CASCADE;
    """
    )
    cursor.execute(
        """
        ALTER TABLE director_mapping
        ADD CONSTRAINT fk_director_movie
        FOREIGN KEY (movie_title) REFERENCES movie(title) ON DELETE CASCADE;
    """
    )
    cursor.execute(
        """
        ALTER TABLE role_mapping
        ADD CONSTRAINT fk_role_name
        FOREIGN KEY (name) REFERENCES names(name) ON DELETE CASCADE;
    """
    )
    cursor.execute(
        """
        ALTER TABLE director_mapping
        ADD CONSTRAINT fk_director_name
        FOREIGN KEY (name) REFERENCES names(name) ON DELETE CASCADE;
    """
    )

    print("[Step 7] Key constraints constructed successfully!")

    cursor.close()
    conn.close()


def define_auxiliary(sql_file_path, dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname={dbname} user=postgres password=postgres port={port}"
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("[Step 8] Defining auxiliary tables .... ")
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

    print("[Step 8] Completed defining auxiliary tables.")

    cur.close()
    conn.close()


# ====== main ======
@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    args = cfg.sparta.db
    port = args.port
    db_name = args.db_name
    excel_path = os.path.join(cfg.root_dir_path, args.excel_path)
    aux_file_path = os.path.join(cfg.root_dir_path, args.aux_file_path)
    create_database(port)
    create_tables(port)
    insert_data_from_excel(
        port,
        excel_path,
    )
    modify_known_for_movies(port)
    join_with_name(port)
    join_with_movie_title(port)
    construct_key_constraint(port)
    define_auxiliary(aux_file_path, db_name, port)  # for provenance


if __name__ == "__main__":
    main()
