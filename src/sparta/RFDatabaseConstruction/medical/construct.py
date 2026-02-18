import pandas as pd
import psycopg2
import hydra
import os
from omegaconf import DictConfig
from config.path import ABS_CONFIG_DIR, DEFAULT_CONFIG_FILE_NAME


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
    cursor.execute("DROP DATABASE IF EXISTS medical;")
    cursor.execute("CREATE DATABASE medical;")
    cursor.close()
    conn.close()
    print("[Step 1] Database 'medical' created successfully.")


# ====== 2. Create tables ======
def create_tables(port, dbname):
    conn = psycopg2.connect(
        host="localhost", dbname=dbname, user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    schema_sql = """
    DROP TABLE IF EXISTS appointments CASCADE;
    CREATE TABLE appointments (
      appointment_id VARCHAR(10),
      patient_id VARCHAR(10),
      doctor_id VARCHAR(10),
      appointment_date DATE,
      appointment_time TIME,
      reason_for_visit VARCHAR(20),
      status VARCHAR(20)
    );

    DROP TABLE IF EXISTS billing CASCADE;
    CREATE TABLE billing (
        bill_id VARCHAR(10),
        patient_id VARCHAR(10),
        treatment_id VARCHAR(10),
        bill_date DATE,
        amount DECIMAL(10, 2),
        payment_method VARCHAR(20),
        payment_status VARCHAR(20)
    );

    DROP TABLE IF EXISTS doctors CASCADE;
    CREATE TABLE doctors (
        doctor_id VARCHAR(10),
        first_name VARCHAR(10),
        last_name VARCHAR(10),
        specialization VARCHAR(50),
        phone_number VARCHAR(20),
        years_experience INT,
        hospital_branch VARCHAR(50),
        email VARCHAR(100)
    );

    DROP TABLE IF EXISTS patients CASCADE;
    CREATE TABLE patients (
        patient_id VARCHAR(10),
        first_name VARCHAR(10),
        last_name VARCHAR(10),
        gender VARCHAR(10),
        date_of_birth DATE,
        contact_number VARCHAR(20),
        address VARCHAR(100),
        registration_date DATE,
        insurance_provider VARCHAR(50),
        insurance_number VARCHAR(20),
        email VARCHAR(100)
    );

    DROP TABLE IF EXISTS treatments CASCADE;
    CREATE TABLE treatments (
        treatment_id VARCHAR(10),
        appointment_id VARCHAR(10),
        treatment_type VARCHAR(50),
        description VARCHAR(200),
        cost DECIMAL(10, 2),
        treatment_date DATE
    );
    """
    for q in schema_sql.strip().split(";"):
        q = q.strip()
        if q:
            cursor.execute(q)

    cursor.close()
    conn.close()
    print("[Step 2] All tables created successfully.")


# ====== 3. Insert data from csv files ======
def insert_data_from_csv(port, dbname, data_dir_path):
    conn = psycopg2.connect(
        host="localhost", dbname=dbname, user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # Load data from csv files
    csv_files = [
        "appointments.csv",
        "billing.csv",
        "doctors.csv",
        "patients.csv",
        "treatments.csv",
    ]
    for csv_file in csv_files:
        table_name = csv_file.split(".")[0].lower()
        df = pd.read_csv(f"{data_dir_path}/{csv_file}")

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

    cursor.close()
    conn.close()
    print("[Step 3] All data inserted successfully.")


def remove_columns(port, dbname):
    conn = psycopg2.connect(
        host="localhost", dbname=dbname, user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # patients table
    patient_columns_to_remove = ["insurance_number", "contact_number", "email"]
    for col in patient_columns_to_remove:
        try:
            cursor.execute(f"ALTER TABLE patients DROP COLUMN IF EXISTS {col};")
            print(f"[Step 4] Removed column '{col}' from patients.")
        except Exception as e:
            print(f"[Step 4] Failed to remove column '{col}' from patients: {e}")

    # doctors table
    doctor_columns_to_remove = ["phone_number", "email"]
    for col in doctor_columns_to_remove:
        try:
            cursor.execute(f"ALTER TABLE doctors DROP COLUMN IF EXISTS {col};")
            print(f"[Step 4] Removed column '{col}' from doctors.")
        except Exception as e:
            print(f"[Step 4] Failed to remove column '{col}' from doctors: {e}")

    cursor.close()
    conn.close()
    print("[Step 4] Column removal process completed.")


def denormalize_names(port, dbname):
    conn = psycopg2.connect(
        host="localhost", dbname=dbname, user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    # --- 1. patients.name 추가 및 업데이트
    cursor.execute("ALTER TABLE patients ADD COLUMN IF NOT EXISTS name VARCHAR(50);")
    cursor.execute("UPDATE patients SET name = first_name || ' ' || last_name;")
    print("[Step 5] Added and populated patients.name")

    # --- [NEW STEP] 중복 name을 가진 patients 삭제
    print("[Step 5] Checking for duplicate names in patients...")
    cursor.execute(
        """
        SELECT first_name, last_name
        FROM patients
        GROUP BY first_name, last_name
        HAVING COUNT(*) > 1;
    """
    )
    dup_name_rows = cursor.fetchall()

    if dup_name_rows:
        print("[Step 5] Found duplicate first_name/last_name combos:")
        print(dup_name_rows)

        # 미들네임 후보
        middle_names = ["Aiden", "Grace", "James"]

        for first_name, last_name in dup_name_rows:
            # 이 first_name/last_name를 가진 환자들을 patient_id로 정렬
            cursor.execute(
                """
                SELECT patient_id
                FROM patients
                WHERE first_name = %s AND last_name = %s
                ORDER BY patient_id;
            """,
                (first_name, last_name),
            )
            dup_patient_ids = [row[0] for row in cursor.fetchall()]

            # 첫 번째는 그대로 두고, 두 번째부터 미들네임 부여
            for idx, pid in enumerate(dup_patient_ids):
                if idx == 0:
                    continue  # 첫 번째는 변경하지 않음
                mid_idx = idx - 1
                if mid_idx < len(middle_names):
                    middle = middle_names[mid_idx]
                else:
                    middle = f"X{idx}"  # 혹시 4개 이상일 때 대비

                new_name = f"{first_name} {middle} {last_name}"
                cursor.execute(
                    """
                    UPDATE patients
                    SET name = %s
                    WHERE patient_id = %s;
                """,
                    (new_name, pid),
                )
                print(f"[Step 5] Updated duplicate patient_id={pid} -> name={new_name}")
    else:
        print("[Step 5] No duplicate names found in patients.")

    # --- 2. billing.patient_name 추가 및 업데이트
    cursor.execute(
        "ALTER TABLE billing ADD COLUMN IF NOT EXISTS patient_name VARCHAR(50);"
    )
    cursor.execute(
        """
        UPDATE billing b
        SET patient_name = p.name
        FROM patients p
        WHERE b.patient_id = p.patient_id;
    """
    )
    print("[Step 5] Added and populated billing.patient_name")

    # --- 3. appointments.patient_name 추가 및 업데이트
    cursor.execute(
        "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS patient_name VARCHAR(50);"
    )
    cursor.execute(
        """
        UPDATE appointments a
        SET patient_name = p.name
        FROM patients p
        WHERE a.patient_id = p.patient_id;
    """
    )
    print("[Step 5] Added and populated appointments.patient_name")

    # --- 4. patient_id 컬럼 제거
    cursor.execute("ALTER TABLE billing DROP COLUMN IF EXISTS patient_id;")
    cursor.execute("ALTER TABLE appointments DROP COLUMN IF EXISTS patient_id;")
    print("[Step 5] Removed patient_id from billing and appointments")

    # --- 5. doctors.name 추가 및 업데이트
    cursor.execute("ALTER TABLE doctors ADD COLUMN IF NOT EXISTS name VARCHAR(50);")
    cursor.execute("UPDATE doctors SET name = first_name || ' ' || last_name;")
    print("[Step 5] Added and populated doctors.name")

    # --- 6. appointments.doctor_name 추가 및 업데이트
    cursor.execute(
        "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS doctor_name VARCHAR(50);"
    )
    cursor.execute(
        """
        UPDATE appointments a
        SET doctor_name = d.name
        FROM doctors d
        WHERE a.doctor_id = d.doctor_id;
    """
    )
    print("[Step 5] Added and populated appointments.doctor_name")

    # --- 7. doctor_id 컬럼 제거
    cursor.execute("ALTER TABLE appointments DROP COLUMN IF EXISTS doctor_id;")
    print("[Step 5] Removed doctor_id from appointments")

    # --- 8. patients 테이블에서 first_name, last_name 컬럼 제거
    cursor.execute("ALTER TABLE patients DROP COLUMN IF EXISTS first_name;")
    cursor.execute("ALTER TABLE patients DROP COLUMN IF EXISTS last_name;")
    print("[Step 5] Removed first_name and last_name from patients")

    # --- 9. doctors 테이블에서 first_name, last_name 컬럼 제거
    cursor.execute("ALTER TABLE doctors DROP COLUMN IF EXISTS first_name;")
    cursor.execute("ALTER TABLE doctors DROP COLUMN IF EXISTS last_name;")
    print("[Step 5] Removed first_name and last_name from doctors")

    cursor.close()
    conn.close()
    print("[Step 5] Denormalization completed.")


def construct_key_constraint(port, dbname):
    conn = psycopg2.connect(
        host="localhost", dbname=dbname, user="postgres", password="postgres", port=port
    )
    conn.autocommit = True
    cursor = conn.cursor()

    print("[Step 6] Starting to add primary and foreign key constraints...")

    # =======================
    # 1. PRIMARY KEYS
    # =======================
    pk_statements = [
        "ALTER TABLE treatments ADD PRIMARY KEY (treatment_id);",
        "ALTER TABLE billing ADD PRIMARY KEY (bill_id);",
        "ALTER TABLE appointments ADD PRIMARY KEY (appointment_id);",
        "ALTER TABLE patients ADD PRIMARY KEY (name);",
        "ALTER TABLE doctors ADD PRIMARY KEY (name);",
    ]

    for sql in pk_statements:
        try:
            cursor.execute(sql)
            print(f"[Step 6] Executed: {sql}")
        except Exception as e:
            print(f"[Step 6][PK] Failed on: {sql}\n  Reason: {e}")

    # =======================
    # 2. FOREIGN KEYS
    # =======================
    fk_statements = [
        # treatments.appointment_id -> appointments.appointment_id
        """ALTER TABLE treatments 
           ADD CONSTRAINT fk_treatments_appointment
           FOREIGN KEY (appointment_id)
           REFERENCES appointments (appointment_id)
           ON DELETE CASCADE;""",
        # billing.treatment_id -> treatments.treatment_id
        """ALTER TABLE billing
           ADD CONSTRAINT fk_billing_treatment
           FOREIGN KEY (treatment_id)
           REFERENCES treatments (treatment_id)
           ON DELETE CASCADE;""",
        # billing.patient_name -> patients.name
        """ALTER TABLE billing
           ADD CONSTRAINT fk_billing_patient
           FOREIGN KEY (patient_name)
           REFERENCES patients (name)
           ON DELETE CASCADE;""",
        # appointments.patient_name -> patients.name
        """ALTER TABLE appointments
           ADD CONSTRAINT fk_appointments_patient
           FOREIGN KEY (patient_name)
           REFERENCES patients (name)
           ON DELETE CASCADE;""",
        # appointments.doctor_name -> doctors.name
        """ALTER TABLE appointments
           ADD CONSTRAINT fk_appointments_doctor
           FOREIGN KEY (doctor_name)
           REFERENCES doctors (name)
           ON DELETE CASCADE;""",
    ]

    for sql in fk_statements:
        try:
            cursor.execute(sql)
            print(f"[Step 6] Executed: {sql.splitlines()[0]} ...")
        except Exception as e:
            print(f"[Step 6][FK] Failed on: {sql.splitlines()[0]}\n  Reason: {e}")

    cursor.close()
    conn.close()
    print("[Step 6] Key constraints construction completed.")


def define_auxiliary(sql_file_path, dbname, port):
    conn = psycopg2.connect(
        f"host=localhost dbname={dbname} user=postgres password=postgres port={port}"
    )
    conn.autocommit = True
    cur = conn.cursor()

    print("[Step 7] Defining auxiliary tables .... ")
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

    print("[Step 7] Completed defining auxiliary tables.")

    cur.close()
    conn.close()


@hydra.main(
    version_base=None, config_path=ABS_CONFIG_DIR, config_name=DEFAULT_CONFIG_FILE_NAME
)
def main(cfg: DictConfig):
    args = cfg.sparta.db
    port = args.port
    dbname = args.db_name
    aux_file_path = os.path.join(cfg.root_dir_path, args.aux_file_path)
    data_dir_path = os.path.join(cfg.root_dir_path, args.data_dir_path)

    create_database(port)
    create_tables(port, dbname)
    insert_data_from_csv(port, dbname, data_dir_path)
    remove_columns(port, dbname)
    denormalize_names(port, dbname)
    construct_key_constraint(port, dbname)
    define_auxiliary(aux_file_path, dbname, port)  # for provenance


if __name__ == "__main__":
    main()
