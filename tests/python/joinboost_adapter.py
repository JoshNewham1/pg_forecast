import re
import time
from sqlalchemy import text
import sys
import os
import psycopg2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eval/JoinBoost/src")))
from joinboost.executor import DuckdbExecutor, SPJAData

class PostgresExecutor(DuckdbExecutor):
    def __init__(self, conn, debug=False):
        super().__init__(conn, debug)

        # Convert any numeric types to Python float and not Decimal
        DEC2FLOAT = psycopg2.extensions.new_type(
            psycopg2.extensions.DECIMAL.values,
            'DEC2FLOAT',
            lambda value, curs: float(value) if value is not None else None)
        psycopg2.extensions.register_type(DEC2FLOAT)

        self.prefix = "jb_tmp_"
        
    def get_schema(self, table: str) -> list:
        schema = 'public'
        name = table
        if '.' in table:
            schema, name = table.split('.')
            
        sql = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = :name AND table_schema = :schema
        """
        res = self.conn.execute(text(sql), {"name": name, "schema": schema}).fetchall()
        return [r[0] for r in res]

    def _execute_query(self, q):
        # Type Compatibility Fixes (Fast Replace)
        q = q.replace("AS DOUBLE", "AS DOUBLE PRECISION")
        
        # Fix Tuple IN Syntax: (a,b) IN (SELECT (a,b)...) -> (a,b) IN (SELECT a,b...)
        if "SELECT (" in q.upper():
            q = re.sub(r'(?i)IN\s*\(\s*SELECT\s*\(([^)]+)\)\s+FROM', r'IN (SELECT \1 FROM', q)

        # Optimisation (assuming no NULLs in joins)
        q = q.replace('IS NOT DISTINCT FROM', '=')

        # CTAS Rewrite for MVCC Avoidance (performs CASE in memory instead of in UPDATE)
        if "UPDATE" in q.upper() and "JB_TMP_0" in q.upper() and "SET S =" in q.upper():
            # Extract the math inside the UPDATE statement
            case_statement = q[q.upper().find("(CASE"):].rstrip("; ")

            # Fetch column list dynamically (excluding s)
            schema = self.conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'jb_tmp_0' AND table_schema = current_schema()
                ORDER BY ordinal_position
            """))
            cols = [r[0] for r in schema.fetchall()]
            non_s_cols = [c for c in cols if c.lower() != "s"]
            select_cols = ", ".join(non_s_cols)
            
            # jb_tmp_0 schema from logs: s, c, item_nbr, date, store_nbr
            q = f"""
            BEGIN;
            CREATE UNLOGGED TABLE jb_tmp_0_new AS 
            SELECT {select_cols}, 
                   s - {case_statement} AS s
            FROM jb_tmp_0;
            
            DROP TABLE jb_tmp_0 CASCADE;
            ALTER TABLE jb_tmp_0_new RENAME TO jb_tmp_0;
            COMMIT;
            """

        start = time.perf_counter()
            
        try:
            result = self.conn.execute(text(q))
        except Exception as e:
            # Fallback for strict typing issues if standard execution fails
            print(f"Query failed, attempting CAST fix. Error: {e}")
            # Heavy regex only on failure
            q = re.sub(r'([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)\s*(<=|>=|<|>|=)\s*(\d+)(?!\.)', 
                       r'CAST(\1 AS DOUBLE PRECISION) \2 \3', q)
            result = self.conn.execute(text(q))

        if self.debug:
            with open("joinboost.log", "a") as f:
                f.write(f"SQL: {q} \n")
                f.write(f"Time: {time.perf_counter() - start:.4f}s \n")
            
        if q.strip().upper().startswith(("SELECT", "WITH", "RETURNING")):
            return result.fetchall()
        else:
            if not self.conn.in_transaction():
                self.conn.commit()
            return None

    def _spja_query_to_table(self, spja_data: SPJAData) -> str:
        # 1. Clean up the query string
        spja = self.spja_query(spja_data, parenthesize=False)
        
        # Fast-path: Convert 'IS NOT DISTINCT FROM' to '=' for Postgres performance
        # JoinBoost uses this for NULL safety, but it's a performance killer here.
        spja = spja.replace("IS NOT DISTINCT FROM", "=")
        
        name_ = self.get_next_name()
        
        if spja_data.replace:
            self._execute_query(f"DROP TABLE IF EXISTS {name_} CASCADE")
        
        # 2. Use UNLOGGED for speed (no WAL overhead)
        sql = f"CREATE UNLOGGED TABLE {name_} AS {spja}"
        self._execute_query(sql)
        
        # Building B-Trees on these temp tables forces Postgres into Nested Loops 
        # and kills performance on 80M rows. We rely on Hash Joins via work_mem instead.
        self._execute_query(f"ANALYZE {name_}")
            
        return name_

    def spja_query(self, spja_data: SPJAData, parenthesize: bool = True):
        # Intercept the query generation to inject Postgres-specific syntax
        # Specifically for Sampling which differs between DuckDB and Postgres
        
        sql = super().spja_query(spja_data, parenthesize=False)
        
        # If JoinBoost generated `random()` for join sampling, Postgres needs `random()`. 
        # DuckDB sometimes uses `drandom()`.
        sql = sql.replace("drandom()", "random()")
        
        if parenthesize:
            return f"({sql})"
        return sql
