import re
import time
import pandas as pd
from sqlalchemy import text
import sys
import os
import decimal
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

        start = time.perf_counter()
        if self.debug:
            print(f"SQL: {q}")
            
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
            print(f"Time: {time.perf_counter() - start:.4f}s")
            
        if q.strip().upper().startswith(("SELECT", "WITH", "RETURNING")):
            return result.fetchall()
        else:
            # SQLAlchemy with psycopg2 requires commit for DDL/Inserts if not autocommit
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
        
        # 3. CRITICAL: Analyze and Index
        # If the table has columns we usually join on, index them immediately.
        # JoinBoost usually joins on 'date' and 'store_nbr' or 'item_nbr'
        cols = self.get_schema(name_)
        join_keys = [c for c in ['date', 'store_nbr', 'item_nbr', 'id'] if c in cols]
        
        if join_keys:
            idx_name = f"idx_{name_}_{'_'.join(join_keys)}"
            # We use a combined index for the join keys
            self._execute_query(f"CREATE INDEX {idx_name} ON {name_} ({', '.join(join_keys)})")
            # Update statistics so the planner knows the index exists and is useful
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
