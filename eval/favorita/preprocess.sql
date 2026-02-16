-- ==========================================
-- Create tables with an extra random_feature for preprocessing
-- ==========================================
CREATE SCHEMA IF NOT EXISTS fav;
-- DROP TABLE IF EXISTS fav.holidays CASCADE;
-- DROP TABLE IF EXISTS fav.items CASCADE;
-- DROP TABLE IF EXISTS fav.oil CASCADE;
-- DROP TABLE IF EXISTS fav.stores CASCADE;
-- DROP TABLE IF EXISTS fav.trans CASCADE;
-- DROP TABLE IF EXISTS fav.sales CASCADE;

CREATE TABLE fav.items (
    item_nbr INT PRIMARY KEY,
    family TEXT,
    class TEXT,
    perishable INT,
    f1 INT
);

CREATE TABLE fav.holidays (
    holiday_id SERIAL PRIMARY KEY,
    date DATE,
    htype TEXT,
    locale TEXT,
    locale_name TEXT,
    description TEXT,
    transferred BOOLEAN,
    f2 INT
);

CREATE TABLE fav.oil (
    date DATE PRIMARY KEY,
    dcoilwtico NUMERIC,
    f3 INT
);

CREATE TABLE fav.stores (
    store_nbr INT PRIMARY KEY,
    city TEXT,
    state TEXT,
    stype TEXT,
    cluster INT,
    f4 INT
);

CREATE TABLE fav.trans (
    date DATE,
    store_nbr INT REFERENCES fav.stores(store_nbr),
    transactions INT,
    f5 INT,
    PRIMARY KEY(date, store_nbr)
);

CREATE TABLE fav.sales (
    id INT,
    date DATE,
    store_nbr INT REFERENCES fav.stores(store_nbr),
    item_nbr INT REFERENCES fav.items(item_nbr),
    unit_sales NUMERIC,
    onpromotion BOOLEAN,
    target NUMERIC,
    CONSTRAINT sales_pk PRIMARY KEY (id, date)
);

CREATE INDEX IF NOT EXISTS idx_trans_split ON fav.trans (date, store_nbr, transactions);
CREATE INDEX IF NOT EXISTS idx_holidays_date ON fav.holidays (date);
CREATE INDEX IF NOT EXISTS idx_sales_lookup ON fav.sales (date, store_nbr, item_nbr);

-- -- ==========================================
-- -- Load CSVs
-- -- ==========================================
COPY fav.holidays(date,htype,locale,locale_name,description,transferred)
FROM '/home/josh/diss/eval/favorita/data/holidays.csv' DELIMITER ',' CSV HEADER;

COPY fav.items(item_nbr,family,class,perishable)
FROM '/home/josh/diss/eval/favorita/data/items.csv' DELIMITER ',' CSV HEADER;

COPY fav.oil(date,dcoilwtico)
FROM '/home/josh/diss/eval/favorita/data/oil.csv' DELIMITER ',' CSV HEADER;

COPY fav.stores(store_nbr,city,state,stype,cluster)
FROM '/home/josh/diss/eval/favorita/data/stores.csv' DELIMITER ',' CSV HEADER;

COPY fav.trans(date,store_nbr,transactions)
FROM '/home/josh/diss/eval/favorita/data/trans.csv' DELIMITER ',' CSV HEADER;

COPY fav.sales(id,date,store_nbr,item_nbr,unit_sales,onpromotion)
FROM '/home/josh/diss/eval/favorita/data/sales.csv' DELIMITER ',' CSV HEADER;

-- ==========================================
-- Preprocessing
-- ==========================================
-- Set seed for reproducibility
SELECT setseed(0.42);
-- Fill in transactions table for dates with no transactions
INSERT INTO fav.trans(date, store_nbr, transactions, f5)
SELECT
    s.date,
    s.store_nbr,
    0 AS transactions,
    floor(random()*1000 + 1)::INT as f5
FROM fav.sales s
WHERE NOT EXISTS (
    SELECT 1 
    FROM fav.trans t 
    WHERE 
        t.date = s.date 
        AND t.store_nbr = s.store_nbr 
    ORDER BY t.date
) 
GROUP BY s.date, s.store_nbr;

-- Impute one feature per table with random integers [1,1000]
DO $$
BEGIN
    UPDATE fav.items SET f1 = floor(random()*1000 + 1)::INT;
    UPDATE fav.holidays SET f2 = floor(random()*1000 + 1)::INT;
    UPDATE fav.oil SET f3 = floor(random()*1000 + 1)::INT;
    UPDATE fav.stores SET f4 = floor(random()*1000 + 1)::INT;
    UPDATE fav.trans SET f5 = floor(random()*1000 + 1)::INT;
END $$;

-- ==========================================
-- Automatically dictionary encode all TEXT columns
-- ==========================================
DO $$
DECLARE
    tbl RECORD;
    col RECORD;
    dict_table TEXT;
    sql_text TEXT;
BEGIN
    -- Loop over all user tables
    FOR tbl IN
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema='fav' AND table_type='BASE TABLE' AND table_name NOT LIKE '%_dict' AND table_name NOT LIKE '%_di'
    LOOP
        -- Loop over TEXT columns in the table
        FOR col IN
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = tbl.table_name
              AND data_type = 'text'
        LOOP
            -- Create dictionary table dynamically
            dict_table := tbl.table_name || '_' || col.column_name || '_dict';
            sql_text := format('DROP TABLE IF EXISTS fav.%I', dict_table);
            EXECUTE sql_text;

            sql_text := format(
                'CREATE TABLE fav.%I AS SELECT %I AS original_value, row_number() OVER () AS dict_id FROM (SELECT DISTINCT %I FROM fav.%I) t',
                dict_table, col.column_name, col.column_name, tbl.table_name
            );
            EXECUTE sql_text;

            -- Update original table to use dictionary ID
            sql_text := format(
                'UPDATE fav.%I AS t SET %I = d.dict_id FROM fav.%I AS d WHERE t.%I = d.original_value',
                tbl.table_name, col.column_name, dict_table, col.column_name
            );
            RAISE NOTICE '%', sql_text;
            EXECUTE sql_text;
        END LOOP;
    END LOOP;
END $$;

-- ==========================================
-- Automatically encode BOOLEAN columns as 0/1
-- ==========================================
DO $$
DECLARE
    tbl RECORD;
    col RECORD;
    sql_text TEXT;
BEGIN
    FOR tbl IN
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema='fav' AND table_type='BASE TABLE'
    LOOP
        FOR col IN
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = tbl.table_name
              AND data_type = 'boolean'
        LOOP
            sql_text := format(
                'ALTER TABLE fav.%I
                ALTER COLUMN %I TYPE int
                USING %I::int;',
                tbl.table_name, col.column_name, col.column_name
            );
            EXECUTE sql_text;
        END LOOP;
    END LOOP;
END $$;

-- ==========================================
-- Compute target variable for sales
-- y = log(unit_sales) + log(oil_price) - 10*store_random - 10*day_of_year + trans_random^2
-- ==========================================
UPDATE fav.sales s
SET target = sub.target
FROM (
    SELECT 
        s.id,
        COALESCE(LOG(ABS(s.unit_sales) + 1),0)
      + COALESCE(LOG(ABS(o.dcoilwtico) + 1),0)
      - 10 * COALESCE(st.f4,0)
      - 10 * EXTRACT(DOY FROM s.date)
      + COALESCE(t.f5,0)^2 AS target
    FROM fav.sales s
    INNER JOIN fav.stores st ON s.store_nbr = st.store_nbr
    LEFT JOIN fav.trans t ON t.date = s.date AND t.store_nbr = s.store_nbr
    LEFT JOIN fav.oil o ON o.date = s.date
) AS sub
WHERE s.id = sub.id;
