CREATE EXTENSION IF NOT EXISTS citus;
CREATE EXTENSION IF NOT EXISTS citus_columnar;

SET search_path TO fav, public;
DROP TABLE IF EXISTS fav.citus_sales CASCADE;
DROP TABLE IF EXISTS fav.citus_trans CASCADE;
DROP TABLE IF EXISTS fav.citus_items CASCADE;
DROP TABLE IF EXISTS fav.citus_stores CASCADE;
DROP TABLE IF EXISTS fav.citus_holidays CASCADE;
DROP TABLE IF EXISTS fav.citus_oil CASCADE;

-- =====================================================
-- Dimension tables --> reference tables
-- =====================================================

-- ITEMS
CREATE TABLE IF NOT EXISTS fav.citus_items (
    item_nbr INT PRIMARY KEY,
    family TEXT,
    class TEXT,
    perishable INT,
    f1 INT
);

SELECT create_reference_table('fav.citus_items');

INSERT INTO fav.citus_items
SELECT * FROM fav.items;


-- STORES
CREATE TABLE IF NOT EXISTS fav.citus_stores (
    store_nbr INT PRIMARY KEY,
    city TEXT,
    state TEXT,
    stype TEXT,
    cluster INT,
    f4 INT
);

SELECT create_reference_table('fav.citus_stores');

INSERT INTO fav.citus_stores
SELECT * FROM fav.stores;


-- HOLIDAYS
CREATE TABLE IF NOT EXISTS fav.citus_holidays (
    holiday_id INT PRIMARY KEY,
    date DATE,
    htype TEXT,
    locale TEXT,
    locale_name TEXT,
    description TEXT,
    transferred INTEGER,
    f2 INT
);

SELECT create_reference_table('fav.citus_holidays');

INSERT INTO fav.citus_holidays
SELECT * FROM fav.holidays;


-- OIL
CREATE TABLE IF NOT EXISTS fav.citus_oil (
    date DATE PRIMARY KEY,
    dcoilwtico NUMERIC,
    f3 INT
);

SELECT create_reference_table('fav.citus_oil');

INSERT INTO fav.citus_oil
SELECT * FROM fav.oil;


-- =====================================================
-- Columnar fact tables
-- =====================================================

-- TRANSACTIONS
CREATE TABLE IF NOT EXISTS fav.citus_trans (
    date DATE,
    store_nbr INT,
    transactions INT,
    f5 INT,
    PRIMARY KEY (store_nbr, date)
) USING columnar;

SELECT create_distributed_table(
    'fav.citus_trans',
    'store_nbr'
);

INSERT INTO fav.citus_trans
SELECT
    date,
    store_nbr,
    transactions,
    f5
FROM fav.trans;


-- SALES (main fact)
CREATE TABLE IF NOT EXISTS fav.citus_sales (
    id BIGINT,
    date DATE,
    store_nbr INT,
    item_nbr INT,
    unit_sales NUMERIC,
    onpromotion INT,
    target NUMERIC,
    PRIMARY KEY (store_nbr, date, id)
) 
USING columnar
WITH (compression = 'zstd', compression_level = 3, stripe_row_limit = 2000000, chunk_group_row_limit = 100000);;

SELECT create_distributed_table(
    'fav.citus_sales',
    'store_nbr'
);

INSERT INTO fav.citus_sales
SELECT * FROM fav.sales;

-- =====================================================
-- Indexes and optimisation
-- =====================================================

CREATE INDEX IF NOT EXISTS citus_trans_date_idx
ON fav.citus_trans (date);

CREATE INDEX IF NOT EXISTS citus_sales_date_idx
ON fav.citus_sales (date);

CREATE INDEX IF NOT EXISTS citus_sales_item_idx
ON fav.citus_sales (item_nbr);

ALTER TABLE fav.citus_sales
SET (columnar.compression = 'zstd');

ALTER TABLE fav.citus_trans
SET (columnar.compression = 'zstd');