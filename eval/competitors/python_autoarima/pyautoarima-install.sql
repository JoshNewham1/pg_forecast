/*
    pyautoarima assumes you have the following packages installed
    in the system's Python instance or in a virtualenv that has
    been referenced in `pyautoarima_config` below

    * pmdarima
    * numpy
*/

BEGIN;

-- =========================================================
--  Register extensions
-- =========================================================
CREATE EXTENSION IF NOT EXISTS plpython3u;

-- =========================================================
--  Result types
-- =========================================================

CREATE TYPE pyautoarima_fit_result AS (
    p INT,
    d INT,
    q INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    css DOUBLE PRECISION
);

CREATE TYPE pyautoarima_incremental_state AS (
    p INT, -- AR order
    q INT, -- MA order
    e_lags DOUBLE PRECISION[], -- Last q residuals
    last_value DOUBLE PRECISION, -- Last observed value (for difference handling)
    css DOUBLE PRECISION
);

-- =========================================================
--  Tables
-- =========================================================

CREATE TABLE model_pyautoarima_stats (
    id BIGSERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    d INT,
    c DOUBLE PRECISION,
    incremental_state pyautoarima_incremental_state,
    is_incremental BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN,
    CONSTRAINT unique_model_pyautoarima_stats UNIQUE (model_id, phi, theta, d, c)
);

CREATE TABLE pyautoarima_vertices (
    id BIGSERIAL PRIMARY KEY,
    stats_id BIGINT NOT NULL REFERENCES model_pyautoarima_stats(id) ON DELETE CASCADE,
    vertex_index INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    incremental_state pyautoarima_incremental_state
);

CREATE TABLE pyautoarima_config (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- =========================================================
--  Python virtualenv
-- =========================================================

INSERT INTO pyautoarima_config (key, value)
VALUES ('venv', '/home/josh/diss/.venv/lib/python3.12/site-packages');

CREATE OR REPLACE FUNCTION activate_python_venv()
RETURNS VOID AS
$$
venv = plpy.execute("SELECT value FROM pyautoarima_config WHERE key='venv'")[0]['value']

from sys import path
path.append(venv)
$$
LANGUAGE plpython3u VOLATILE;

-- =========================================================
-- Incremental ARIMA update functions
-- =========================================================
CREATE OR REPLACE FUNCTION pyautoarima_generate_vertices(
    stats_id BIGINT,
    p INT,
    q INT,
    d_order INT,
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    c DOUBLE PRECISION,
    tolerance DOUBLE PRECISION DEFAULT 0.05
)
RETURNS VOID
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
import numpy as np

arr_phi = np.array(phi if phi else [], dtype=float)
arr_theta = np.array(theta if theta else [], dtype=float)
params = np.concatenate([arr_phi, arr_theta])
dim = len(params)

if dim == 0:
    return

vertices = []

# Permutations of verties
for i in range(dim):
    p_new = params.copy()
    sign = 1.0 if p_new[i] >= 0 else -1.0
    p_new[i] += sign * tolerance
    vertices.append(p_new)

# Contraction for (d+1)th vertex
norm = np.linalg.norm(params)
scale_factor = 0.5
if norm > tolerance:
    scale_factor = 1.0 - (tolerance / norm)
p_contracted = params * scale_factor
vertices.append(p_contracted)

# Insert all vertices
for idx, v_params in enumerate(vertices):
    new_phi = list(v_params[:len(arr_phi)])
    new_theta = list(v_params[len(arr_phi):])
    
    init_lags = [0.0] * q
    
    plpy.execute(plpy.prepare(
        """
        INSERT INTO pyautoarima_vertices 
        (stats_id, vertex_index, phi, theta, incremental_state)
        VALUES ($1, $2, $3, $4, ($5, $6, $7, $8, 0.0)::pyautoarima_incremental_state)
        """,
        ["bigint", "int", "double precision[]", "double precision[]", "int", "int", "double precision[]", "double precision"]
    ), [stats_id, idx + 1, new_phi, new_theta, p, q, init_lags, 0.0])
    with open("pyautoarima.log", "a") as f:
        f.writelines([f"pyautoarima: Created vertex {idx+1} with phi {new_phi} and theta {new_theta}\n"])
$$;

CREATE OR REPLACE FUNCTION pyautoarima_incremental_update(
    stats_id BIGINT,
    is_incremental BOOLEAN,
    new_y DOUBLE PRECISION
)
RETURNS BOOLEAN
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
import numpy as np

def step_arima(val, state, phi, theta, d, c):
    p, q = state['p'], state['q']
    e_lags = list(state['e_lags']) if state['e_lags'] else [0.0]*q
    last_val = state['last_value'] if state['last_value'] is not None else 0.0
    css = state['css'] if state['css'] is not None else 0.0
    
    y_diff = val
    if d > 0:
        y_diff = val - last_val
        
    yhat = c
    if p > 0:
        # Simple AR(1) approximation for incremental context
        yhat += sum(phi[i] * last_val for i in range(min(p, 1))) 
    if q > 0:
        yhat += sum(theta[i] * e_lags[i] for i in range(q))
        
    e = y_diff - yhat
    
    if q > 0:
        e_lags = [e] + e_lags[:-1]

    new_css = css + (e**2)
    
    return {
        'p': p, 'q': q, 
        'e_lags': e_lags, 
        'last_value': val, 
        'css': new_css
    }, e

# Update centre
res = plpy.execute(f"SELECT * FROM model_pyautoarima_stats WHERE id = {stats_id}")
if not res: return True
centre = res[0]

new_c_state, innovation = step_arima(
    new_y, centre['incremental_state'], 
    centre['phi'], centre['theta'], centre['d'], centre['c']
)

is_ok = True

if not is_incremental:
    return False      
else:
    v_rows = plpy.execute(f"SELECT * FROM pyautoarima_vertices WHERE stats_id = {stats_id}")
    min_vertex_css = float('inf')
    
    for row in v_rows:
        new_v_state, _ = step_arima(
            new_y, row['incremental_state'], 
            row['phi'], row['theta'], centre['d'], centre['c']
        )
        if new_v_state['css'] < min_vertex_css:
            min_vertex_css = new_v_state['css']
            
        plpy.execute(plpy.prepare(
            "UPDATE pyautoarima_vertices SET incremental_state = $1 WHERE id = $2",
            ["pyautoarima_incremental_state", "bigint"]
        ), [new_v_state, row['id']])
        
    if min_vertex_css < new_c_state['css']:
        is_ok = False

# Save centre
plpy.execute(plpy.prepare(
    "UPDATE model_pyautoarima_stats SET incremental_state = $1 WHERE id = $2",
    ["pyautoarima_incremental_state", "bigint"]
), [new_c_state, stats_id])

return is_ok
$$;

-- =========================================================
--  Python AutoARIMA fit (thin wrapper)
-- =========================================================
-- This function is called by pyautoarima_train below

CREATE OR REPLACE FUNCTION pyautoarima_fit_series(
    y DOUBLE PRECISION[]
)
RETURNS pyautoarima_fit_result
LANGUAGE plpython3u
AS $$
plpy.execute("SELECT activate_python_venv();")
import pmdarima as pm
import numpy as np

model = pm.auto_arima(
    y,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5, max_q=5, max_d=2,
)

p, d, q = model.order
residuals = model.resid()
css = np.sum(residuals ** 2)

with open("pyautoarima.log", "a") as f:
    f.writelines([f"Trained pyautoarima with {len(y)} records, p = {p}, d = {d}, q = {q}, css = {css}\n"])

phi = []
theta = []
c = 0.0

for name, val in zip(model.arima_res_.param_names, model.arima_res_.params):
    if name in ("const", "intercept"):
        c = float(val)
    elif name.startswith("ar.L"):
        phi.append(float(val))
    elif name.startswith("ma.L"):
        theta.append(float(val))

return (p, d, q, phi, theta, c, css)
$$;

-- =========================================================
--  Train & register AutoARIMA model
-- =========================================================

CREATE OR REPLACE FUNCTION pyautoarima_create(
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    use_continuous_agg BOOLEAN DEFAULT FALSE,
    is_incremental BOOLEAN DEFAULT FALSE
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_model_id BIGINT;
    v_fit pyautoarima_fit_result;
    v_stats_id BIGINT;
    v_view_name TEXT;
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    -- Register model
    INSERT INTO models (model_type, input_table, date_column, value_column)
    VALUES ('pyautoarima', input_table, date_column, value_column)
    RETURNING id INTO v_model_id;

    v_fit := pyautoarima_train(input_table, date_column, value_column, is_incremental);

    IF is_incremental THEN
        SELECT s.id FROM model_pyautoarima_stats s
        INTO v_stats_id
        WHERE s.model_id = v_model_id;

        PERFORM pyautoarima_generate_vertices(
            v_stats_id, v_fit.p, v_fit.q, v_fit.d, v_fit.phi, v_fit.theta, v_fit.c
        );
    END IF;


    -- Optional: use TimescaleDB continuous aggregate
    IF use_continuous_agg THEN
        v_view_name := format(
            'pyautoarima_cagg_%s_%s',
            input_table, value_column
        );

        EXECUTE format(
            'CREATE MATERIALIZED VIEW IF NOT EXISTS %I
             WITH (timescaledb.continuous)
             AS SELECT
                time_bucket(''1 day'', %I) AS bucket,
                avg(%I) AS value
             FROM %I
             GROUP BY bucket;',
            v_view_name, date_column, value_column, input_table
        );
    END IF;

    -- Create trigger
    EXECUTE format(
        'CREATE TRIGGER pyautoarima_on_insert_%I_%I
         AFTER INSERT ON %I
         REFERENCING NEW TABLE AS new_rows
         FOR EACH STATEMENT
         EXECUTE FUNCTION pyautoarima_on_insert_trigger(%L, %L, %L, %L);',
        input_table, value_column,
        input_table,
        input_table, date_column, value_column, is_incremental
    );

    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION pyautoarima_train(
    input_tab TEXT,
    date_col TEXT,
    value_col TEXT,
    is_incremental BOOLEAN
)
RETURNS pyautoarima_fit_result
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_model_id BIGINT;
    v_series DOUBLE PRECISION[];
    v_fit pyautoarima_fit_result;
    v_incremental_state pyautoarima_incremental_state;
BEGIN
    PERFORM set_config('search_path', 'public,pg_temp', true);

    -- Extract series
    v_series := series_to_array(input_tab, date_col, value_col, TRUE);

    -- Fit AutoARIMA
    v_fit := pyautoarima_fit_series(v_series);

    -- Register model
    SELECT m.id FROM models m INTO v_model_id
    WHERE m.model_type = 'pyautoarima' 
        AND m.input_table = input_tab
        AND m.date_column = date_col
        AND m.value_column = value_col;

    v_incremental_state := (v_fit.p, v_fit.q, array_fill(0.0, ARRAY[v_fit.q]), 0.0, 0.0);

    -- Store ARIMA stats + incremental state
    UPDATE model_pyautoarima_stats
    SET is_active = FALSE
    WHERE model_id = v_model_id;
    
    INSERT INTO model_pyautoarima_stats (
        model_id, phi, theta, d, c, incremental_state, "is_incremental", is_active
    )
    VALUES (
        v_model_id,
        v_fit.phi,
        v_fit.theta,
        v_fit.d,
        v_fit.c,
        v_incremental_state,
        is_incremental,
        TRUE
    )
    ON CONFLICT ON CONSTRAINT unique_model_pyautoarima_stats
    DO UPDATE SET incremental_state = v_incremental_state, is_active = TRUE;
    
    RETURN v_fit;
END;
$$;

-- =========================================================
--  Incremental INSERT trigger (geometric breach detection)
-- =========================================================

CREATE OR REPLACE FUNCTION pyautoarima_on_insert_trigger()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_input_table TEXT := TG_ARGV[0];
    v_date_column TEXT := TG_ARGV[1];
    v_value_column TEXT := TG_ARGV[2];
    v_is_incremental BOOLEAN := TG_ARGV[3];
    v_model RECORD;
    v_ok BOOLEAN;
    v_value DOUBLE PRECISION;
BEGIN
    -- Security precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    -- Load model
    SELECT
        m.id,
        s.id AS stats_id,
        (s.incremental_state).p,
        s.d,
        (s.incremental_state).q,
        s.phi, s.theta, s.c,
        s.incremental_state
    INTO v_model
    FROM models m
    INNER JOIN model_pyautoarima_stats s ON s.model_id = m.id
    WHERE
        m.model_type = 'pyautoarima'
        AND m.input_table = v_input_table
        AND m.date_column = v_date_column
        AND m.value_column = v_value_column
        AND s.is_active = TRUE
    LIMIT 1;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    IF v_model.p + v_model.q = 0 THEN
        v_is_incremental := FALSE;
    END IF;

    IF v_is_incremental AND NOT EXISTS (SELECT 1 FROM pyautoarima_vertices WHERE stats_id = v_model.stats_id) THEN
        PERFORM pyautoarima_generate_vertices(
            v_model.stats_id, v_model.p, v_model.q, v_model.d, v_model.phi, v_model.theta, v_model.c
        );
    END IF; 

    -- Apply incremental updates in timestamp order
    FOR v_value IN
        EXECUTE format(
            'SELECT %I FROM new_rows ORDER BY %I',
            v_value_column,
            v_date_column
        )
    LOOP
        v_ok := pyautoarima_incremental_update(
            v_model.stats_id,
            v_is_incremental,
            v_value
        );

        -- Stop early if geometry breached
        IF NOT v_ok THEN
            EXIT;
        END IF;
    END LOOP;

    -- Retrain once if breached
    IF NOT v_ok THEN
        PERFORM pyautoarima_train(
            v_input_table,
            v_date_column,
            v_value_column,
            v_is_incremental
        );
    END IF;

    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION pyautoarima_forecast(
    input_table TEXT,
    date_column TEXT,
    value_column TEXT,
    horizon INT,
    forecast_step INTERVAL DEFAULT '1 day'
)
RETURNS TABLE(forecast_date TIMESTAMP, forecast_value DOUBLE PRECISION)
LANGUAGE plpython3u
SECURITY DEFINER
AS $$
plpy.execute("SELECT activate_python_venv();")
import numpy as np

# -------------------------------
# Load model
# -------------------------------
sql = f"""
SELECT 
    (s.incremental_state).p,
    s.d,
    (s.incremental_state).q,
    s.phi, s.theta, s.c,
    s.incremental_state
FROM models m
JOIN model_pyautoarima_stats s ON s.model_id = m.id
WHERE m.model_type = 'pyautoarima'
  AND m.input_table = {input_table!r}
  AND m.date_column = {date_column!r}
  AND m.value_column = {value_column!r}
  AND s.is_active = TRUE
LIMIT 1
"""
model_row = plpy.execute(sql)
if not model_row:
    plpy.error(f"No trained pyautoarima model found for {input_table}, {date_column}, {value_column}")
m = model_row[0]

p, d, q = m["p"], m["d"], m["q"]
phi, theta, c = m["phi"], m["theta"], m["c"]
residuals = m["incremental_state"]["e_lags"]

# -------------------------------
# Load time series
# -------------------------------
arr_sql = f"""
SELECT array_agg({value_column} ORDER BY {date_column}) AS vals
FROM {input_table}
WHERE {value_column} IS NOT NULL
"""
arr_row = plpy.execute(arr_sql)
vals = arr_row[0]["vals"]
if not vals or len(vals) < 2:
    plpy.error(f"No data available for forecasting in {input_table}.{value_column}")

vals = np.array(vals, dtype=float)

# -------------------------------
# Difference if needed
# -------------------------------
if d > 0:
    initial_vals = vals[:d].copy()
    for i in range(d):
        vals[i+1:] = vals[i+1:] - vals[i:-1].copy()

# -------------------------------
# Select last values for forecast
# -------------------------------
ncond = max(p, q)
last_vals = vals[-ncond:] if ncond > 0 else np.array([], dtype=float)

# -------------------------------
# Forecast
# -------------------------------
forecast = np.zeros(horizon, dtype=float)

for t in range(horizon):
    yhat = c
    if p > 0:
        yhat += sum(phi[i]*last_vals[-(i+1)] for i in range(p))
    if q > 0:
        yhat += sum(theta[i]*residuals[-(i+1)] for i in range(q))
    eps = 0  # assume zero innovation for forecast
    forecast[t] = yhat
    # Shift last_vals and residuals
    if p > 0:
        last_vals = np.append(last_vals[1:], yhat)
    if q > 0:
        residuals = np.append(residuals[1:], eps)

# -------------------------------
# Re-integrate if differenced
# -------------------------------
if d > 0:
    for i in range(horizon):
        forecast[i] += vals[-d+i] if i < d else forecast[i-1]

# -------------------------------
# Compute forecast dates
# -------------------------------
date_sql = f"SELECT MAX({date_column}) AS last_date FROM {input_table}"
last_date = plpy.execute(date_sql)[0]["last_date"]
arr_forecasts = "{" + ",".join(str(float(v)) for v in forecast) + "}"

return plpy.execute(f"""
    SELECT
        '{last_date}'::timestamp + (i * '{forecast_step}'::interval) AS forecast_date,
        ('{arr_forecasts}'::double precision[])[i] AS forecast_value
    FROM generate_series(1, {horizon}) AS i
""")
$$;

COMMIT;