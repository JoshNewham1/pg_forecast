# Usage Demo
0. Populate table
```sql
CREATE TABLE IF NOT EXISTS time_series_data (
    t TIMESTAMP WITHOUT TIME ZONE PRIMARY KEY,
    value DOUBLE PRECISION NOT NULL
);
TRUNCATE TABLE time_series_data;
INSERT INTO time_series_data (t, value) VALUES
('2026-01-01 00:00:00', 128.0),
('2026-01-01 00:00:01', 64.0),
('2026-01-01 00:00:02', 32.0),
('2026-01-01 00:00:03', 16.0),
('2026-01-01 00:00:04', 8.0),
('2026-01-01 00:00:05', 4.0),
('2026-01-01 00:00:06', 2.0),
('2026-01-01 00:00:07', 1.0),
('2026-01-01 00:00:08', 0.5),
('2026-01-01 00:00:09', 0.25);
```
1. Enable timing & show the `time_series_data` table:
```
\timing
```

```sql
SELECT * FROM time_series_data;
```

2. Create a forecast:
```sql
SELECT create_forecast(
    model_name   => 'autoarima',
    input_table  => 'time_series_data',
    date_column  => 't',
    value_column => 'value'
);
```

3. Run a forecast
```sql
SELECT * FROM 
run_forecast(
    model_name    => 'autoarima',
    input_table   => 'time_series_data',
    date_column   => 't',
    value_column  => 'value',
    horizon       => 14,
    forecast_step => '1 second'
);
```

4. Insert an "expected" value
```sql
INSERT INTO time_series_data(t, value)
VALUES ('2026-01-01 00:00:10', 0.125);
```

5. Insert an "unexpected" value
```sql
INSERT INTO time_series_data(t, value)
VALUES ('2026-01-01 00:00:11', 10.0);
```

_Cleanup_
```sql
DELETE FROM time_series_data WHERE t > '2026-01-01 00:00:09';
SELECT remove_forecast('autoarima', 'time_series_data', 't', 'value');
```

_(Optional): Show vertices_
```bash
sudo -u postgres psql -d diss
```

```sql
SELECT arima_id, vertex_id, phi, theta, (incremental_state).css AS css
FROM arima_vertices;
```

# Performance Test Demo

1. Open Power BI live report

2. Run performance stress test and visualise it (single inserts, 50k rows)
```bash
pytest "tests/python/performance_tests.py" -o log_cli=true -o log_cli_level="DEBUG" -k "pgforecast and single"
```