-- Raw files as tables, with the types the contracts promise.
create or replace table sites as
select site_id, country, segment, charger_type, points::int as points, opened::date as opened,
       lat, lon, traffic_index, bev_density_index, grid_kw
from read_parquet('{data}/sites.parquet');

create or replace table site_day as
select site_id, date::date as date, sessions::int as sessions, kwh, revenue_eur, price_eur_kwh, availability
from read_parquet('{data}/daily.parquet');

create or replace table price_changes as
select site_id, valid_from::date as valid_from, price_eur_kwh from read_parquet('{data}/prices.parquet');

create or replace table competitor_events as
select site_id, opened::date as opened, operator from read_parquet('{data}/competitor_events.parquet');

create or replace table trial_hourly as
select site_id, date::date as date, hour::int as hour, sessions::int as sessions, kwh, price_eur_kwh, discount_day
from read_parquet('{data}/trial_hourly.parquet');

create or replace table trial_assignment as
select site_id, date::date as date, discount from read_parquet('{data}/trial_assignment.parquet');

create or replace table candidates as select * from read_parquet('{data}/candidates.parquet');
