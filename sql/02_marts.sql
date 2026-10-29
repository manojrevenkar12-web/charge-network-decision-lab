-- Weekly site panel: only complete weeks (Monday to Sunday) a site was open all seven days.
create or replace table site_week as
select site_id,
       date_trunc('week', date)::date as week,
       count(*) as days,
       sum(sessions) as sessions,
       sum(kwh) as kwh,
       sum(revenue_eur) as revenue_eur,
       avg(price_eur_kwh) as price_eur_kwh,
       avg(availability) as availability
from site_day
group by all
having count(*) = 7;

-- The 2025 price increase: the first week each site paid the higher price (null if never).
create or replace table price_increase as
select site_id, min(valid_from) as first_day, date_trunc('week', min(valid_from))::date as first_week
from price_changes p
where valid_from > (select min(valid_from) from price_changes)
group by site_id;

-- Monthly energy by country and site, for forecasting.
create or replace table site_month as
select s.country, d.site_id, date_trunc('month', d.date)::date as month, sum(d.kwh) as kwh, sum(d.sessions) as sessions,
       count(*) as days
from site_day d join sites s using (site_id)
group by all;

create or replace table country_month as
select country, month, sum(kwh) as kwh, sum(sessions) as sessions from site_month group by all;

-- Trial: hourly outcomes rolled up per site and day, with the assignment.
create or replace table trial_day as
select h.site_id, h.date, a.discount is not null as in_trial, coalesce(a.discount, false) as discount,
       sum(h.sessions) as sessions,
       sum(h.kwh) as kwh,
       sum(h.kwh * h.price_eur_kwh) as revenue_eur,
       sum(case when h.hour in ({offpeak_hours}) then h.kwh else 0 end) / nullif(sum(h.kwh), 0) as offpeak_share,
       sum(case when h.hour not in ({offpeak_hours}) then h.sessions else 0 end) as peak_sessions
from trial_hourly h left join trial_assignment a using (site_id, date)
group by all;
