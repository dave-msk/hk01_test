# Q1

The user count can be extracted as follows on MySQL:

```sql
select count(*) from (
    select distinct uid from piwik_track as pt 
    where pt.uid in (
        select uid from piwik_track
        where date(time) = '2017-04-01'
        and event_name = 'FIRST_INSTALL')
    and date(pt.time) between date('2017-04-02') and date('2017-04-08')
    group by pt.uid) as t;
```
