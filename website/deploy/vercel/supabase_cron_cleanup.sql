-- Optional: run this file alone if you already ran supabase_setup.sql
-- and only need to add or reset the 48-hour purge cron job.

create extension if not exists pg_cron with schema extensions;

select cron.unschedule(jobid)
from cron.job
where jobname = 'purge-leaf-uploads-every-48h';

select cron.schedule(
  'purge-leaf-uploads-every-48h',
  '0 0 1-31/2 * *',
  $$
    delete from storage.objects
    where bucket_id = 'leaf-uploads';
  $$
);
