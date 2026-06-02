-- Reset the 48-hour leaf-uploads purge cron (Storage API, not direct SQL DELETE).
-- Run in Supabase Dashboard → SQL Editor.
--
-- Prerequisites:
--   1. pg_cron enabled (Database → Extensions)
--   2. pg_net enabled (Database → Extensions)
--   3. Vault secrets created — run supabase_purge_function.sql first (includes vault + function)

create extension if not exists pg_cron with schema extensions;

select cron.unschedule(jobid)
from cron.job
where jobname = 'purge-leaf-uploads-every-48h';

select cron.schedule(
  'purge-leaf-uploads-every-48h',
  '0 0 1-31/2 * *',
  $$select public.purge_leaf_uploads_storage();$$
);

-- Verify:
-- select jobid, jobname, schedule, command from cron.job where jobname = 'purge-leaf-uploads-every-48h';

-- Manual test (deletes all current leaf uploads via Storage API):
-- select public.purge_leaf_uploads_storage();
