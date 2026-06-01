-- Supabase Storage: public leaf image previews + 48-hour purge cron
-- Run in Supabase Dashboard → SQL Editor (in order, or run this whole file once)

-- ---------------------------------------------------------------------------
-- 1. Enable pg_cron (also enable "pg_cron" under Database → Extensions)
-- ---------------------------------------------------------------------------
create extension if not exists pg_cron with schema extensions;

-- ---------------------------------------------------------------------------
-- 2. Public bucket for leaf upload previews
-- ---------------------------------------------------------------------------
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'leaf-uploads',
  'leaf-uploads',
  true,
  10485760,  -- 10 MB max per image
  array['image/jpeg', 'image/png', 'image/webp', 'image/gif']
)
on conflict (id) do update set
  public = true,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- ---------------------------------------------------------------------------
-- 3. Storage policies (public read, server uploads via service role)
-- ---------------------------------------------------------------------------
drop policy if exists "Public read leaf uploads" on storage.objects;
create policy "Public read leaf uploads"
on storage.objects for select
using (bucket_id = 'leaf-uploads');

drop policy if exists "Service role insert leaf uploads" on storage.objects;
create policy "Service role insert leaf uploads"
on storage.objects for insert
to service_role
with check (bucket_id = 'leaf-uploads');

drop policy if exists "Service role update leaf uploads" on storage.objects;
create policy "Service role update leaf uploads"
on storage.objects for update
to service_role
using (bucket_id = 'leaf-uploads');

drop policy if exists "Service role delete leaf uploads" on storage.objects;
create policy "Service role delete leaf uploads"
on storage.objects for delete
to service_role
using (bucket_id = 'leaf-uploads');

-- pg_cron runs as superuser and can delete without the policy above.

-- ---------------------------------------------------------------------------
-- 4. Cron: delete ALL leaf images every 48 hours (00:00 UTC every 2 days)
--    Schedule: minute 0, hour 0, every 2nd day of the month (1, 3, 5, …)
--    To change timing, edit the cron expression below.
-- ---------------------------------------------------------------------------
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

-- Verify the job was registered:
-- select jobid, jobname, schedule, command from cron.job where jobname = 'purge-leaf-uploads-every-48h';

-- Manual test (optional — deletes all current leaf uploads immediately):
-- delete from storage.objects where bucket_id = 'leaf-uploads';
