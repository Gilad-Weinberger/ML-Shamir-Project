-- Supabase Storage: public leaf image previews + 48-hour purge cron
-- Run in Supabase Dashboard → SQL Editor (in order, or run this whole file once)

-- ---------------------------------------------------------------------------
-- 1. Enable extensions (also enable under Database → Extensions)
--    pg_cron — scheduled purge
--    pg_net  — Storage API DELETE requests from SQL
-- ---------------------------------------------------------------------------
create extension if not exists pg_cron with schema extensions;
create extension if not exists pg_net;
create extension if not exists http with schema extensions;

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

-- ---------------------------------------------------------------------------
-- 4. Vault secrets for purge cron (ONE-TIME — replace placeholders, then run)
--    Project Settings → API → Project URL and service_role key (secret).
-- ---------------------------------------------------------------------------
-- select vault.create_secret(
--   'https://YOUR_PROJECT_REF.supabase.co',
--   'supabase_project_url'
-- );
-- select vault.create_secret(
--   'YOUR_SERVICE_ROLE_JWT',
--   'supabase_service_role_key'
-- );

-- ---------------------------------------------------------------------------
-- 5. Purge function (Storage API — direct DELETE on storage.objects is blocked)
-- ---------------------------------------------------------------------------
create or replace function public.purge_leaf_uploads_storage()
returns integer
language plpgsql
security definer
set search_path = public, extensions, storage, net
as $$
declare
  v_project_url text;
  v_service_key text;
  v_obj record;
  v_request_url text;
  v_headers jsonb;
  v_deleted integer := 0;
begin
  select decrypted_secret into v_project_url
  from vault.decrypted_secrets
  where name = 'supabase_project_url'
  limit 1;

  select decrypted_secret into v_service_key
  from vault.decrypted_secrets
  where name = 'supabase_service_role_key'
  limit 1;

  if v_project_url is null or v_service_key is null then
    raise exception
      'Missing Vault secrets supabase_project_url and/or supabase_service_role_key. '
      'Uncomment and run the vault.create_secret block above.';
  end if;

  v_headers := jsonb_build_object(
    'apikey', v_service_key,
    'Authorization', 'Bearer ' || v_service_key
  );

  for v_obj in
    select name
    from storage.objects
    where bucket_id = 'leaf-uploads'
      and name is not null
      and name != '.emptyFolderPlaceholder'
  loop
    v_request_url := rtrim(v_project_url, '/')
      || '/storage/v1/object/leaf-uploads/'
      || (
        select string_agg(extensions.urlencode(part), '/')
        from unnest(string_to_array(v_obj.name, '/')) as part
      );

    perform net.http_delete(
      url := v_request_url,
      headers := v_headers,
      timeout_milliseconds := 30000
    );

    v_deleted := v_deleted + 1;
  end loop;

  return v_deleted;
end;
$$;

revoke all on function public.purge_leaf_uploads_storage() from public;
grant execute on function public.purge_leaf_uploads_storage() to postgres;

-- ---------------------------------------------------------------------------
-- 6. Cron: delete ALL leaf images every 48 hours (00:00 UTC every 2 days)
--    Schedule: minute 0, hour 0, every 2nd day of the month (1, 3, 5, …)
--    To change timing, edit the cron expression below.
-- ---------------------------------------------------------------------------
select cron.unschedule(jobid)
from cron.job
where jobname = 'purge-leaf-uploads-every-48h';

select cron.schedule(
  'purge-leaf-uploads-every-48h',
  '0 0 1-31/2 * *',
  $$select public.purge_leaf_uploads_storage();$$
);

-- Verify the job was registered:
-- select jobid, jobname, schedule, command from cron.job where jobname = 'purge-leaf-uploads-every-48h';

-- Manual test (optional — deletes all current leaf uploads via Storage API):
-- select public.purge_leaf_uploads_storage();
