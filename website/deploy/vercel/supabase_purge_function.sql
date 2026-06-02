-- Purge leaf-uploads via Storage API (required since Supabase blocks direct DELETE on storage.objects)
-- Run after supabase_setup.sql, or include this file before scheduling the cron job.

create extension if not exists pg_net;
create extension if not exists http with schema extensions;

-- ---------------------------------------------------------------------------
-- Vault secrets (ONE-TIME): replace placeholders, then run this block once.
-- Project Settings → API → Project URL and service_role key (secret).
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
-- Deletes every object in leaf-uploads through DELETE /storage/v1/object/...
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
      'Uncomment and run the vault.create_secret block in supabase_purge_function.sql.';
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
