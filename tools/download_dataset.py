"""Download, verify and extract the full pinned BONES-SEED Uniform BVH dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'motions/source/bones_seed'
REPO = 'bones-studio/seed'
REVISION = '2f59b2077b9da34dd4e43618e705c7cb962c9a66'
LICENSE_SHA256 = 'eb9b70b084e0a5dcda8465a4be2ba2f5b81e3c3800305c727777a3176defbc5f'


def verified(content, expected):
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('File SHA-256 does not match the pinned source.')
    return content

ARCHIVE_SHA256 = '0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c'


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def download_archive():
    """Fetch bounded HTTP ranges; commit only a contiguous prefix for safe resume."""
    from concurrent.futures import ThreadPoolExecutor
    import time
    import httpx2
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    archive = FOLDER / 'soma_uniform.tar.gz'
    if archive.exists():
        return archive
    url = hf_hub_url(REPO, archive.name, repo_type='dataset', revision=REVISION)
    metadata = get_hf_file_metadata(url, token=True, timeout=30)
    if metadata.etag != ARCHIVE_SHA256 or metadata.size != 45188088297:
        raise ValueError('Unexpected official archive metadata.')
    partial = archive.with_name(archive.name + '.partial')
    legacy = archive.with_suffix('.tar.gz.partial')
    if not partial.exists() and legacy.exists():
        legacy.replace(partial)
    if not partial.exists():
        candidates = list((FOLDER/'.cache/huggingface/download').glob(f'*.{ARCHIVE_SHA256}.*.incomplete'))
        if candidates:
            max(candidates, key=lambda p: p.stat().st_size).replace(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > metadata.size:
        raise ValueError('Partial archive is larger than the official file.')
    print(f'Resuming archive at {offset}/{metadata.size} bytes.', flush=True)
    chunk_size = 16 * 1024 * 1024

    def fetch(bounds):
        start, end = bounds
        location = metadata.location
        for attempt in range(3):
            try:
                with httpx2.Client(timeout=60, follow_redirects=True) as client:
                    response = client.get(location, headers={
                        'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
                if response.status_code == 403:
                    refreshed = get_hf_file_metadata(url, token=True, timeout=30)
                    if refreshed.etag != ARCHIVE_SHA256:
                        raise ValueError('Archive metadata changed.')
                    location = refreshed.location
                expected = f'bytes {start}-{end}/{metadata.size}'
                if (response.status_code != 206 or response.headers.get('content-range') != expected
                        or len(response.content) != end-start+1):
                    raise ValueError('Invalid range response.')
                return response.content
            except Exception:
                if attempt == 2:
                    raise RuntimeError(f'Range download failed at byte {start}; rerun to resume.') from None
                time.sleep(2 ** attempt)

    with partial.open('r+b' if partial.exists() else 'w+b') as stream, ThreadPoolExecutor(max_workers=8) as pool:
        stream.seek(0, 2)
        while offset < metadata.size:
            # Queue at most 512 MiB so slow ranges do not idle the other workers.
            ranges = [(start, min(start+chunk_size, metadata.size)-1)
                      for start in range(offset, min(offset+32*chunk_size, metadata.size), chunk_size)]
            # map preserves order; never create holes if a later request finishes first.
            for data in pool.map(fetch, ranges):
                stream.write(data)
                offset += len(data)
            stream.flush()
            print(f'Downloaded {offset}/{metadata.size} bytes.', flush=True)
    print('Verifying resumed archive SHA-256.', flush=True)
    if file_hash(partial) != ARCHIVE_SHA256:
        raise ValueError('Partial archive SHA-256 mismatch; do not use it as a source.')
    partial.replace(archive)
    return archive


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    from huggingface_hub import hf_hub_download
    def download(name):
        return Path(hf_hub_download(REPO, name, repo_type='dataset',
                                   revision=REVISION, local_dir=FOLDER))
    license_path = download('LICENSE.md')
    verified(license_path.read_bytes(), LICENSE_SHA256)
    metadata = download('metadata/seed_metadata_v004.csv')
    with metadata.open() as stream:
        rows = list(csv.DictReader(stream))
    members = {row['move_soma_uniform_path']: row for row in rows}
    names = [PurePosixPath(name).name for name in members]
    if len(members) != len(rows) or len(set(names)) != len(rows):
        raise ValueError('Duplicate source paths or filenames in metadata.')
    print(f'Downloading archive for {len(rows)} motions.', flush=True)
    already_downloaded = (FOLDER/'soma_uniform.tar.gz').exists()
    archive = download_archive()
    if already_downloaded:
        print('Verifying full archive SHA-256.', flush=True)
        if file_hash(archive) != ARCHIVE_SHA256:
            raise ValueError('Archive SHA-256 mismatch.')
    index = FOLDER / 'dataset_index.jsonl'
    pending = index.with_suffix('.jsonl.tmp')
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream, pending.open('w') as records:
        for member in stream:
            if member.isdir():
                continue
            name = member.name.removeprefix('./')
            if not name.endswith('.bvh'):
                continue
            if name not in members or name in seen or not member.isfile() or member.size > 100_000_000:
                raise ValueError(f'Unexpected archive member: {name}')
            data = stream.extractfile(member).read()
            if len(data) != member.size or not data.lstrip().startswith(b'HIERARCHY'):
                raise ValueError(f'Invalid BVH: {name}')
            path = FOLDER / PurePosixPath(name).name
            digest = hashlib.sha256(data).hexdigest()
            if path.exists():
                if file_hash(path) != digest:
                    raise FileExistsError(f'Different existing source: {path}')
            else:
                temporary = path.with_suffix('.bvh.tmp')
                temporary.write_bytes(data)
                temporary.replace(path)
            sidecar = path.with_suffix('.source.json')
            provenance = {
                'dataset': REPO, 'revision': REVISION, 'archive': archive.name,
                'archive_member': name, 'archive_sha256': ARCHIVE_SHA256,
                'sha256': digest, 'license': 'BONES-SEED Dataset License Agreement',
                'license_file': str(license_path.relative_to(ROOT)),
                'license_sha256': LICENSE_SHA256,
            }
            if sidecar.exists():
                old = json.loads(sidecar.read_text())
                if old.get('sha256') != digest or old.get('revision') != REVISION:
                    raise ValueError(f'Conflicting source record: {sidecar}')
                provenance = {**old, **provenance}
            temporary = sidecar.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(provenance, indent=2) + '\n')
            temporary.replace(sidecar)
            records.write(json.dumps({'source': str(path.relative_to(ROOT)), 'sha256': digest,
                                      'archive_member': name}) + '\n')
            seen.add(name)
            if len(seen) % 1000 == 0:
                print(f'Extracted and verified {len(seen)}/{len(rows)}', flush=True)
        if seen != set(members):
            raise ValueError(f'Missing {len(set(members) - seen)} motions.')
    pending.replace(index)
    print(f'COMPLETE: {len(seen)} verified BVHs; {index.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
