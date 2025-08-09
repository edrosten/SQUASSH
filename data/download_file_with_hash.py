import hashlib
import os
from pathlib import Path
import tempfile

import requests
import tqdm

cache_dir=Path('cache')

def _download(url: str, filename: Path, sha256: str)->None:
    """
    Downloads a file and checks the hash
    """
    with requests.get(url, stream=True, timeout=1000000) as r: 
        r.raise_for_status()

        total_size = int(r.headers.get('content-length', 0))
        
        progress_bar:tqdm.tqdm = tqdm.tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            desc=filename.name
        )

        checksum = hashlib.sha256()
        try:
            f =  tempfile.NamedTemporaryFile(delete=False, mode='wb', dir=filename.parent) # pylint: disable=consider-using-with
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                checksum.update(chunk)
                progress_bar.update(len(chunk))
            f.close()
            Path(f.name).rename(filename)
        except:
            f.close()
            os.unlink(f.name)
            raise
         
        # Close the progress bar
        progress_bar.close()

        if checksum.hexdigest() != sha256:
            raise RuntimeError(f"Error: downloaded file {f} does not have the right hash")

def _sha256sum(filename: Path)->str:
    with open(filename, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def ensure_cached_files_exist(files: dict[str,str])->None:
    ''' Download files and cache them if they don't exist'''
    for digest, fname in files.items():
        if (cache_dir/fname).exists():
            print(f"Checking {fname}")
            if _sha256sum(cache_dir/fname) != digest:
                print("Error!")
                print(f"File {fname} in the cache does not match the hash")
                print("Your cache has got corrupted. Delete it.")
                raise RuntimeError(f"File {fname} in the cache does not match the hash")
        else:
            (cache_dir/fname).parent.mkdir(parents=True, exist_ok=True)
            _download(f'https://github.com/edrosten/SQUASSH/raw/667ef241179492c815d62c73351f1bc0e0b03f51/data/{requests.utils.quote(fname)}', cache_dir/fname, digest) # type: ignore[attr-defined]
