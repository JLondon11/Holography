"""Download and safely extract the *real* IXI T1 MRI archive from its official host.

No artificial scans, no model training, no automatic removal of source archive.
Designed for a workstation with sufficient disk space. External data licensing
and access conditions must be reviewed by the researcher.

Usage:
 python data/acquire_ixi_t1.py --output-dir '/Volumes/BioMedAI Data Lake/IXI-T1' --download --extract

The download uses a temporary .part file and atomic rename to avoid treating
an incomplete archive as acquired. The original archive is retained.
"""
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
NAME = "IXI-T1.tar"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def acquire(directory):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    archive=directory/NAME
    part=directory/(NAME+".part")
    if not archive.exists():
        if part.exists():
            raise RuntimeError(f"Incomplete prior transfer at {part}; inspect or delete it before retrying")
        request=urllib.request.Request(URL,headers={"User-Agent":"Holography-Research-Data-Acquisition/1.0"})
        with urllib.request.urlopen(request,timeout=120) as stream,part.open("xb") as out:
            shutil.copyfileobj(stream,out,length=4*1024*1024)
        if not tarfile.is_tarfile(part):
            raise ValueError(f"Downloaded bytes are not a readable tar archive: {part}")
        part.replace(archive)
    if not tarfile.is_tarfile(archive):
        raise ValueError("Archive is not a valid tar file")
    metadata={"source_url":URL,"archive_path":str(archive),"sha256":digest(archive),
              "archive_bytes":archive.stat().st_size,
              "dataset":"IXI T1 MRI","source_kind":"acquired_real_mri",
              "license_source":"https://brain-development.org/ixi-dataset/"}
    (directory/"acquisition_record.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return archive,metadata

def safe_extract(archive,dest):
    dest=Path(dest).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,"r:") as tf:
        members=tf.getmembers()
        if not members:
            raise ValueError("Empty archive")
        for member in members:
            path=Path(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError(f"Unsafe or unsupported tar member: {member.name}")
            resolved=(dest/path).resolve()
            if not resolved.is_relative_to(dest):
                raise ValueError(f"Tar member escapes extraction root: {member.name}")
            if not (member.name.endswith(".nii") or member.name.endswith(".nii.gz")):
                raise ValueError(f"Unexpected non-MRI file: {member.name}")
            if resolved.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
        for member in members:
            target=dest/member.name
            target.parent.mkdir(parents=True,exist_ok=True)
            source=tf.extractfile(member)
            if source is None:
                raise ValueError(f"Archive entry inaccessible: {member.name}")
            with source,target.open("xb") as sink:
                shutil.copyfileobj(source,sink,length=4*1024*1024)
    return len(members)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--download",action="store_true")
    p.add_argument("--extract",action="store_true")
    args=p.parse_args()
    if not args.download and not args.extract:
        p.error("Explicit --download and/or --extract required")
    directory=args.output_dir.resolve()
    if args.download:
        archive,metadata=acquire(directory)
        print(json.dumps(metadata,indent=2))
    else:
        archive=directory/NAME
        if not archive.is_file():
            raise FileNotFoundError(archive)
    if args.extract:
        count=safe_extract(archive,directory/"nifti")
        print(f"Extracted {count} source MRI NIfTI files into {directory/'nifti'}")
        print("Next: run data/build_ixi_manifest.py on the nifti directory")

if __name__=="__main__":
    main()
