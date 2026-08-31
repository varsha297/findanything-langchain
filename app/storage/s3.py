import os
import tempfile
from pathlib import Path

import boto3


session = boto3.Session(
    profile_name=os.getenv(
        "AWS_PROFILE",
        "default",
    ),
    region_name=os.getenv(
        "AWS_REGION",
        "eu-north-1",
    ),
)

s3_client = session.client("s3")


def download_s3_object(
    bucket: str,
    key: str,
    original_name: str,
) -> str:
    suffix = Path(
        original_name
    ).suffix

    temporary_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary_file.close()

    try:
        s3_client.download_file(
            bucket,
            key,
            temporary_file.name,
        )
    except Exception:
        Path(
            temporary_file.name
        ).unlink(
            missing_ok=True
        )
        raise

    return temporary_file.name