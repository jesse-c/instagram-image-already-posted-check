# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "boto3",
#     "structlog",
# ]
# ///

import os
import concurrent.futures
from typing import Optional, List, Tuple, Any

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger()


def get_client(
    region_name: str,
    profile_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> Any:
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client("s3", region_name=region_name)
    elif aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        raise ValueError("neither authentication method provided")


def create_bucket_if_not_exists(
    s3_client: Any, bucket_name: str, region_name: str
) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logger.info(f"Bucket '{bucket_name}' does not exist. Creating now...")
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region_name},
            )
            logger.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            raise


def upload_file(args: Tuple[Any, str, str, str]) -> str:
    s3_client, local_path, bucket_name, s3_path = args
    try:
        logger.info(f"Uploading {local_path} to {bucket_name}/{s3_path}")
        s3_client.upload_file(local_path, bucket_name, s3_path)
        return f"Successfully uploaded {local_path} to {bucket_name}/{s3_path}"
    except FileNotFoundError:
        return f"The file {local_path} was not found"
    except Exception as e:
        return f"Error uploading {local_path}: {str(e)}"


def upload_to_s3(
    s3_client: Any, bucket_name: str, local_directory: str, max_workers: int = 10
) -> None:
    upload_args: List[Tuple[Any, str, str, str]] = []
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            s3_path = os.path.relpath(local_path, local_directory)
            upload_args.append((s3_client, local_path, bucket_name, s3_path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(upload_file, upload_args))

    for result in results:
        logger.info(result)


if __name__ == "__main__":
    local_directory: str = "data/"

    region_name: Optional[str] = os.getenv("REGION_NAME")
    if region_name is None:
        raise ValueError("missing region name")
    bucket_name: Optional[str] = os.getenv("BUCKET_NAME")
    if bucket_name is None:
        raise ValueError("missing bucket name")

    profile_name: Optional[str] = os.getenv("PROFILE_NAME")
    aws_access_key_id: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")

    if profile_name:
        logger.info("Using AWS profile for authentication")
        s3_client = get_client(region_name, profile_name=profile_name)
    elif aws_access_key_id and aws_secret_access_key:
        logger.info("Using AWS access key and secret for authentication")
        s3_client = get_client(
            region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
    else:
        raise ValueError("neither authentication method provided")

    create_bucket_if_not_exists(
        s3_client,
        bucket_name,
        region_name,
    )

    upload_to_s3(
        s3_client,
        bucket_name,
        local_directory,
    )
