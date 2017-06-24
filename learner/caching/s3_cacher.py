from learner.caching.object_cacher import ObjectCacher
from learner.data_output.std_logger import L
import os
import boto3


class S3Cacher():
    """
    Pickles and caches files to an S3 backend
    """

    def __init__(self, directory='cache/', bucket_name='icpe-machine-learning-cache', bucket_location='EU'):
        self.object_cacher = ObjectCacher(directory)
        self.client = boto3.client('s3')
        self.bucket_location = bucket_location

        # Download all files available in the bucket
        self.bucket_name = bucket_name
        self.bucket = self.find_or_create_bucket(bucket_name)
        self.download_bucket()

    def download_bucket(self, bucket_name=None):
        if bucket_name is None:
            bucket_name = self.bucket_name

        bucket_content_list = self.client.list_objects(Bucket=bucket_name)
        if not 'Contents' in bucket_content_list:
            L.info('Bucket is empty, not downloading anything')
            return False

        for content in bucket_content_list['Contents']:
            key = content['Key']
            if self.file_available(key):
                L.info('Not downloading file: %s ' % key)
                continue

            L.info('Downloading file: %s ' % key)
            self.client.download_file(Bucket=bucket_name, Key=key, Filename=self.get_dirred_file(key))
        return True

    def find_or_create_bucket(self, bucket_name):
        if (bucket_name in map(lambda w: w['Name'], self.client.list_buckets()['Buckets'])):
            return True

        self.client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': self.bucket_location})

    def get_dirred_file(self, cache_name):
        return self.object_cacher.get_dirred_file(cache_name)

    def read_cache(self, cache_name):
        return self.object_cacher.read_cache(cache_name)

    def write_cache(self, data, cache_name):
        self.object_cacher.write_cache(data, cache_name)

        dirred_cache_name = self.get_dirred_file(cache_name)
        self.client.upload_file(dirred_cache_name, self.bucket_name, cache_name)
        L.info('Uploaded to S3')

    def file_available(self, cache_name, add_dir=True):
        return self.object_cacher.file_available(cache_name, add_dir=add_dir)

    def files_in_dir(self):
        return self.object_cacher.files_in_dir()
