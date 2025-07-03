import boto3
from app.core.config import settings
import uuid

class R2Service:
    def __init__(self):
        if not all([
            settings.CLOUDFLARE_R2_ACCOUNT_ID,
            settings.CLOUDFLARE_R2_ACCESS_KEY_ID,
            settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
            settings.CLOUDFLARE_R2_BUCKET_NAME
        ]):
            raise ValueError("Cloudflare R2 credentials are not fully configured.")

        r2_endpoint_url = f"https://{settings.CLOUDFLARE_R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
        
        self.client = boto3.client(
            service_name='s3',
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=settings.CLOUDFLARE_R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.CLOUDFLARE_R2_SECRET_ACCESS_KEY,
            region_name='auto',
        )
        self.bucket_name = settings.CLOUDFLARE_R2_BUCKET_NAME

    def upload_file_from_bytes(self, data: bytes, content_type: str = 'application/octet-stream') -> str:
        """
        Uploads a file-like object (bytes) to R2 and returns the object name.
        """
        # Generate a unique object name to avoid overwriting files
        object_name = f"audio/{uuid.uuid4()}.mp3"
        
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=object_name,
            Body=data,
            ContentType=content_type
        )
        return object_name

    def generate_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL to share an R2 object.
        :param object_name: string
        :param expiration: Time in seconds for the presigned URL to remain valid.
        :return: Presigned URL as string. If error, returns None.
        """
        try:
            response = self.client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
        except Exception as e:
            print(f"Error generating presigned URL: {e}")
            return None
        
        return response

# Create a singleton instance to be used across the application
try:
    r2_service = R2Service()
except ValueError:
    r2_service = None 