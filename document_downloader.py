import requests
import logging
from pathlib import Path
from urllib.parse import urlparse
import tempfile

logger = logging.getLogger(__name__)


class DocumentDownloader:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def download_document(self, url: str) -> Path:
        try:
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name

            if not filename or '.' not in filename:
                if 'pdf' in url.lower():
                    filename = 'document.pdf'
                elif 'ppt' in url.lower() or 'powerpoint' in url.lower():
                    filename = 'document.pptx'
                else:
                    filename = 'document.pdf'

            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()

            temp_dir = Path(tempfile.gettempdir())
            file_path = temp_dir / filename

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded document to {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error downloading document from {url}: {e}")
            raise