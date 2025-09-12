# logging_setup.py
import logging
import json
from prometheus_client import Counter, Histogram

# --- Structured Logging Setup ---
class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if hasattr(record, "client_ip"):
            log_record["client_ip"] = record.client_ip
        if hasattr(record, "filename"):
            log_record["filename"] = record.filename
        if hasattr(record, "status"):
            log_record["status"] = record.status
        return json.dumps(log_record)

# Set up logger
handler = logging.StreamHandler()
handler.setFormatter(JSONLogFormatter())
logger = logging.getLogger("fairmap")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# --- Prometheus Metrics ---
UPLOAD_SUCCESS = Counter("upload_success_total", "Number of successful map uploads")
UPLOAD_FAILURE = Counter("upload_failure_total", "Number of failed map uploads")
UPLOAD_SIZE = Histogram(
    "upload_file_size_bytes", 
    "Size of uploaded files in bytes",
    buckets=[1e5, 1e6, 1e7, 5e7, 1e8]
)
UPLOAD_LATENCY = Histogram(
    "upload_latency_seconds", 
    "Time taken to process uploads",
    buckets=[0.5, 1, 2, 5, 10, 30]
)
