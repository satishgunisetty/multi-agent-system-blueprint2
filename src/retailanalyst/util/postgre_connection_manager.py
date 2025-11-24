from databricks.sdk import WorkspaceClient
import uuid
import psycopg
import time
import threading
import logging
from urllib.parse import quote_plus
from langgraph.checkpoint.postgres import PostgresSaver

log = logging.getLogger(__name__)


class RobustLakebaseConnectionManager:
    """Production-ready Lakebase connection manager following Databricks best practices."""

    def __init__(self, instance_name="source-to-pay-memory"):
        self.instance_name = instance_name
        self.workspace_client = WorkspaceClient()
        self._current_token = None
        self._token_expiry = 0
        self._lock = threading.RLock()
        self._token_refresh_margin = 300  # Refresh 5 minutes before expiry

    def _generate_oauth_token(self):
        """Generate OAuth token following Databricks documentation patterns."""
        try:
            log.info("🔑 Generating fresh OAuth token for Lakebase...")

            # Use Databricks SDK pattern from documentation
            cred = self.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()), instance_names=[self.instance_name]
            )

            # OAuth tokens expire after 1 hour according to docs
            self._current_token = cred.token
            self._token_expiry = time.time() + 3600  # 1 hour from now

            log.info("OAuth token generated successfully")
            return True

        except Exception as e:
            log.error(f"OAuth token generation failed: {e}")
            return False

    def get_current_token(self):
        """Get current valid OAuth token with auto-refresh."""
        with self._lock:
            current_time = time.time()

            # Check if token needs refresh (5 minutes before expiry)
            if self._current_token is None or current_time >= (
                self._token_expiry - self._token_refresh_margin
            ):

                log.info("Token expired or near expiry, refreshing...")
                if not self._generate_oauth_token():
                    raise Exception("Failed to generate OAuth token")

            return self._current_token

    def create_connection_string(self, username):
        """Create connection string with current valid token."""
        token = self.get_current_token()

        encoded_username = quote_plus(username)
        encoded_token = quote_plus(token)

        # Enhanced connection string based on documentation best practices
        conn_str = (
            f"postgresql://{encoded_username}:{encoded_token}@"
            f"instance-f3fd2ceb-e103-44cf-a582-9a786995ebda.database.cloud.databricks.com:5432/"
            f"databricks_postgres?"
            f"sslmode=require&"  # SSL mandatory for token auth per docs
            f"connect_timeout=30&"
            f"application_name=langgraph_multiagent&"
            f"keepalives_idle=300&"
            f"keepalives_interval=15&"
            f"keepalives_count=3&"
            f"tcp_user_timeout=30000"
        )

        return conn_str


class TokenRotatingPostgresSaver:
    """PostgresSaver with automatic token rotation using correct API."""

    def __init__(
        self,
        instance_name="source-to-pay-memory",
        username="satish_gunisetty@epam.com",
    ):
        self.connection_manager = RobustLakebaseConnectionManager(instance_name)
        self.username = username
        self._checkpointer = None
        self._checkpointer_context = None  # Store context manager
        self._last_created = 0
        self._recreation_interval = 1800  # 30 minutes
        self._setup_complete = False
        self._lock = threading.RLock()

    def _create_fresh_checkpointer(self):
        """Create new PostgresSaver with fresh token using correct API."""
        try:
            log.info("🔧 Creating fresh PostgresSaver with rotated token...")

            # Get properly encoded connection string
            conn_str = self.connection_manager.create_connection_string(self.username)

            # Test connection first
            log.info("🔍 Testing connection with fresh token...")
            with psycopg.connect(conn_str, connect_timeout=10) as test_conn:
                with test_conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                log.info("Connection test successful")

            # FIX: Use from_conn_string() method instead of constructor
            checkpointer_context = PostgresSaver.from_conn_string(conn_str)
            checkpointer = checkpointer_context.__enter__()

            # Setup schema only once
            if not self._setup_complete:
                checkpointer.setup()
                self._setup_complete = True
                log.info("PostgresSaver schema initialized")
            else:
                log.info("PostgresSaver recreated with fresh token")

            self._last_created = time.time()
            return checkpointer, checkpointer_context

        except Exception as e:
            log.error(f"Failed to create PostgresSaver: {e}")
            raise

    def get_checkpointer(self):
        """Get PostgresSaver with token rotation management."""
        with self._lock:
            current_time = time.time()

            needs_recreation = (
                self._checkpointer is None
                or (current_time - self._last_created) > self._recreation_interval
            )

            if needs_recreation:
                log.info("Recreating PostgresSaver with fresh token...")

                # Clean up old context if exists
                if self._checkpointer_context:
                    try:
                        self._checkpointer_context.__exit__(None, None, None)
                    except Exception as cleanup_error:
                        log.warning(f"Cleanup warning: {cleanup_error}")

                # Create new checkpointer and context
                self._checkpointer, self._checkpointer_context = (
                    self._create_fresh_checkpointer()
                )

            return self._checkpointer
