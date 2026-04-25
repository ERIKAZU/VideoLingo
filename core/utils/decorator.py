import functools
import time
import re
import os
from rich import print as rprint

# ------------------------------
# retry decorator
# ------------------------------

def _parse_retry_delay(error_str):
    """Parse server-suggested retry delay from 429 error messages (e.g. 'retryDelay': '55s')."""
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", error_str)
    if match:
        return float(match.group(1))
    return None

def except_handler(error_msg, retry=0, delay=1, default_return=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for i in range(retry + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    rprint(f"[red]{error_msg}: {e}, retry: {i+1}/{retry}[/red]")
                    if i == retry:
                        if default_return is not None:
                            return default_return
                        raise last_exception
                    # Use server-suggested delay for 429 errors, otherwise exponential backoff
                    error_str = str(e)
                    server_delay = _parse_retry_delay(error_str) if '429' in error_str else None
                    if server_delay:
                        wait_time = server_delay + 5  # add buffer
                        rprint(f"[yellow]⏳ Rate limited, waiting {wait_time:.0f}s (server suggested {server_delay:.0f}s)...[/yellow]")
                    else:
                        wait_time = delay * (2 ** i)
                        rprint(f"[yellow]⏳ Retrying in {wait_time:.0f}s...[/yellow]")
                    time.sleep(wait_time)
        return wrapper
    return decorator


# ------------------------------
# check file exists decorator
# ------------------------------

def check_file_exists(file_path):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(file_path):
                rprint(f"[yellow]⚠️ File <{file_path}> already exists, skip <{func.__name__}> step.[/yellow]")
                return
            return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    @except_handler("function execution failed", retry=3, delay=1)
    def test_function():
        raise Exception("test exception")
    test_function()
