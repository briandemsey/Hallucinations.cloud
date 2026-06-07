import subprocess
import time
import pytest

# Uncomment if server is not already running
# @pytest.fixture(scope="session", autouse=True)
# def start_server():
#     proc = subprocess.Popen(
#         ['streamlit', 'run', 'Hallucinations_1_28_26.py', '--server.headless=true'],
#         cwd=r"F:\Hallucinations_6_4_25"
#     )
#     time.sleep(4)
#     yield
#     proc.terminate()
