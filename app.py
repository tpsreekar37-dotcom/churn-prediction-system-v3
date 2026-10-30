"""
Root entry point forwarding to dashboard/app.py.
Allows running with either:
  streamlit run app.py
  streamlit run dashboard/app.py
"""
import os
import runpy

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py")
    runpy.run_path(target, run_name="__main__")
