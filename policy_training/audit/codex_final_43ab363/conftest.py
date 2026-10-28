import json
from pathlib import Path

_reports = []

def pytest_runtest_logreport(report):
    if report.when == 'call' or (report.failed and report.when != 'call'):
        _reports.append({'test': report.nodeid, 'phase': report.when, 'outcome': report.outcome,
                         'failure': str(report.longrepr) if report.failed else None})

def pytest_sessionfinish(session, exitstatus):
    Path('/tmp/r2r2r-codex-final-evidence/adversarial_results.json').write_text(
        json.dumps({'exitstatus': int(exitstatus), 'results': _reports}, indent=2))
