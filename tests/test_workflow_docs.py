from pathlib import Path
import os
import re
import shutil
import subprocess

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from gsdb.cli import app


ROOT = Path(__file__).resolve().parents[1]


def test_active_document_links_resolve_without_workspace_archive():
    files = [ROOT / 'README.md', ROOT / 'AGENTS.md', *ROOT.glob('docs/*.md'), *ROOT.glob('tools/*/README.md')]
    for document in files:
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            linked = (document.parent / target.split('#')[0]).resolve()
            assert linked.is_relative_to(ROOT), (document, target)
            assert linked.exists(), (document, target)


def test_manual_command_options_match_current_cli():
    text = (ROOT / 'docs' / 'CURRENT-WORKFLOW.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```powershell\n(.*?)```', text, re.S)
    count = 0
    for block in blocks:
        block = re.sub(r'`\s*\n\s*', ' ', block)
        for command in re.findall(r'^Invoke-Gsdb\s+([^\n]+)', block, re.M):
            tokens = command.split()
            node = get_command(app)
            names = []
            while tokens and tokens[0] in getattr(node, 'commands', {}):
                name = tokens.pop(0)
                names.append(name)
                node = node.commands[name]
            allowed = {'--help'}
            for param in node.params:
                allowed.update(getattr(param, 'opts', []) + getattr(param, 'secondary_opts', []))
            assert set(re.findall(r'--[\w-]+', command)) <= allowed, command
            result = CliRunner().invoke(app, names + ['--help'])
            assert result.exit_code == 0, result.stdout
            count += 1
    assert count >= 15


@pytest.mark.skipif(os.name != 'nt', reason='PowerShell parser requires Windows environment')
def test_powershell_examples_and_retained_scripts_parse(tmp_path):
    powershell = shutil.which('pwsh') or shutil.which('powershell')
    if not powershell:
        pytest.skip('PowerShell unavailable')
    text = '\n'.join(p.read_text(encoding='utf-8') for p in [ROOT / 'README.md', *ROOT.glob('docs/*.md')])
    examples = tmp_path / 'documentation-examples.ps1'
    examples.write_text('\n'.join(re.findall(r'```powershell\n(.*?)```', text, re.S)), encoding='utf-8')
    paths = [examples, ROOT / 'gsdb.ps1', *ROOT.glob('scripts/*.ps1'), *ROOT.glob('tools/*/build.ps1')]
    statements = []
    for path in paths:
        literal = str(path).replace("'", "''")
        statements.append("$ParseErrors=$null; [System.Management.Automation.Language.Parser]::ParseFile('" + literal + "',[ref]$null,[ref]$ParseErrors) | Out-Null; if($ParseErrors){$ParseErrors | Out-String;exit 1}")
    result = subprocess.run([powershell, '-NoProfile', '-Command', '\n'.join(statements)],
                            capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stdout + result.stderr
