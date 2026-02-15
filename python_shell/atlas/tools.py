from pathlib import Path
from typing import Dict, Any, List
import logging
import re
import subprocess
import time

# Import the rust core functions/classes
from atlas.semantic_engine import RepoGraph, check_syntax

log = logging.getLogger("atlas")

def language_from_path(path: Path) -> str:
    """Helper to get language name from file extension."""
    ext = path.suffix.lstrip('.').lower()
    # This map can be expanded
    lang_map = {
        "py": "python",
        "rs": "rust",
        "js": "javascript",
        "ts": "typescript",
        "go": "go",
    }
    return lang_map.get(ext, "unknown")


class ToolExecutor:
    def __init__(self, project_root: Path, repo_graph: RepoGraph):
        self.project_root = project_root
        self.repo_graph = repo_graph
        self.execution_log: List[Dict[str, Any]] = []

    def log_action(self, tool: str, target: str, success: bool, **kwargs):
        """Log all tool executions for debugging."""
        log_entry = {
            'tool': tool,
            'target': target,
            'success': success,
            'timestamp': time.time(),
            **kwargs
        }
        self.execution_log.append(log_entry)
        log.debug(f"Tool Executed: {log_entry}")

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read file with safety checks to prevent path traversal."""
        try:
            full_path = (self.project_root / path).resolve()
            
            # Security: prevent path traversal
            if not full_path.is_relative_to(self.project_root):
                error_msg = "Path traversal detected"
                self.log_action('read_file', path, success=False, error=error_msg)
                return {'success': False, 'error': error_msg}
            
            content = full_path.read_text(encoding="utf-8")
            self.log_action('read_file', path, success=True)
            return {
                'success': True,
                'content': content,
                'lines': len(content.split('\n'))
            }
        except Exception as e:
            self.log_action('read_file', path, success=False, error=str(e))
            return {'success': False, 'error': str(e)}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write file with backup and reflexive syntax validation.
        """
        # 1. Validation Step (The "Reflexive Sensory Loop")
        try:
            lang = language_from_path(Path(path))
            if lang != "unknown":
                syntax_check = check_syntax(content, lang)
                if not syntax_check.is_valid:
                    error_msg = f"Syntax Error on line {syntax_check.line}: {syntax_check.message}. Please fix the code before saving."
                    self.log_action('write_file', path, success=False, status='refused_by_parser', error=error_msg)
                    return {
                        'success': False,
                        'status': 'refused_by_parser',
                        'error': error_msg,
                    }
        except Exception as e:
            error_msg = f"An error occurred during syntax validation: {e}"
            self.log_action('write_file', path, success=False, status='parser_failed', error=error_msg)
            return {'success': False, 'status': 'parser_failed', 'error': error_msg}

        # 2. Execution Step (with safety checks)
        try:
            full_path = (self.project_root / path).resolve()
            if not full_path.is_relative_to(self.project_root):
                error_msg = "Path traversal detected"
                self.log_action('write_file', path, success=False, error=error_msg)
                return {'success': False, 'error': error_msg}

            # Create backup if file exists
            if full_path.exists():
                backup_path = full_path.with_suffix(full_path.suffix + f".backup.{int(time.time())}")
                full_path.rename(backup_path)
                log.info(f"  → Created backup: {backup_path.name}")
            
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")

            # 3. Update Graph
            # The agent's file watcher will pick up the change and update the graph.
            # Calling update_file here directly would be redundant and could cause race conditions.
            
            self.log_action('write_file', path, success=True)
            return {
                'success': True,
                'status': 'file_written',
                'bytes_written': len(content.encode('utf-8'))
            }
        except Exception as e:
            self.log_action('write_file', path, success=False, error=str(e))
            return {'success': False, 'status': 'write_failed', 'error': str(e)}

    def list_directory(self, path: str) -> Dict[str, Any]:
        """Lists directory contents with safety checks."""
        try:
            full_path = (self.project_root / path).resolve()
            if not full_path.is_relative_to(self.project_root):
                error_msg = "Path traversal detected"
                self.log_action('list_directory', path, success=False, error=error_msg)
                return {'success': False, 'error': error_msg}
            
            if not full_path.is_dir():
                error_msg = "Path is not a directory"
                self.log_action('list_directory', path, success=False, error=error_msg)
                return {'success': False, 'error': error_msg}

            entries = [p.name for p in full_path.iterdir()]
            self.log_action('list_directory', path, success=True)
            return {'success': True, 'entries': entries}

        except Exception as e:
            self.log_action('list_directory', path, success=False, error=str(e))
            return {'success': False, 'error': str(e)}

    def generate_repository_map(self, output_path: str) -> Dict[str, Any]:
        """
        Generates a human-readable map of the repository's architecture and saves it to a file.
        This map includes top-ranked files by architectural importance and a directory structure view.

        Args:
            output_path (str): The path where the repository map should be saved (e.g., 'repo_map.md').
        """
        try:
            # 1. Generate the map from the repo graph
            # We can set a limit for the number of top files to show to keep it concise.
            map_content = self.repo_graph.generate_map(max_files=50)

            # 2. Safely resolve the output path
            full_path = (self.project_root / output_path).resolve()
            if not full_path.is_relative_to(self.project_root):
                error_msg = "Path traversal detected. Cannot write outside the project root."
                self.log_action('generate_repository_map', output_path, success=False, error=error_msg)
                return {'success': False, 'error': error_msg}

            # 3. Write the content to the file
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(map_content, encoding="utf-8")
            
            self.log_action('generate_repository_map', output_path, success=True, bytes_written=len(map_content.encode('utf-8')))
            return {
                'success': True,
                'message': f"Repository map successfully saved to {output_path}"
            }
        except Exception as e:
            self.log_action('generate_repository_map', output_path, success=False, error=str(e))
            return {'success': False, 'error': str(e)}

# For backward compatibility with agent.py which might still import FileTools
FileTools = ToolExecutor

