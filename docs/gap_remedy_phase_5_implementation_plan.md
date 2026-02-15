# Phase 5 Implementation Plan: Observability & Polish

**Goal:** Implement logging, progress indicators, and robust error handling to improve the agent's usability, debuggability, and resilience, as outlined in Phase 5 of `1_29_26_implementation_plan.md`.

---

## 1. Logging Infrastructure

To gain visibility into the agent's operations, we will introduce structured logging in both the Rust core and the Python shell.

### 1.1. Add Logging to Rust Core

**File:** `rust_core/Cargo.toml`
**Action:** Add the `log` and `env_logger` crates.

```toml
[dependencies]
# ... other dependencies
log = "0.4"
env_logger = "0.11"
```

**File:** `rust_core/src/lib.rs`
**Action:** Use the `log` macros in critical functions.

```rust
// Add to imports
use log::{info, debug, warn};

// In critical functions like RepoGraph::update_file
impl RepoGraph {
    pub fn update_file(&mut self, file_path: &PathBuf, content: &str) -> Result<UpdateResult, GraphError> {
        debug!("Updating file: {}", file_path.display());
        
        let tier = self.classify_change(file_path, content)?;
        info!("Change classification for {}: {:?}", file_path.display(), tier);
        
        // ... rest of logic ...
        
        info!("Update complete: {:?}", result);
        Ok(result)
    }
}
```

### 1.2. Add Logging to Python Shell

**File:** `python_shell/atlas/agent.py`
**Action:** Configure the Python `logging` module to use `rich` for beautiful output.

```python
# Add to imports
import logging
from rich.logging import RichHandler

# Configure logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)]
)
log = logging.getLogger("atlas")


# In AtlasAgent class, use the logger
class AtlasAgent:
    def __init__(self, project_root: Path, log_level: str = "INFO"):
        # ...
        log.info(f"Initializing agent for {project_root}")
    
    def _handle_file_modified(self, file_path: Path):
        log.debug(f"Processing modification: {file_path}")
        try:
            # ...
            log.info(f"Updated {file_path.name}: +{result.edges_added} / -{result.edges_removed} edges")
        except Exception as e:
            log.error(f"Failed to update {file_path}: {e}", exc_info=True)
```

---

## 2. Progress Indicators

Long-running operations should provide feedback to the user. We will use `rich.progress` for this.

### 2.1. Implement Progress Bars for Initialization

**File:** `python_shell/atlas/agent.py`
**Action:** Wrap the `initialize` method's steps in a `rich.progress` context.

```python
# Add to imports
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.console import Console

console = Console()

# In AtlasAgent class
def initialize(self):
    """Initialize with beautiful progress indicators."""
    console.print(f"[bold cyan]Initializing Atlas for {self.project_root}[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task1 = progress.add_task("[cyan]Scanning repository...", total=None)
        files = scan_repository(str(self.project_root))
        progress.update(task1, completed=True, total=1)
        console.print(f"  ✓ Found {len(files)} files")
        
        task2 = progress.add_task(
            "[yellow]Building dependency graph...",
            total=len(files)
        )
        self.repo_graph.build_complete(files, str(self.project_root))
        progress.update(task2, completed=len(files))
        
        task3 = progress.add_task("[magenta]Calculating PageRank...", total=None)
        self.repo_graph.ensure_pagerank_up_to_date()
        progress.update(task3, completed=True, total=1)
```

---

## 3. Error Handling and Recovery

The agent should be resilient to errors and provide clear feedback when issues arise.

### 3.1. Implement Graph Health Checks

**File to Create:** `python_shell/atlas/health.py`
**Action:** Create a `GraphHealthChecker` to diagnose potential issues with the repo graph.

```python
from dataclasses import dataclass
from typing import List
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"

@dataclass
class HealthCheck:
    status: HealthStatus
    issues: List[str]
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

class GraphHealthChecker:
    def check_health(self, repo_graph) -> HealthCheck:
        """Check graph health and return issues."""
        issues = []
        stats = repo_graph.get_statistics()
        
        if stats.node_count == 0:
            issues.append("Graph is empty - no files indexed")
            return HealthCheck(HealthStatus.CRITICAL, issues)
        
        if stats.node_count > 10 and stats.edge_count == 0:
            issues.append("No edges in graph - imports may not be resolving")
        
        if stats.symbol_edges == 0 and stats.node_count > 5:
            issues.append("No symbol edges - semantic analysis may have failed")
            
        if not issues:
            status = HealthStatus.HEALTHY
        elif len(issues) <= 2:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.CRITICAL
        
        return HealthCheck(status, issues)
```

### 3.2. Implement Auto-Recovery Logic

**File:** `python_shell/atlas/agent.py`
**Action:** Add error handling to `_handle_file_modified` to gracefully handle non-fatal errors.

```python
# In AtlasAgent class
def _handle_file_modified(self, file_path: Path):
    """Handle file modification with error recovery."""
    try:
        content = file_path.read_text()
        result = self.repo_graph.update_file(str(file_path), content)
        # ... success logging
    except GraphError as e:
        log.error(f"Graph error updating {file_path}: {e}")
        if "ParseError" in str(e):
            log.warning("This is a non-fatal syntax error. The graph state for this file will be stale until the error is fixed.")
        else:
            log.error("A more serious graph error occurred. The graph may be inconsistent.")
    except Exception as e:
        log.error(f"An unexpected error occurred while processing {file_path}: {e}", exc_info=True)
```

---

## 4. Execution and Verification

1.  Apply the `Cargo.toml` changes and add logging macros to the Rust core.
2.  Update the Python agent with `rich` logging and progress bars.
3.  Create the `health.py` module and implement the `GraphHealthChecker`.
4.  Add the new error handling logic to the agent's file modification handler.
5.  Run the agent and verify:
    *   Logs are printed to the console with `rich` formatting.
    *   Progress bars are displayed during the initial scan.
    *   Introducing a syntax error in a file logs a `ParseError` but does not crash the agent.
    *   The `GraphHealthChecker` can be instantiated and run (manual test).