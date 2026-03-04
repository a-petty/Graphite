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
