"""
FractiGit.py

Implements sophisticated GitHub utilities for FractiAI, providing advanced repository management,
CI/CD integration, analytics, and automation capabilities. Enables intelligent tracking of
development patterns and automated optimization of workflows.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import yaml
import json
from github import Github, GithubIntegration
from github.Repository import Repository
from github.PullRequest import PullRequest
from github.Workflow import Workflow
from github.WorkflowRun import WorkflowRun

logger = logging.getLogger(__name__)

@dataclass
class GitConfig:
    """Configuration for GitHub integration"""
    token: str
    app_id: Optional[int] = None
    app_private_key: Optional[str] = None
    organization: Optional[str] = None
    repository: Optional[str] = None
    base_branch: str = "main"
    auto_merge: bool = False
    require_reviews: int = 1
    
    # Advanced settings
    workflow_concurrency: int = 3
    retry_attempts: int = 3
    cache_ttl: int = 300
    metrics_window: int = 7
    alert_threshold: float = 0.8

class PullRequestStatus(str, Enum):
    """Status of pull requests"""
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DRAFT = "draft"
    CONFLICT = "conflict"
    REVIEW = "review"

@dataclass
class WorkflowMetrics:
    """Metrics for workflow analysis"""
    success_rate: float
    avg_duration: float
    failure_types: Dict[str, int]
    resource_usage: Dict[str, float]
    bottlenecks: List[str]

class FractiGit:
    """Advanced GitHub integration and management"""
    
    def __init__(self, config: GitConfig):
        self.config = config
        self._setup_client()
        self._cache: Dict[str, Any] = {}
        self._metrics: Dict[str, List[Any]] = {}
        
    def _setup_client(self) -> None:
        """Initialize GitHub client"""
        if self.config.app_id and self.config.app_private_key:
            # Use GitHub App authentication
            integration = GithubIntegration(
                self.config.app_id,
                self.config.app_private_key
            )
            self.client = integration.get_github_for_installation()
        else:
            # Use personal access token
            self.client = Github(self.config.token)
            
        if self.config.organization:
            self.org = self.client.get_organization(self.config.organization)
        else:
            self.org = None
            
    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: Optional[str] = None,
        reviewers: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        draft: bool = False
    ) -> PullRequest:
        """Create a new pull request with advanced options"""
        try:
            repo = await self.get_repository()
            
            # Create pull request
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch or self.config.base_branch,
                draft=draft
            )
            
            if reviewers:
                pr.create_review_request(reviewers)
                
            if labels:
                pr.add_to_labels(*labels)
                
            # Update metrics
            await self._update_pr_metrics(pr)
            
            return pr
            
        except Exception as e:
            logger.error(f"Error creating pull request: {str(e)}")
            raise
            
    async def get_repository(self) -> Repository:
        """Get repository with caching"""
        cache_key = f"repo:{self.config.repository}"
        
        if cache_key in self._cache:
            cache_time, repo = self._cache[cache_key]
            if datetime.now() - cache_time < timedelta(seconds=self.config.cache_ttl):
                return repo
                
        if self.org:
            repo = self.org.get_repo(self.config.repository)
        else:
            repo = self.client.get_repo(self.config.repository)
            
        self._cache[cache_key] = (datetime.now(), repo)
        return repo
        
    async def analyze_workflows(
        self,
        days: Optional[int] = None
    ) -> Dict[str, WorkflowMetrics]:
        """Analyze workflow performance and patterns"""
        repo = await self.get_repository()
        workflows = repo.get_workflows()
        
        metrics = {}
        for workflow in workflows:
            # Get workflow runs
            runs = workflow.get_runs()
            if days:
                cutoff = datetime.now() - timedelta(days=days)
                runs = [r for r in runs if r.created_at > cutoff]
                
            if not runs:
                continue
                
            # Compute metrics
            success_count = sum(1 for r in runs if r.conclusion == "success")
            total_duration = sum(
                (r.updated_at - r.created_at).total_seconds()
                for r in runs if r.conclusion
            )
            
            # Analyze failures
            failure_types = {}
            for run in runs:
                if run.conclusion and run.conclusion != "success":
                    failure_types[run.conclusion] = failure_types.get(run.conclusion, 0) + 1
                    
            # Analyze resource usage
            resource_usage = await self._analyze_resource_usage(runs)
            
            # Detect bottlenecks
            bottlenecks = await self._detect_bottlenecks(runs)
            
            metrics[workflow.name] = WorkflowMetrics(
                success_rate=success_count / len(runs),
                avg_duration=total_duration / len(runs),
                failure_types=failure_types,
                resource_usage=resource_usage,
                bottlenecks=bottlenecks
            )
            
        return metrics
        
    async def _analyze_resource_usage(
        self,
        runs: List[WorkflowRun]
    ) -> Dict[str, float]:
        """Analyze resource usage patterns in workflow runs"""
        usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "duration": 0.0
        }
        
        for run in runs:
            if run.conclusion:
                timing = await self._get_run_timing(run)
                usage["cpu"] += timing.get("cpu_percentage", 0)
                usage["memory"] += timing.get("memory_mb", 0)
                usage["duration"] += timing.get("duration_seconds", 0)
                
        count = len([r for r in runs if r.conclusion])
        if count > 0:
            usage = {k: v / count for k, v in usage.items()}
            
        return usage
        
    async def _detect_bottlenecks(
        self,
        runs: List[WorkflowRun]
    ) -> List[str]:
        """Detect workflow bottlenecks"""
        bottlenecks = []
        
        # Analyze job dependencies and timing
        for run in runs:
            if run.conclusion == "success":
                jobs = run.jobs()
                job_timing = {}
                
                for job in jobs:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    job_timing[job.name] = duration
                    
                # Find jobs taking longer than average
                avg_duration = sum(job_timing.values()) / len(job_timing)
                threshold = avg_duration * 1.5
                
                for job_name, duration in job_timing.items():
                    if duration > threshold:
                        bottlenecks.append(job_name)
                        
        # Return unique bottlenecks
        return list(set(bottlenecks))
        
    async def _get_run_timing(self, run: WorkflowRun) -> Dict[str, float]:
        """Get detailed timing information for a workflow run"""
        timing = {
            "cpu_percentage": 0.0,
            "memory_mb": 0.0,
            "duration_seconds": 0.0
        }
        
        try:
            jobs = run.jobs()
            for job in jobs:
                if job.conclusion:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    timing["duration_seconds"] += duration
                    
                    # Extract resource usage from job logs
                    logs = job.get_logs()
                    timing.update(await self._parse_resource_usage(logs))
                    
        except Exception as e:
            logger.warning(f"Error getting run timing: {str(e)}")
            
        return timing
        
    async def _parse_resource_usage(self, logs: str) -> Dict[str, float]:
        """Parse resource usage from job logs"""
        usage = {
            "cpu_percentage": 0.0,
            "memory_mb": 0.0
        }
        
        try:
            # Parse resource usage from log lines
            for line in logs.split("\n"):
                if "CPU usage:" in line:
                    usage["cpu_percentage"] = float(line.split(":")[1].strip().rstrip("%"))
                elif "Memory usage:" in line:
                    usage["memory_mb"] = float(line.split(":")[1].strip().rstrip("MB"))
                    
        except Exception as e:
            logger.warning(f"Error parsing resource usage: {str(e)}")
            
        return usage
        
    async def optimize_workflow(
        self,
        workflow_name: str,
        target_metric: str = "duration"
    ) -> Dict[str, Any]:
        """Optimize workflow configuration based on analysis"""
        try:
            # Get workflow metrics
            metrics = await self.analyze_workflows(days=self.config.metrics_window)
            if workflow_name not in metrics:
                raise ValueError(f"Workflow {workflow_name} not found")
                
            workflow_metrics = metrics[workflow_name]
            
            # Generate optimization suggestions
            suggestions = {
                "concurrency": self._optimize_concurrency(workflow_metrics),
                "caching": self._optimize_caching(workflow_metrics),
                "job_order": self._optimize_job_order(workflow_metrics),
                "resource_allocation": self._optimize_resources(workflow_metrics)
            }
            
            # Apply optimizations if auto-optimization is enabled
            if self.config.auto_merge:
                await self._apply_workflow_optimizations(workflow_name, suggestions)
                
            return suggestions
            
        except Exception as e:
            logger.error(f"Error optimizing workflow: {str(e)}")
            raise
            
    def _optimize_concurrency(
        self,
        metrics: WorkflowMetrics
    ) -> Dict[str, Any]:
        """Optimize workflow concurrency settings"""
        suggestions = {
            "current_concurrency": self.config.workflow_concurrency,
            "suggested_concurrency": self.config.workflow_concurrency,
            "reason": "No change needed"
        }
        
        # Analyze if current concurrency is optimal
        if metrics.avg_duration > 300 and metrics.success_rate > 0.9:
            suggestions["suggested_concurrency"] = min(
                self.config.workflow_concurrency + 1,
                5
            )
            suggestions["reason"] = "Long duration with high success rate"
            
        elif metrics.success_rate < 0.7:
            suggestions["suggested_concurrency"] = max(
                self.config.workflow_concurrency - 1,
                1
            )
            suggestions["reason"] = "Low success rate"
            
        return suggestions
        
    def _optimize_caching(
        self,
        metrics: WorkflowMetrics
    ) -> Dict[str, Any]:
        """Optimize workflow caching strategy"""
        return {
            "cache_paths": self._identify_cache_paths(metrics),
            "cache_key_strategy": self._generate_cache_keys(metrics),
            "estimated_savings": self._estimate_cache_savings(metrics)
        }
        
    def _optimize_job_order(
        self,
        metrics: WorkflowMetrics
    ) -> Dict[str, Any]:
        """Optimize job ordering in workflow"""
        return {
            "parallel_jobs": self._identify_parallel_jobs(metrics),
            "dependencies": self._optimize_dependencies(metrics),
            "estimated_improvement": self._estimate_ordering_improvement(metrics)
        }
        
    def _optimize_resources(
        self,
        metrics: WorkflowMetrics
    ) -> Dict[str, Any]:
        """Optimize resource allocation"""
        return {
            "cpu_allocation": self._optimize_cpu_allocation(metrics),
            "memory_allocation": self._optimize_memory_allocation(metrics),
            "estimated_cost_savings": self._estimate_resource_savings(metrics)
        }
        
    async def _apply_workflow_optimizations(
        self,
        workflow_name: str,
        suggestions: Dict[str, Any]
    ) -> None:
        """Apply workflow optimizations"""
        try:
            repo = await self.get_repository()
            workflow = next(
                w for w in repo.get_workflows()
                if w.name == workflow_name
            )
            
            # Read current workflow
            content = workflow.file.decoded_content.decode()
            workflow_data = yaml.safe_load(content)
            
            # Apply optimizations
            workflow_data = await self._update_workflow_config(
                workflow_data,
                suggestions
            )
            
            # Create pull request with changes
            branch_name = f"workflow-optimization/{workflow_name}"
            await self.create_branch(branch_name)
            
            # Update workflow file
            repo.update_file(
                workflow.file.path,
                f"Optimize {workflow_name} workflow",
                yaml.dump(workflow_data),
                workflow.file.sha,
                branch=branch_name
            )
            
            # Create pull request
            await self.create_pull_request(
                title=f"Optimize {workflow_name} workflow",
                body=self._generate_optimization_pr_body(suggestions),
                head_branch=branch_name,
                labels=["automation", "optimization"]
            )
            
        except Exception as e:
            logger.error(f"Error applying workflow optimizations: {str(e)}")
            raise
            
    async def create_branch(
        self,
        branch_name: str,
        base_branch: Optional[str] = None
    ) -> None:
        """Create a new branch"""
        try:
            repo = await self.get_repository()
            base = repo.get_branch(base_branch or self.config.base_branch)
            repo.create_git_ref(
                f"refs/heads/{branch_name}",
                base.commit.sha
            )
        except Exception as e:
            logger.error(f"Error creating branch: {str(e)}")
            raise
            
    def _generate_optimization_pr_body(
        self,
        suggestions: Dict[str, Any]
    ) -> str:
        """Generate detailed pull request body for workflow optimizations"""
        sections = [
            "# Workflow Optimizations\n",
            "## Changes Summary\n"
        ]
        
        # Add concurrency changes
        if "concurrency" in suggestions:
            sections.append("### Concurrency Optimization\n")
            conc = suggestions["concurrency"]
            sections.append(
                f"- Current: {conc['current_concurrency']}\n"
                f"- Suggested: {conc['suggested_concurrency']}\n"
                f"- Reason: {conc['reason']}\n"
            )
            
        # Add caching changes
        if "caching" in suggestions:
            sections.append("### Caching Strategy\n")
            cache = suggestions["caching"]
            sections.append(
                f"- Cache paths: {', '.join(cache['cache_paths'])}\n"
                f"- Estimated savings: {cache['estimated_savings']}s\n"
            )
            
        # Add job ordering changes
        if "job_order" in suggestions:
            sections.append("### Job Ordering\n")
            order = suggestions["job_order"]
            sections.append(
                f"- Parallel jobs: {', '.join(order['parallel_jobs'])}\n"
                f"- Estimated improvement: {order['estimated_improvement']}s\n"
            )
            
        # Add resource allocation changes
        if "resource_allocation" in suggestions:
            sections.append("### Resource Allocation\n")
            resources = suggestions["resource_allocation"]
            sections.append(
                f"- CPU: {resources['cpu_allocation']}\n"
                f"- Memory: {resources['memory_allocation']}\n"
                f"- Estimated cost savings: ${resources['estimated_cost_savings']:.2f}\n"
            )
            
        return "\n".join(sections)
        
    async def _update_workflow_config(
        self,
        workflow_data: Dict[str, Any],
        suggestions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update workflow configuration with optimizations"""
        # Update concurrency
        if "concurrency" in suggestions:
            workflow_data["concurrency"] = {
                "group": "${{ github.workflow }}",
                "cancel-in-progress": True
            }
            
        # Update job configurations
        if "jobs" in workflow_data:
            for job_name, job_config in workflow_data["jobs"].items():
                # Update caching
                if "caching" in suggestions:
                    cache_paths = suggestions["caching"]["cache_paths"]
                    if cache_paths:
                        job_config["steps"].insert(1, {
                            "uses": "actions/cache@v2",
                            "with": {
                                "path": "\n".join(cache_paths),
                                "key": suggestions["caching"]["cache_key_strategy"]
                            }
                        })
                        
                # Update resource allocation
                if "resource_allocation" in suggestions:
                    resources = suggestions["resource_allocation"]
                    job_config["runs-on"] = self._get_runner_label(resources)
                    
        return workflow_data
        
    def _get_runner_label(self, resources: Dict[str, Any]) -> str:
        """Get appropriate runner label based on resource requirements"""
        cpu = resources["cpu_allocation"]
        memory = resources["memory_allocation"]
        
        if cpu <= 2 and memory <= 7:
            return "ubuntu-latest"
        elif cpu <= 4 and memory <= 16:
            return "ubuntu-latest-4-core"
        else:
            return "ubuntu-latest-8-core"
            
    async def _update_pr_metrics(self, pr: PullRequest) -> None:
        """Update pull request metrics"""
        metrics_key = f"pr_metrics_{pr.number}"
        
        metrics = {
            "created_at": pr.created_at.isoformat(),
            "updated_at": pr.updated_at.isoformat(),
            "status": self._get_pr_status(pr),
            "review_time": self._calculate_review_time(pr),
            "changes": {
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changed_files": pr.changed_files
            }
        }
        
        self._metrics[metrics_key] = metrics
        
    def _get_pr_status(self, pr: PullRequest) -> PullRequestStatus:
        """Get detailed pull request status"""
        if pr.merged:
            return PullRequestStatus.MERGED
        elif pr.state == "closed":
            return PullRequestStatus.CLOSED
        elif pr.draft:
            return PullRequestStatus.DRAFT
        elif pr.mergeable_state == "dirty":
            return PullRequestStatus.CONFLICT
        elif pr.reviews:
            return PullRequestStatus.REVIEW
        else:
            return PullRequestStatus.OPEN
            
    def _calculate_review_time(self, pr: PullRequest) -> Optional[float]:
        """Calculate time spent in review"""
        if not pr.reviews:
            return None
            
        first_review = min(r.submitted_at for r in pr.reviews)
        if pr.merged:
            end_time = pr.merged_at
        elif pr.state == "closed":
            end_time = pr.closed_at
        else:
            end_time = datetime.now()
            
        return (end_time - first_review).total_seconds()
        
    def _identify_cache_paths(
        self,
        metrics: WorkflowMetrics
    ) -> List[str]:
        """Identify paths that should be cached"""
        cache_paths = []
        
        # Add common cache paths based on workflow patterns
        if any("npm" in b for b in metrics.bottlenecks):
            cache_paths.append("~/.npm")
            cache_paths.append("node_modules")
            
        if any("pip" in b for b in metrics.bottlenecks):
            cache_paths.append("~/.cache/pip")
            cache_paths.append(".venv")
            
        if any("gradle" in b for b in metrics.bottlenecks):
            cache_paths.append("~/.gradle/caches")
            cache_paths.append("~/.gradle/wrapper")
            
        return cache_paths
        
    def _generate_cache_keys(
        self,
        metrics: WorkflowMetrics
    ) -> str:
        """Generate optimal cache key strategy"""
        return "${{ runner.os }}-${{ hashFiles('**/lockfiles') }}"
        
    def _estimate_cache_savings(
        self,
        metrics: WorkflowMetrics
    ) -> float:
        """Estimate time savings from caching"""
        # Estimate based on historical timing data
        return metrics.avg_duration * 0.2  # Assume 20% improvement
        
    def _identify_parallel_jobs(
        self,
        metrics: WorkflowMetrics
    ) -> List[str]:
        """Identify jobs that can run in parallel"""
        return [b for b in metrics.bottlenecks if "test" in b.lower()]
        
    def _optimize_dependencies(
        self,
        metrics: WorkflowMetrics
    ) -> Dict[str, List[str]]:
        """Optimize job dependencies"""
        return {"build": [], "test": ["build"], "deploy": ["test"]}
        
    def _estimate_ordering_improvement(
        self,
        metrics: WorkflowMetrics
    ) -> float:
        """Estimate improvement from job reordering"""
        return metrics.avg_duration * 0.15  # Assume 15% improvement
        
    def _optimize_cpu_allocation(
        self,
        metrics: WorkflowMetrics
    ) -> int:
        """Optimize CPU allocation"""
        usage = metrics.resource_usage.get("cpu", 0)
        return max(2, min(8, int(usage * 1.2)))  # Add 20% buffer
        
    def _optimize_memory_allocation(
        self,
        metrics: WorkflowMetrics
    ) -> int:
        """Optimize memory allocation"""
        usage = metrics.resource_usage.get("memory", 0)
        return max(4, min(16, int(usage * 1.2)))  # Add 20% buffer
        
    def _estimate_resource_savings(
        self,
        metrics: WorkflowMetrics
    ) -> float:
        """Estimate cost savings from resource optimization"""
        current_cost = self._calculate_resource_cost(
            metrics.resource_usage.get("cpu", 0),
            metrics.resource_usage.get("memory", 0),
            metrics.avg_duration
        )
        
        optimized_cost = self._calculate_resource_cost(
            self._optimize_cpu_allocation(metrics),
            self._optimize_memory_allocation(metrics),
            metrics.avg_duration
        )
        
        return current_cost - optimized_cost
        
    def _calculate_resource_cost(
        self,
        cpu: float,
        memory: float,
        duration: float
    ) -> float:
        """Calculate cost for resource usage"""
        # Simple cost model: $0.008 per CPU minute, $0.001 per GB minute
        cpu_cost = (cpu / 100) * 0.008 * (duration / 60)
        memory_cost = (memory / 1024) * 0.001 * (duration / 60)
        return cpu_cost + memory_cost
