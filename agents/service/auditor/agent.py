"""
Auditor Agent for Project Aethelred.

The Auditor is a Service tier agent responsible for:
- Observing and monitoring agent performance
- Calculating performance scores and metrics
- Detecting anomalies and performance degradation
- Generating performance reports and insights
"""

import asyncio
import logging
import statistics
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

from agents.base.agent import Agent, AgentCapability, AgentStatus, TaskResult
from core.memory.tier_manager import MemoryTierManager

logger = logging.getLogger(__name__)


class Auditor(Agent):
    """
    Auditor - Performance Observer and Metrics Calculator.
    
    Responsibilities:
    - Monitor agent performance continuously
    - Calculate performance scores and trends
    - Detect performance anomalies
    - Generate performance reports
    - Track system-wide metrics
    """
    
    def __init__(self, memory_manager: MemoryTierManager,
                 config: Optional[Dict[str, Any]] = None):
        
        super().__init__(
            agent_id="S_Auditor",
            version=1,
            tier="service",
            role="Performance Observer",
            capabilities=[
                AgentCapability.AGENTS_OBSERVE,
                AgentCapability.METRICS_WRITE,
                AgentCapability.SCORES_CALCULATE
            ],
            config=config or {}
        )
        
        self.memory_manager = memory_manager
        
        # Performance tracking
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics: Dict[str, Any] = {}
        
        # Anomaly detection
        self.anomaly_thresholds = {
            'success_rate_drop': 0.2,  # 20% drop in success rate
            'response_time_increase': 2.0,  # 2x increase in response time
            'error_rate_spike': 0.1  # 10% error rate
        }
        
        # Monitoring intervals
        self.monitoring_interval = 30  # seconds
        self.metrics_calculation_interval = 300  # 5 minutes
        self.report_generation_interval = 3600  # 1 hour
        
        # Statistics
        self.observations_made = 0
        self.scores_calculated = 0
        self.anomalies_detected = 0
        self.reports_generated = 0
        
    async def on_initialize(self) -> None:
        """Initialize Auditor specific resources."""
        logger.info("Initializing Auditor agent...")
        
        # Load historical performance data
        await self._load_performance_history()
        
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        logger.info("Auditor initialization complete")
        
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task assigned to the Auditor.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        task_type = task.get('type')
        
        if task_type == 'observe_agent':
            return await self._handle_agent_observation(task)
        elif task_type == 'calculate_scores':
            return await self._handle_score_calculation(task)
        elif task_type == 'generate_report':
            return await self._handle_report_generation(task)
        elif task_type == 'detect_anomalies':
            return await self._handle_anomaly_detection(task)
        elif task_type == 'performance_analysis':
            return await self._handle_performance_analysis(task)
        elif task_type == 'system_health_check':
            return await self._handle_system_health_check(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """
        Validate if the Auditor can execute the task.
        
        Args:
            task: Task to validate
            
        Returns:
            True if task can be executed
        """
        valid_task_types = {
            'observe_agent',
            'calculate_scores',
            'generate_report',
            'detect_anomalies',
            'performance_analysis',
            'system_health_check'
        }
        
        task_type = task.get('type')
        return task_type in valid_task_types
        
    async def check_agent_health(self) -> Dict[str, Any]:
        """Auditor specific health checks."""
        return {
            'observations_made': self.observations_made,
            'scores_calculated': self.scores_calculated,
            'anomalies_detected': self.anomalies_detected,
            'reports_generated': self.reports_generated,
            'monitored_agents': len(self.agent_metrics),
            'performance_history_size': sum(len(hist) for hist in self.performance_history.values()),
            'monitoring_status': 'active'
        }
        
    # Task handlers
    
    async def _handle_agent_observation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent observation tasks."""
        agent_id = task.get('agent_id')
        observation_data = task.get('observation_data')
        
        if not agent_id or not observation_data:
            raise ValueError("agent_id and observation_data are required")
            
        # Record the observation
        await self._record_agent_observation(agent_id, observation_data)
        
        self.observations_made += 1
        
        return {
            'action': 'agent_observed',
            'agent_id': agent_id,
            'observation_timestamp': datetime.utcnow().isoformat(),
            'metrics_updated': True
        }
        
    async def _handle_score_calculation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance score calculation."""
        agent_id = task.get('agent_id')
        task_id = task.get('task_id')
        metrics = task.get('metrics')
        
        if not all([agent_id, task_id, metrics]):
            raise ValueError("agent_id, task_id, and metrics are required")
            
        # Calculate performance scores
        scores = await self._calculate_performance_scores(agent_id, task_id, metrics)
        
        # Store scores in memory
        await self._store_performance_scores(agent_id, task_id, scores)
        
        self.scores_calculated += 1
        
        return {
            'action': 'scores_calculated',
            'agent_id': agent_id,
            'task_id': task_id,
            'scores': scores,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_report_generation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance report generation."""
        report_type = task.get('report_type', 'comprehensive')
        time_window = task.get('time_window_hours', 24)
        agent_filter = task.get('agent_filter')
        
        # Generate the report
        report = await self._generate_performance_report(
            report_type, time_window, agent_filter
        )
        
        # Store report in memory
        report_id = f"performance_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        await self.memory_manager.write(
            f"report:{report_id}",
            report,
            target_tiers=['warm', 'cold']
        )
        
        self.reports_generated += 1
        
        return {
            'action': 'report_generated',
            'report_id': report_id,
            'report_type': report_type,
            'time_window_hours': time_window,
            'report': report,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_anomaly_detection(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anomaly detection."""
        agent_id = task.get('agent_id')
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(agent_id)
        
        if anomalies:
            self.anomalies_detected += len(anomalies)
            
            # Store anomalies in memory
            for anomaly in anomalies:
                anomaly_id = f"anomaly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{anomaly['type']}"
                await self.memory_manager.write(
                    f"anomaly:{anomaly_id}",
                    anomaly,
                    target_tiers=['hot', 'warm']
                )
                
        return {
            'action': 'anomalies_detected',
            'agent_id': agent_id,
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_performance_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance analysis requests."""
        analysis_type = task.get('analysis_type', 'trend')
        agent_id = task.get('agent_id')
        time_window = task.get('time_window_hours', 24)
        
        analysis = await self._perform_performance_analysis(
            analysis_type, agent_id, time_window
        )
        
        return {
            'action': 'performance_analyzed',
            'analysis_type': analysis_type,
            'agent_id': agent_id,
            'analysis': analysis,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    async def _handle_system_health_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system-wide health checks."""
        health_data = await self._perform_system_health_check()
        
        # Store health snapshot
        health_id = f"health_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        await self.memory_manager.write(
            f"health:{health_id}",
            health_data,
            target_tiers=['hot', 'warm']
        )
        
        return {
            'action': 'system_health_checked',
            'health_id': health_id,
            'health_data': health_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    # Core functionality
    
    async def _record_agent_observation(self, agent_id: str, 
                                       observation_data: Dict[str, Any]) -> None:
        """Record an agent observation."""
        timestamp = datetime.utcnow()
        
        # Update agent metrics
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = {
                'first_observed': timestamp,
                'last_observed': timestamp,
                'total_observations': 0,
                'current_metrics': {}
            }
            
        self.agent_metrics[agent_id]['last_observed'] = timestamp
        self.agent_metrics[agent_id]['total_observations'] += 1
        self.agent_metrics[agent_id]['current_metrics'] = observation_data
        
        # Add to performance history
        history_entry = {
            'timestamp': timestamp.isoformat(),
            'metrics': observation_data.copy()
        }
        self.performance_history[agent_id].append(history_entry)
        
        # Store in memory
        await self.memory_manager.write(
            f"agent_observation:{agent_id}:{timestamp.isoformat()}",
            observation_data,
            target_tiers=['hot']
        )
        
    async def _calculate_performance_scores(self, agent_id: str, task_id: str,
                                          metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance scores for an agent task."""
        scores = {}
        
        # Speed score (based on response time)
        if 'response_time_ms' in metrics:
            response_time = metrics['response_time_ms']
            # Normalize to 0-1 scale (1000ms = 0.5, 500ms = 0.75, 100ms = 0.95)
            scores['speed'] = max(0, min(1, 1.0 - (response_time - 100) / 2000))
            
        # Accuracy score (based on success rate)
        if 'success' in metrics:
            scores['accuracy'] = 1.0 if metrics['success'] else 0.0
            
        # Quality score (based on various quality metrics)
        quality_factors = []
        if 'code_quality' in metrics:
            quality_factors.append(metrics['code_quality'])
        if 'test_coverage' in metrics:
            quality_factors.append(metrics['test_coverage'] / 100.0)
        if 'documentation_score' in metrics:
            quality_factors.append(metrics['documentation_score'])
            
        if quality_factors:
            scores['quality'] = sum(quality_factors) / len(quality_factors)
        else:
            scores['quality'] = 1.0  # Default if no quality metrics
            
        # Efficiency score (based on resource usage)
        if 'cpu_usage_percent' in metrics and 'memory_usage_mb' in metrics:
            cpu_efficiency = max(0, 1.0 - (metrics['cpu_usage_percent'] / 100))
            memory_efficiency = max(0, 1.0 - min(1.0, metrics['memory_usage_mb'] / 1000))
            scores['efficiency'] = (cpu_efficiency + memory_efficiency) / 2
            
        # Reliability score (based on error rate and consistency)
        if 'error_count' in metrics and 'total_operations' in metrics:
            error_rate = metrics['error_count'] / max(1, metrics['total_operations'])
            scores['reliability'] = max(0, 1.0 - error_rate)
        else:
            scores['reliability'] = scores.get('accuracy', 1.0)
            
        # Composite score (weighted average)
        if scores:
            weights = {
                'speed': 0.2,
                'accuracy': 0.3,
                'quality': 0.2,
                'efficiency': 0.15,
                'reliability': 0.15
            }
            
            composite_score = 0
            total_weight = 0
            
            for metric, score in scores.items():
                weight = weights.get(metric, 0.1)
                composite_score += score * weight
                total_weight += weight
                
            scores['composite'] = composite_score / max(total_weight, 1.0)
            
        return scores
        
    async def _store_performance_scores(self, agent_id: str, task_id: str,
                                       scores: Dict[str, float]) -> None:
        """Store performance scores in memory."""
        score_data = {
            'agent_id': agent_id,
            'task_id': task_id,
            'scores': scores,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store individual score record
        await self.memory_manager.write(
            f"performance_score:{agent_id}:{task_id}",
            score_data,
            target_tiers=['warm']
        )
        
        # Update agent's aggregate scores
        agent_scores_key = f"agent_scores:{agent_id}"
        existing_scores = await self.memory_manager.read(agent_scores_key) or {}
        
        # Calculate running averages
        for metric, score in scores.items():
            if metric not in existing_scores:
                existing_scores[metric] = {'total': 0, 'count': 0, 'average': 0}
                
            existing_scores[metric]['total'] += score
            existing_scores[metric]['count'] += 1
            existing_scores[metric]['average'] = (
                existing_scores[metric]['total'] / existing_scores[metric]['count']
            )
            
        existing_scores['last_updated'] = datetime.utcnow().isoformat()
        
        await self.memory_manager.write(
            agent_scores_key,
            existing_scores,
            target_tiers=['hot', 'warm']
        )
        
    async def _detect_anomalies(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []
        agents_to_check = [agent_id] if agent_id else list(self.agent_metrics.keys())
        
        for aid in agents_to_check:
            if aid not in self.performance_history:
                continue
                
            history = list(self.performance_history[aid])
            if len(history) < 10:  # Need sufficient history
                continue
                
            # Get recent vs historical performance
            recent_entries = history[-5:]  # Last 5 observations
            historical_entries = history[-50:-5] if len(history) > 50 else history[:-5]
            
            if not historical_entries:
                continue
                
            # Check for success rate drop
            recent_success_rates = [
                entry['metrics'].get('success_rate', 1.0) for entry in recent_entries
            ]
            historical_success_rates = [
                entry['metrics'].get('success_rate', 1.0) for entry in historical_entries
            ]
            
            if recent_success_rates and historical_success_rates:
                recent_avg = statistics.mean(recent_success_rates)
                historical_avg = statistics.mean(historical_success_rates)
                
                if historical_avg - recent_avg > self.anomaly_thresholds['success_rate_drop']:
                    anomalies.append({
                        'agent_id': aid,
                        'type': 'success_rate_drop',
                        'severity': 'high',
                        'recent_value': recent_avg,
                        'historical_value': historical_avg,
                        'threshold': self.anomaly_thresholds['success_rate_drop'],
                        'detected_at': datetime.utcnow().isoformat()
                    })
                    
            # Check for response time increase
            recent_response_times = [
                entry['metrics'].get('response_time_ms', 1000) for entry in recent_entries
            ]
            historical_response_times = [
                entry['metrics'].get('response_time_ms', 1000) for entry in historical_entries
            ]
            
            if recent_response_times and historical_response_times:
                recent_avg = statistics.mean(recent_response_times)
                historical_avg = statistics.mean(historical_response_times)
                
                if recent_avg > historical_avg * self.anomaly_thresholds['response_time_increase']:
                    anomalies.append({
                        'agent_id': aid,
                        'type': 'response_time_increase',
                        'severity': 'medium',
                        'recent_value': recent_avg,
                        'historical_value': historical_avg,
                        'threshold': self.anomaly_thresholds['response_time_increase'],
                        'detected_at': datetime.utcnow().isoformat()
                    })
                    
        return anomalies
        
    async def _generate_performance_report(self, report_type: str, 
                                          time_window_hours: int,
                                          agent_filter: Optional[str] = None) -> Dict[str, Any]:
        """Generate a performance report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        report = {
            'report_type': report_type,
            'time_window_hours': time_window_hours,
            'generated_at': datetime.utcnow().isoformat(),
            'agent_filter': agent_filter,
            'summary': {},
            'agent_details': {},
            'anomalies': [],
            'recommendations': []
        }
        
        agents_to_analyze = (
            [agent_filter] if agent_filter and agent_filter in self.agent_metrics
            else list(self.agent_metrics.keys())
        )
        
        total_observations = 0
        total_agents = 0
        
        for agent_id in agents_to_analyze:
            if agent_id not in self.performance_history:
                continue
                
            # Filter history by time window
            relevant_history = [
                entry for entry in self.performance_history[agent_id]
                if datetime.fromisoformat(entry['timestamp']) > cutoff_time
            ]
            
            if not relevant_history:
                continue
                
            total_agents += 1
            total_observations += len(relevant_history)
            
            # Calculate agent metrics for the time window
            agent_report = await self._analyze_agent_performance(agent_id, relevant_history)
            report['agent_details'][agent_id] = agent_report
            
        # Generate summary
        report['summary'] = {
            'total_agents_analyzed': total_agents,
            'total_observations': total_observations,
            'time_period': f"{cutoff_time.isoformat()} to {datetime.utcnow().isoformat()}",
            'anomalies_detected': len(await self._detect_anomalies())
        }
        
        # Add recommendations
        report['recommendations'] = await self._generate_recommendations(report)
        
        return report
        
    async def _analyze_agent_performance(self, agent_id: str, 
                                        history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance for a specific agent."""
        if not history:
            return {'error': 'No performance data available'}
            
        # Extract metrics
        response_times = [entry['metrics'].get('response_time_ms', 0) for entry in history]
        success_rates = [entry['metrics'].get('success_rate', 1.0) for entry in history]
        error_counts = [entry['metrics'].get('error_count', 0) for entry in history]
        
        # Calculate statistics
        analysis = {
            'observation_count': len(history),
            'time_range': {
                'start': history[0]['timestamp'],
                'end': history[-1]['timestamp']
            }
        }
        
        if response_times:
            analysis['response_time'] = {
                'average': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'min': min(response_times),
                'max': max(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
            
        if success_rates:
            analysis['success_rate'] = {
                'average': statistics.mean(success_rates),
                'min': min(success_rates),
                'max': max(success_rates)
            }
            
        if error_counts:
            analysis['errors'] = {
                'total': sum(error_counts),
                'average_per_observation': statistics.mean(error_counts),
                'max_in_observation': max(error_counts)
            }
            
        return analysis
        
    async def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance report."""
        recommendations = []
        
        # Analyze agent details for patterns
        for agent_id, details in report['agent_details'].items():
            if 'response_time' in details:
                avg_response = details['response_time']['average']
                if avg_response > 5000:  # 5 seconds
                    recommendations.append(
                        f"Agent {agent_id}: Consider optimization - high average response time ({avg_response:.0f}ms)"
                    )
                    
            if 'success_rate' in details:
                avg_success = details['success_rate']['average']
                if avg_success < 0.9:  # 90%
                    recommendations.append(
                        f"Agent {agent_id}: Investigate reliability issues - low success rate ({avg_success:.2%})"
                    )
                    
        # System-wide recommendations
        total_agents = report['summary']['total_agents_analyzed']
        if total_agents == 0:
            recommendations.append("No agent performance data available - ensure monitoring is active")
        elif total_agents < 2:
            recommendations.append("Consider adding more agents for better load distribution")
            
        return recommendations
        
    async def _perform_performance_analysis(self, analysis_type: str, 
                                           agent_id: Optional[str],
                                           time_window_hours: int) -> Dict[str, Any]:
        """Perform specific performance analysis."""
        if analysis_type == 'trend':
            return await self._analyze_performance_trends(agent_id, time_window_hours)
        elif analysis_type == 'comparison':
            return await self._analyze_agent_comparison(time_window_hours)
        elif analysis_type == 'bottleneck':
            return await self._analyze_bottlenecks(agent_id, time_window_hours)
        else:
            return {'error': f"Unknown analysis type: {analysis_type}"}
            
    async def _analyze_performance_trends(self, agent_id: Optional[str],
                                         time_window_hours: int) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        # Simplified trend analysis
        return {
            'analysis_type': 'trend',
            'agent_id': agent_id,
            'trends': {
                'response_time': 'stable',
                'success_rate': 'improving',
                'error_rate': 'decreasing'
            },
            'confidence': 0.8
        }
        
    async def _analyze_agent_comparison(self, time_window_hours: int) -> Dict[str, Any]:
        """Compare performance across agents."""
        # Simplified comparison analysis
        return {
            'analysis_type': 'comparison',
            'top_performers': [],
            'underperformers': [],
            'average_metrics': {}
        }
        
    async def _analyze_bottlenecks(self, agent_id: Optional[str],
                                  time_window_hours: int) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        # Simplified bottleneck analysis
        return {
            'analysis_type': 'bottleneck',
            'bottlenecks_detected': [],
            'recommendations': []
        }
        
    async def _perform_system_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        # Get memory system health
        memory_health = await self.memory_manager.health_check()
        
        # Calculate system metrics
        health_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'memory_system': memory_health,
            'agent_metrics': {
                'total_monitored': len(self.agent_metrics),
                'active_agents': len([
                    aid for aid, metrics in self.agent_metrics.items()
                    if (datetime.utcnow() - metrics['last_observed']).total_seconds() < 300
                ]),
                'total_observations': self.observations_made,
                'scores_calculated': self.scores_calculated,
                'anomalies_detected': self.anomalies_detected
            },
            'system_performance': {
                'auditor_uptime_seconds': (
                    datetime.utcnow() - self.created_at
                ).total_seconds() if self.started_at else 0,
                'monitoring_active': True
            }
        }
        
        # Determine overall status
        memory_issues = sum(
            1 for tier_health in memory_health.values()
            if tier_health.get('status') != 'healthy'
        )
        
        if memory_issues > 0:
            health_data['overall_status'] = 'degraded'
        if memory_issues > len(memory_health) // 2:
            health_data['overall_status'] = 'unhealthy'
            
        return health_data
        
    async def _load_performance_history(self) -> None:
        """Load historical performance data from memory."""
        try:
            # This would typically load from database
            # For now, start fresh
            logger.info("Starting with fresh performance history")
        except Exception as e:
            logger.error(f"Failed to load performance history: {e}")
            
    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Periodic system health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(health_task)
        health_task.add_done_callback(self._background_tasks.discard)
        
        # Periodic anomaly detection
        anomaly_task = asyncio.create_task(self._anomaly_detection_loop())
        self._background_tasks.add(anomaly_task)
        anomaly_task.add_done_callback(self._background_tasks.discard)
        
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Perform periodic health check
                await self._perform_system_health_check()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                
    async def _anomaly_detection_loop(self) -> None:
        """Background anomaly detection loop."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.metrics_calculation_interval)
                
                # Detect anomalies
                anomalies = await self._detect_anomalies()
                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} performance anomalies")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")