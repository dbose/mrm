"""DAG (Directed Acyclic Graph) management for model dependencies"""

from typing import Dict, List, Set, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


def parse_model_dependencies(model_config: Dict) -> List[str]:
    """
    Parse model dependencies from config
    
    Args:
        model_config: Model configuration dict
    
    Returns:
        List of dependency model names
    """
    depends_on = model_config.get('model', {}).get('depends_on', [])
    
    result = []
    for dep in depends_on:
        if isinstance(dep, str):
            # Handle ref('name') syntax
            import re
            if dep.startswith('ref('):
                match = re.match(r"ref\(['\"]([^'\"]+)['\"]\)", dep)
                if match:
                    result.append(match.group(1))
            else:
                result.append(dep)
    
    return result


class ModelDAG:
    """
    Manages model dependency graph
    
    Similar to dbt's DAG, tracks dependencies between models
    and enables graph-based selection
    """
    
    def __init__(self):
        """Initialize empty DAG"""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependencies
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # node -> dependents
        self.metadata: Dict[str, Dict] = {}
    
    def add_node(self, node_name: str, metadata: Optional[Dict] = None):
        """
        Add a node to the DAG
        
        Args:
            node_name: Name of the model
            metadata: Optional metadata about the model
        """
        self.nodes.add(node_name)
        if metadata:
            self.metadata[node_name] = metadata
    
    def add_edge(self, from_node: str, to_node: str):
        """
        Add a dependency edge
        
        Args:
            from_node: Dependent model (downstream)
            to_node: Dependency model (upstream)
        
        Example:
            add_edge('expected_loss', 'pd_model')
            # expected_loss depends on pd_model
        """
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)
        
        # Ensure both nodes exist
        self.nodes.add(from_node)
        self.nodes.add(to_node)
    
    def get_dependencies(self, node: str) -> Set[str]:
        """
        Get direct dependencies of a node (upstream)
        
        Args:
            node: Model name
        
        Returns:
            Set of model names this model depends on
        """
        return self.edges.get(node, set())
    
    def get_dependents(self, node: str) -> Set[str]:
        """
        Get direct dependents of a node (downstream)
        
        Args:
            node: Model name
        
        Returns:
            Set of model names that depend on this model
        """
        return self.reverse_edges.get(node, set())
    
    def get_ancestors(self, node: str, include_self: bool = False) -> Set[str]:
        """
        Get all upstream dependencies (transitive closure)
        
        Args:
            node: Model name
            include_self: Whether to include the node itself
        
        Returns:
            Set of all upstream model names
        """
        visited = set()
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add dependencies to queue
            for dep in self.get_dependencies(current):
                if dep not in visited:
                    queue.append(dep)
        
        if not include_self:
            visited.discard(node)
        
        return visited
    
    def get_descendants(self, node: str, include_self: bool = False) -> Set[str]:
        """
        Get all downstream dependents (transitive closure)
        
        Args:
            node: Model name
            include_self: Whether to include the node itself
        
        Returns:
            Set of all downstream model names
        """
        visited = set()
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add dependents to queue
            for dep in self.get_dependents(current):
                if dep not in visited:
                    queue.append(dep)
        
        if not include_self:
            visited.discard(node)
        
        return visited
    
    def select_nodes(self, selector: str) -> Set[str]:
        """
        Select nodes using dbt-style graph operators
        
        Syntax:
            model          - Just the model
            +model         - Model and all ancestors (upstream)
            model+         - Model and all descendants (downstream)
            +model+        - Model, ancestors, and descendants
            @model         - Just the model (explicit)
            1+model        - Model and 1 level of ancestors
            model+2        - Model and 2 levels of descendants
        
        Args:
            selector: Selection string
        
        Returns:
            Set of selected model names
        """
        # Parse selector
        upstream_depth = None
        downstream_depth = None
        include_self = True
        
        # Check for upstream operator
        if selector.startswith('+'):
            upstream_depth = float('inf')
            selector = selector[1:]
        elif selector[0].isdigit():
            # Numeric depth
            i = 0
            while i < len(selector) and selector[i].isdigit():
                i += 1
            if i < len(selector) and selector[i] == '+':
                upstream_depth = int(selector[:i])
                selector = selector[i+1:]
        
        # Check for downstream operator
        if selector.endswith('+'):
            downstream_depth = float('inf')
            selector = selector[:-1]
        elif selector[-1].isdigit() and '+' in selector:
            # Find the + before digits
            i = len(selector) - 1
            while i >= 0 and selector[i].isdigit():
                i -= 1
            if i >= 0 and selector[i] == '+':
                downstream_depth = int(selector[i+1:])
                selector = selector[:i]
        
        # Check for @ (just the node)
        if selector.startswith('@'):
            selector = selector[1:]
            upstream_depth = None
            downstream_depth = None
        
        # Get the base node
        if selector not in self.nodes:
            logger.warning(f"Node '{selector}' not found in DAG")
            return set()
        
        selected = {selector}
        
        # Add upstream nodes
        if upstream_depth is not None:
            if upstream_depth == float('inf'):
                selected.update(self.get_ancestors(selector))
            else:
                selected.update(self._get_ancestors_depth(selector, upstream_depth))
        
        # Add downstream nodes
        if downstream_depth is not None:
            if downstream_depth == float('inf'):
                selected.update(self.get_descendants(selector))
            else:
                selected.update(self._get_descendants_depth(selector, downstream_depth))
        
        return selected
    
    def _get_ancestors_depth(self, node: str, depth: int) -> Set[str]:
        """Get ancestors up to specified depth"""
        if depth <= 0:
            return set()
        
        result = set()
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                deps = self.get_dependencies(n)
                next_level.update(deps)
                result.update(deps)
            
            if not next_level:
                break
            
            current_level = next_level
        
        return result
    
    def _get_descendants_depth(self, node: str, depth: int) -> Set[str]:
        """Get descendants up to specified depth"""
        if depth <= 0:
            return set()
        
        result = set()
        current_level = {node}
        
        for _ in range(depth):
            next_level = set()
            for n in current_level:
                deps = self.get_dependents(n)
                next_level.update(deps)
                result.update(deps)
            
            if not next_level:
                break
            
            current_level = next_level
        
        return result
    
    def topological_sort(self, nodes: Optional[Set[str]] = None) -> List[str]:
        """
        Return nodes in topological order (dependencies first)
        
        Args:
            nodes: Optional subset of nodes to sort (defaults to all)
        
        Returns:
            List of nodes in dependency order
        
        Raises:
            ValueError: If cycle detected
        """
        if nodes is None:
            nodes = self.nodes
        else:
            nodes = set(nodes)
        
        # Calculate in-degrees for selected nodes
        in_degree = {node: 0 for node in nodes}
        
        for node in nodes:
            for dep in self.get_dependencies(node):
                if dep in nodes:
                    in_degree[node] += 1
        
        # Start with nodes that have no dependencies
        queue = deque([node for node in nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Reduce in-degree for dependents
            for dependent in self.get_dependents(node):
                if dependent in nodes:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Check for cycles
        if len(result) != len(nodes):
            remaining = nodes - set(result)
            raise ValueError(f"Cycle detected in model dependencies: {remaining}")
        
        return result
    
    def validate(self) -> bool:
        """
        Validate the DAG
        
        Returns:
            True if valid (no cycles)
        
        Raises:
            ValueError: If cycle detected
        """
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False
    
    def get_execution_levels(self, nodes: Optional[Set[str]] = None) -> List[List[str]]:
        """
        Get nodes grouped by execution level (for parallel execution)
        
        Level 0: No dependencies
        Level 1: Only depends on Level 0
        Level 2: Only depends on Levels 0 and 1
        etc.
        
        Args:
            nodes: Optional subset of nodes
        
        Returns:
            List of lists, each inner list is models that can run in parallel
        """
        if nodes is None:
            nodes = self.nodes
        else:
            nodes = set(nodes)
        
        # Calculate levels
        levels: Dict[str, int] = {}
        
        # Process in topological order
        sorted_nodes = self.topological_sort(nodes)
        
        for node in sorted_nodes:
            # Level is max level of dependencies + 1
            deps = self.get_dependencies(node) & nodes
            
            if not deps:
                levels[node] = 0
            else:
                max_dep_level = max(levels[dep] for dep in deps)
                levels[node] = max_dep_level + 1
        
        # Group by level
        max_level = max(levels.values()) if levels else 0
        result = [[] for _ in range(max_level + 1)]
        
        for node, level in levels.items():
            result[level].append(node)
        
        return result
    
    @classmethod
    def from_model_configs(cls, model_configs: List[Dict]) -> 'ModelDAG':
        """
        Build DAG from model configurations
        
        Args:
            model_configs: List of model config dictionaries
        
        Returns:
            ModelDAG instance
        """
        dag = cls()
        
        # First pass: add all nodes
        for config in model_configs:
            model_name = config['model']['name']
            dag.add_node(model_name, metadata=config['model'])
        
        # Second pass: add edges
        for config in model_configs:
            model_name = config['model']['name']
            depends_on = parse_model_dependencies(config)
            
            for dependency in depends_on:
                dag.add_edge(model_name, dependency)
        
        # Validate
        if not dag.validate():
            raise ValueError("Model dependency graph contains cycles")
        
        return dag
    
    def visualize(self) -> str:
        """
        Create a simple text visualization of the DAG
        
        Returns:
            String representation
        """
        lines = ["Model Dependency Graph:", ""]
        
        # Show each node with its dependencies
        for node in sorted(self.nodes):
            deps = self.get_dependencies(node)
            
            if deps:
                lines.append(f"{node}")
                for dep in sorted(deps):
                    lines.append(f"  └─> {dep}")
            else:
                lines.append(f"{node} (no dependencies)")
        
        return "\n".join(lines)
