import collections
from pathlib import Path
from typing import List, Dict, Tuple
import tiktoken
import logging

from atlas.semantic_engine import RepoGraph, create_skeleton_from_source
from .embeddings import EmbeddingManager # Assuming EmbeddingManager is in embeddings.py

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages context window for LLM queries using the
    "Anchor & Expand" strategy from the roadmap.
    """
    
    def __init__(
        self,
        repo_graph: RepoGraph,
        embedding_manager: EmbeddingManager,
        model: str = "gpt-4",
        max_tokens: int = 100000
    ):
        self.repo_graph = repo_graph
        self.embedding_manager = embedding_manager
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        logger.info(f"ContextManager initialized with model '{model}', max_tokens={max_tokens}")
    
    def assemble_context(
        self,
        user_query: str,
        files_in_scope: List[Path],
        include_map: bool = True
    ) -> str:
        """
        Build context using intelligent three-tier budgeting:
        
        Tier 1 (5%): Repository Map (always first)
        Tier 2 (50%): Neighborhood Files (full content)
        Tier 3 (45%): Architectural Context (skeletons of high-PageRank files)
        """
        context_parts = []
        total_budget = self.max_tokens
        
        # Reserve for system prompt and query
        total_budget -= self.count_tokens(user_query)
        total_budget -= 1000  # System prompt overhead
        
        processed_files = set()
        
        # === TIER 1: Repository Map (5% of budget) ===
        map_budget = int(total_budget * 0.05)
        if include_map:
            map_text = self.repo_graph.generate_map(max_files=50)
            map_tokens = self.count_tokens(map_text)
            
            if map_tokens <= map_budget:
                context_parts.append(("REPOSITORY_MAP", map_text))
                logger.debug(f"Added repo map ({map_tokens} tokens)")
            else:
                # Truncate map if too large
                truncated_map = self._truncate_to_tokens(map_text, map_budget)
                context_parts.append(("REPOSITORY_MAP (truncated)", truncated_map))
                logger.debug(f"Added truncated repo map ({map_budget} tokens)")
        
        remaining_budget = total_budget - map_budget
        
        # === TIER 2: Neighborhood Files - FULL CONTENT (50% of total) ===
        neighborhood_budget = int(total_budget * 0.50)
        
        # 2a. Explicit files in scope (highest priority)
        explicit_content, explicit_processed = self._fill_with_content(
            files_in_scope,
            neighborhood_budget,
            processed_files,
            is_skeleton=False
        )
        if explicit_content:
            context_parts.append(("EXPLICIT_FILES (full content)", explicit_content))
            processed_files.update(explicit_processed)
            neighborhood_budget -= self.count_tokens(explicit_content)
        
        # 2b. Vector search anchors
        all_graph_files = [Path(p) for p, _ in self.repo_graph.get_top_ranked_files(1000)]
        anchor_files = self.embedding_manager.find_relevant_files(
            user_query,
            all_graph_files,
            top_n=5
        )
        
        anchor_content, anchor_processed = self._fill_with_content(
            anchor_files,
            neighborhood_budget,
            processed_files,
            is_skeleton=False
        )
        if anchor_content:
            context_parts.append(("ANCHOR_FILES (full content)", anchor_content))
            processed_files.update(anchor_processed)
            neighborhood_budget -= self.count_tokens(anchor_content)
        
        # 2c. Neighborhood expansion (dependencies + dependents)
        expansion_base = list(files_in_scope) + anchor_files
        weighted_neighborhood = self._get_dependency_neighborhood(expansion_base, processed_files)
        # Extract paths in weight-sorted order (already sorted by _get_dependency_neighborhood)
        neighborhood_paths = [path for path, _weight in weighted_neighborhood]

        neighborhood_content, neighborhood_processed = self._fill_with_content(
            neighborhood_paths,
            neighborhood_budget,
            processed_files,
            is_skeleton=False
        )
        if neighborhood_content:
            context_parts.append(("NEIGHBORHOOD_FILES (full content)", neighborhood_content))
            processed_files.update(neighborhood_processed)
        
        # === TIER 3: Architectural Context - SKELETONS (45% of total) ===
        skeleton_budget = int(total_budget * 0.45)
        
        top_ranked = self.repo_graph.get_top_ranked_files(100)
        skeleton_content, skeleton_processed = self._fill_with_content(
            [Path(p) for p, _ in top_ranked],
            skeleton_budget,
            processed_files,
            is_skeleton=True
        )
        if skeleton_content:
            context_parts.append(("ARCHITECTURAL_CONTEXT (skeletons)", skeleton_content))
            processed_files.update(skeleton_processed)
        
        final_budget = total_budget - sum(self.count_tokens(c[1]) for c in context_parts)
        logger.info(f"Context assembled: {len(processed_files)} files, {final_budget} tokens remaining")
        
        return self._format_context(context_parts)

    def _get_dependency_neighborhood(
        self,
        anchor_files: List[Path],
        already_processed: set,
        max_hops: int = 2,
        max_files: int = 30
    ) -> List[Tuple[Path, float]]:
        """
        Build a neighborhood using multi-hop BFS with edge-type-aware traversal.

        Traversal rules:
        - SymbolUsage edges: follow for up to max_hops (real code dependencies)
        - Import edges: follow for 1 hop only (prevents transitive re-export explosion)

        Returns files sorted by weight descending, capped at max_files.
        Weight uses distance decay: hop 1 = 1.0, hop 2 = 0.5, hop 3 = 0.25, etc.
        """
        # file_path -> best (lowest hop) distance seen
        best_hop: Dict[Path, int] = {}
        # BFS queue: (file_path, current_hop, edge_kind_that_got_us_here)
        queue = collections.deque()

        for anchor in anchor_files:
            canonical = anchor.resolve()
            if canonical not in already_processed:
                queue.append((canonical, 0, None))

        visited_at_hop: Dict[Path, int] = {}
        for anchor in anchor_files:
            visited_at_hop[anchor.resolve()] = 0

        while queue:
            current_path, current_hop, arriving_edge = queue.popleft()

            if current_hop >= max_hops:
                continue

            next_hop = current_hop + 1
            canonical_str = str(current_path)

            # Expand both directions: dependencies and dependents
            neighbors = []
            try:
                neighbors.extend(self.repo_graph.get_dependencies(canonical_str))
            except Exception:
                pass
            try:
                neighbors.extend(self.repo_graph.get_dependents(canonical_str))
            except Exception:
                pass

            for neighbor_str, edge_kind in neighbors:
                neighbor_path = Path(neighbor_str)

                # Import edges: only follow at hop 0 -> 1 (1 hop max)
                if edge_kind == "Import" and current_hop >= 1:
                    continue

                # Skip if we already found this file at an equal or better hop
                if neighbor_path in visited_at_hop and visited_at_hop[neighbor_path] <= next_hop:
                    continue

                visited_at_hop[neighbor_path] = next_hop

                # Record best hop for weight calculation
                if neighbor_path not in best_hop or next_hop < best_hop[neighbor_path]:
                    best_hop[neighbor_path] = next_hop

                queue.append((neighbor_path, next_hop, edge_kind))

        # Remove anchors and already-processed files
        anchor_set = {a.resolve() for a in anchor_files}
        for remove_path in (already_processed | anchor_set):
            best_hop.pop(remove_path, None)

        # Compute weights with distance decay: hop n -> 1 / 2^(n-1)
        weighted: List[Tuple[Path, float]] = []
        for file_path, hop in best_hop.items():
            weight = 1.0 / (2 ** (hop - 1))
            weighted.append((file_path, weight))

        # Sort by weight descending, cap at max_files
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:max_files]

    def _fill_with_content(
        self,
        file_list: List[Path],
        max_tokens: int,
        already_processed: set[Path],
        is_skeleton: bool = True
    ) -> Tuple[str, set[Path]]:
        """
        Fill token budget with either full content or skeletons.
        
        Args:
            file_list: Files to include
            max_tokens: Token budget
            already_processed: Files already in context
            is_skeleton: If True, use skeletons; if False, use full content
        """
        content_parts = []
        token_count = 0
        processed_in_this_call = set()
        
        for file_path in file_list:
            if token_count >= max_tokens:
                break
            if file_path in already_processed:
                continue
                
            try:
                source = file_path.read_text()
                
                if is_skeleton:
                    content = self._extract_signatures(source, file_path.suffix[1:])
                    label = f"# {file_path} (SKELETON)"
                else:
                    content = source
                    label = f"# {file_path} (FULL CONTENT)"
                
                if not content.strip():
                    continue
                    
                formatted = f"\n{label}\n{content}\n"
                tokens = self.count_tokens(formatted)
                
                if token_count + tokens <= max_tokens:
                    content_parts.append(formatted)
                    token_count += tokens
                    processed_in_this_call.add(file_path)
                else:
                    # Truncate if necessary
                    remaining = max_tokens - token_count
                    if remaining > 100:  # Only add if meaningful space left
                        truncated = self._truncate_to_tokens(formatted, remaining)
                        content_parts.append(truncated)
                        token_count = max_tokens
                        processed_in_this_call.add(file_path)
                    break
                    
            except Exception as e:
                logger.debug(f"Could not process {file_path}: {e}")
                continue
        
        return "".join(content_parts), processed_in_this_call
    
    def _extract_signatures(self, content: str, lang_ext: str) -> str:
        """
        Extract function/class signatures from source code using the Rust core.
        """
        try:
            return create_skeleton_from_source(content, lang_ext)
        except Exception as e:
            logger.error(f"Error creating skeleton: {e}")
            # Fallback to a very simple signature extraction
            lines = content.split("\n")
            signatures = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def "):
                    signatures.append(line)
                elif stripped.startswith("fn "): # Rust
                    signatures.append(line)
            return "\n".join(signatures)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens) + "\n\n[... truncated ...]"
    
    def _format_context(self, parts: List[Tuple[str, str]]) -> str:
        """Format context parts into a single string."""
        formatted = []
        for label, content in parts:
            formatted.append(f"{'='*80}\n{label}\n{'='*80}\n{content}\n")
        return "\n".join(formatted)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))